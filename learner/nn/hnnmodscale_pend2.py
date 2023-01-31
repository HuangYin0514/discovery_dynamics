# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import torch
from torch import nn, Tensor

from ._base_module import LossNN
from .mlp import MLP
from .utils_nn import CosSinNet, ReshapeNet, Identity
from ..integrator import ODESolver
from ..utils import dfx


class GlobalPositionTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalPositionTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim*8, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, x_0):
        x = torch.cat([torch.sin(x), torch.cos(x), x, 2 * x, 4 * x, 8 * x, 16 * x, 32 * x],dim=1)
        y = self.mlp(x) + x_0
        return y


class MassNet(nn.Module):
    def __init__(self, q_dim, num_layers=3, hidden_dim=30):
        super(MassNet, self).__init__()

        self.cos_sin_net = CosSinNet()
        self.net = nn.Sequential(
            MLP(input_dim=q_dim*8, hidden_dim=hidden_dim, output_dim=q_dim * q_dim, num_layers=num_layers,
                act=nn.Tanh),
            ReshapeNet(-1, q_dim, q_dim)
        )

    def forward(self, x):
        x = torch.cat([torch.sin(x), torch.cos(x), x, 2 * x, 4 * x, 8 * x, 16 * x, 32 * x],dim=1)
        out = self.net(x)
        return out


class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        self.mlp = MLP(input_dim=input_dim*8, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x):
        x = torch.cat([torch.sin(x), torch.cos(x), x, 2 * x, 4 * x, 8 * x, 16 * x, 32 * x],dim=1)
        y = self.mlp(x)
        return y


class HnnModScale_pend2(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(HnnModScale_pend2, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.global_dim = 2
        self.global_dof = int(obj * self.global_dim)

        self.mass_net = MassNet(q_dim=self.dof, num_layers=1, hidden_dim=50)
        self.global4x = GlobalPositionTransform(input_dim=self.dim,
                                                hidden_dim=16,
                                                output_dim=self.global_dim,
                                                num_layers=1, act=nn.Tanh)
        self.Potential1 = PotentialEnergyCell(input_dim=self.global_dim,
                                              hidden_dim=50,
                                              output_dim=1,
                                              num_layers=1, act=nn.Tanh)
        self.Potential2 = PotentialEnergyCell(input_dim=self.global_dim * 2,
                                              hidden_dim=50,
                                              output_dim=1,
                                              num_layers=1, act=nn.Tanh)

        self.co1 = torch.nn.Parameter(torch.ones(1, dtype=self.Dtype, device=self.Device) * 0.5)
        self.co2 = torch.nn.Parameter(torch.ones(1, dtype=self.Dtype, device=self.Device) * 0.5)

        self.mass = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.mass.weight)

    def tril_Minv(self, q):
        """
        Computes the inverse of a matrix M^{-1}(q)
        But only get the lower triangle of the inverse matrix  M^{-1}(q)
        to get lower triangle of  M^{-1}(q)  = [x, 0]
                                               [x, x]
        """
        mass_net_q = self.mass_net(q)
        res = torch.triu(mass_net_q, diagonal=1)
        res = res + torch.diag_embed(
            torch.nn.functional.softplus(torch.diagonal(mass_net_q, dim1=-2, dim2=-1)),
            dim1=-2,
            dim2=-1,
        )
        res = res.transpose(-1, -2)  # Make lower triangular
        return res

    def Minv(self, q: Tensor, eps=1e-4) -> Tensor:
        """Compute the learned inverse mass matrix M^{-1}(q)
            M = LU
            M^{-1} = (LU)^{-1} = U^{-1} @ L^{-1}
            M^{-1}(q) = [x, 0] @ [x, x]
                        [x, x]   [0, x]
        Args:
            q: bs x D Tensor representing the position
        """
        assert q.ndim == 2
        lower_triangular = self.tril_Minv(q)
        assert lower_triangular.ndim == 3
        diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=q.dtype, device=q.device)
        Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1)) + diag_noise
        return Minv

    def forward(self, t, x):
        bs = x.size(0)
        x, p = x.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # position transformations ----------------------------------------------------------------
        x_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        x_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            for j in range(i):
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += x_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
            x_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4x(
                x[:, (i) * self.dim: (i + 1) * self.dim],
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim])

        # Calculate the potential energy for i-th element ------------------------------------------------------------
        U = 0.
        for i in range(self.obj):
            U += self.co1 * self.mass(
                self.Potential1(x_global[:, i * self.global_dim: (i + 1) * self.global_dim]))

        for i in range(self.obj):
            for j in range(i):
                x_ij = torch.cat(
                    [x_global[:, i * self.global_dim: (i + 1) * self.global_dim],
                     x_global[:, j * self.global_dim: (j + 1) * self.global_dim]],
                    dim=1)
                x_ji = torch.cat(
                    [x_global[:, j * self.global_dim: (j + 1) * self.global_dim],
                     x_global[:, i * self.global_dim: (i + 1) * self.global_dim]],
                    dim=1)
                U += self.co2 * (
                        0.5 * self.mass(self.Potential2(x_ij)) + 0.5 * self.mass(self.Potential2(x_ji)))

        # Calculate the kinetic --------------------------------------------------------------
        T = 0.
        T = (0.5 * p.unsqueeze(1) @ self.Minv(x) @ p.unsqueeze(-1)).squeeze(-1)

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        H = U + T
        dqH = dfx(H.sum(), x)
        dpH = dfx(H.sum(), p)

        v_global = self.Minv(x).matmul(p.unsqueeze(-1)).squeeze(-1)

        # Calculate the Derivative ----------------------------------------------------------------
        dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            # dq_dt[:, i * self.dim:(i + 1) * self.dim] = v_global[:, i * self.dim: (i + 1) * self.dim]
            dq_dt[:, i * self.dim:(i + 1) * self.dim] = dpH[:, i * self.dim: (i + 1) * self.dim]
            dp_dt[:, i * self.dim:(i + 1) * self.dim] = -dqH[:, i * self.dim:(i + 1) * self.dim]
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        return dz_dt

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D)
        return out
