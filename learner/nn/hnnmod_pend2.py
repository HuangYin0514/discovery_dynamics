# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import numpy as np
import torch
from torch import nn, Tensor

from .base_module import LossNN
from .mlp import MLP
from .utils_nn import CosSinNet, ReshapeNet
from ..integrator import ODESolver
from ..utils import dfx, lazy_property


class MassNet(nn.Module):
    def __init__(self, q_dim, num_layers=3, hidden_dim=30):
        super(MassNet, self).__init__()

        self.cos_sin_net = CosSinNet()
        self.net = nn.Sequential(
            MLP(input_dim=q_dim * 2, hidden_dim=hidden_dim, output_dim=q_dim * q_dim, num_layers=num_layers,
                act=nn.Tanh),
            ReshapeNet(-1, q_dim, q_dim)
        )

    def forward(self, q):
        q = self.cos_sin_net(q)
        out = self.net(q)
        return out

class GlobalPositionTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalPositionTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, x_0):
        y = self.mlp(x) + x_0
        return y


class GlobalVelocityTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalVelocityTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, v, v0):
        y = self.mlp(x) * v + v0
        return y


class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x):
        y = self.mlp(x)
        return y


class HnnMod_pend2(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(HnnMod_pend2, self).__init__()

        self.obj = obj
        self.dim = dim
        self.dof = obj * dim

        self.global_dim = 2
        self.global_dof = obj * self.global_dim

        self.global4x = GlobalPositionTransform(input_dim=self.dim,
                                                hidden_dim=16,
                                                output_dim=self.global_dim,
                                                num_layers=1, act=nn.Tanh)
        self.global4v = GlobalVelocityTransform(input_dim=self.dim,
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

        self.mass_net = MassNet(q_dim=self.dof, num_layers=1, hidden_dim=200)


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


    @lazy_property
    def J(self):
        # [ 0, I]
        # [-I, 0]
        states_dim = self.obj * self.dim * 2
        d = int(states_dim / 2)
        res = np.eye(states_dim, k=d) - np.eye(states_dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)

    def forward(self, t, data):
        bs = data.size(0)
        x, p = torch.chunk(data, 2, dim=1)

        # dq_dt = v = Minv @ p
        v = self.Minv(x).matmul(p.unsqueeze(-1)).squeeze(-1)

        L, T, U = 0., 0., torch.zeros((x.shape[0], 1), dtype=self.Dtype, device=self.Device)

        x_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        v_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)

        x_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        v_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.obj):
            for j in range(i):
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += x_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
                v_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += v_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]

            x_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4x(
                x[:, (i) * self.dim: (i + 1) * self.dim],
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim])
            v_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4v(
                x[:, (i) * self.dim: (i + 1) * self.dim],
                v[:, (i) * self.dim: (i + 1) * self.dim],
                v_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim])

        # Calculate the potential energy for i-th element
        for i in range(self.obj):
            U += self.co1 * self.mass(self.Potential1(x_global[:, i * self.global_dim: (i + 1) * self.global_dim]))

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
                U += self.co2 * (0.5 * self.mass(self.Potential2(x_ij)) + 0.5 * self.mass(self.Potential2(x_ji)))

        # Calculate the kinetic energy for i-th element
        for i in range(self.obj):
            T += 0.5 * self.mass(
                v_global[:, (i) * self.global_dim: (i + 1) * self.global_dim].pow(2).sum(axis=1, keepdim=True))

        # Construct Lagrangian
        H = (T + U)

        gradH = dfx(H, data)
        dy = (self.J @ gradH.T).T  # dy shape is (bs, vector)
        return dy

    def integrate_fun(self, t, data):
        data = data.clone().detach()
        divmod_value = torch.sign(data[..., :int(data.shape[-1] // 2)]) * 2 * torch.pi  # pendulum
        data[..., :int(data.shape[-1] // 2)] %= divmod_value  # pendulum
        data = data.clone().detach().requires_grad_(True)
        res = self(t, data)
        return res

    def integrate(self, X, t):
        out = ODESolver(self.integrate_fun, X, t, method='rk4').permute(1, 0, 2)  # (T, D)
        return out