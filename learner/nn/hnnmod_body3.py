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


class DynamicsNet(nn.Module):
    def __init__(self, q_dim, p_dim, num_layers=3, hidden_dim=30):
        super(DynamicsNet, self).__init__()
        self.cos_sin_net = CosSinNet()

        self.dynamics_net = nn.Sequential(
            MLP(input_dim=q_dim * 2 + p_dim, hidden_dim=hidden_dim, output_dim=p_dim, num_layers=num_layers,
                act=nn.Tanh),
            ReshapeNet(-1, p_dim)
        )

    def forward(self, q, p):
        q = self.cos_sin_net(q)
        x = torch.cat([q, p], dim=1)
        out = self.dynamics_net(x)
        return out


class HnnMod_body3(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(HnnMod_body3, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.mass_net = MassNet(q_dim=dim, num_layers=1, hidden_dim=50)
        self.dynamics_net = DynamicsNet(q_dim=dim, p_dim=dim, num_layers=2, hidden_dim=50)

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
        """
        Parameters
        ----------
        t : torch.Tensor
            Time.
        x : torch.Tensor
            State.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        assert (x.ndim == 2)

        bs = x.size(0)

        q, p = x.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.obj):
            Minv = self.Minv(q[:, i * self.dim:(i + 1) * self.dim])
            dq_dt[:, i * self.dim:(i + 1) * self.dim] = Minv.matmul(
                p[:, i * self.dim:(i + 1) * self.dim].unsqueeze(-1)).squeeze(-1)  # dq_dt = v = Minv @ p
            # dp_dt = A(q, v)
            dp_dt[:, i * self.dim:(i + 1) * self.dim] = self.dynamics_net(q[:, i * self.dim:(i + 1) * self.dim],
                                                                          dq_dt[:, i * self.dim:(i + 1) * self.dim])

        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
        return dz_dt

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
