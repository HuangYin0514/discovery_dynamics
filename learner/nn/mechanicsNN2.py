# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/10 10:59 AM
@desc:
"""
import torch
from torch import nn

from ..integrator import ODESolver
from .base_module import LossNN
from .fnn import FNN
from torch import Tensor


class CosSin(nn.Module):
    def __init__(self, q_ndim, angular_dims, only_q=True):
        super().__init__()
        self.q_ndim = q_ndim
        self.angular_dims = tuple(angular_dims)
        self.non_angular_dims = tuple(set(range(q_ndim)) - set(angular_dims))
        self.only_q = only_q

    def forward(self, q_or_qother):
        if self.only_q:
            q = q_or_qother
        else:
            split_dim = q_or_qother.size(-1)
            splits = [self.q_ndim, split_dim - self.q_ndim]
            q, other = q_or_qother.split(splits, dim=-1)

        # print("\n")
        # print(q_or_qother.size())
        # print(q.size())
        # print(other.size())
        # print(self.q_ndim)
        # print("")
        assert q.size(-1) == self.q_ndim

        q_angular = q[..., self.angular_dims]
        q_not_angular = q[..., self.non_angular_dims]

        cos_ang_q, sin_ang_q = torch.cos(q_angular), torch.sin(q_angular)
        q = torch.cat([cos_ang_q, sin_ang_q, q_not_angular], dim=-1)

        if self.only_q:
            q_or_other = q
        else:
            q_or_other = torch.cat([q, other], dim=-1)

        return q_or_other


def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight, gain=0.5)
    return linear


def FCtanh(chin, chout, zero_bias=False, orthogonal_init=False):
    return nn.Sequential(
        Linear(chin, chout, zero_bias, orthogonal_init),
        nn.Tanh()
    )


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class MechanicsNN(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, dim, layers=3, width=30):
        super(MechanicsNN, self).__init__()

        self.dim = dim
        self.layers = layers
        self.width = width

        self.__init_modules()

    def __init_modules(self):
        q_dim = int(self.dim // 2)
        _obj = 2

        self.mass_net = nn.Sequential(
            CosSin(q_dim, range(_obj), only_q=True),
            FCtanh(4, 50, zero_bias=False, orthogonal_init=True),
            Linear(50, q_dim * q_dim, zero_bias=True, orthogonal_init=True),
            Reshape(-1, q_dim, q_dim)
        )

        self.dynamics_net = nn.Sequential(
            CosSin(q_dim, range(_obj), only_q=False),
            FCtanh(q_dim * 2 + (self.dim - q_dim), 50, zero_bias=False, orthogonal_init=True),
            Linear(50, q_dim, zero_bias=False, orthogonal_init=True),
            Reshape(-1, q_dim)
        )

    def tril_Minv(self, q):
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
        q, p = x.chunk(2, dim=-1)
        dq_dt = self.Minv(q).matmul(p.unsqueeze(-1)).squeeze(-1)
        dp_dt = self.dynamics_net(x)
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
        return dz_dt

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
