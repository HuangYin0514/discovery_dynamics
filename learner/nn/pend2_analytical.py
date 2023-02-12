# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/12 9:56 AM
@desc:
"""
import numpy as np

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
from .utils_nn import CosSinNet, ReshapeNet
from ..integrator import ODESolver
from ..utils import dfx


class Pend2_analytical(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(Pend2_analytical, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.mass = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.mass.weight)

    def M(self, x):
        N = self.obj
        M = torch.zeros((x.shape[0], N, N), device=x.device)
        for i in range(N):
            for k in range(N):
                m_sum = 0
                j = i if i >= k else k
                for tmp in range(j, N):
                    m_sum += 1
                M[:, i, k] = torch.cos(x[:, i] - x[:, k]) * m_sum
        return M

    def Minv(self, x):
        return torch.linalg.inv(self.M(x))

    def forward(self, t, coords):
        # __x, __p = torch.chunk(coords, 2, dim=-1)
        # __x = __x % (2 * torch.pi)
        # coords = torch.cat([__x, __p], dim=-1).clone().detach().requires_grad_(True)

        bs = coords.size(0)
        x, p = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # Calculate the potential energy for i-th element ------------------------------------------------------------
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = y - torch.cos(x[:, i])
            U = U + 9.8 * y

        # Calculate the kinetic --------------------------------------------------------------
        T = 0.
        # T = (0.5 * p.unsqueeze(1).bmm(self.Minv(x)).bmm(p.unsqueeze(-1))).squeeze(-1).squeeze(-1)
        # T = (0.5 * p@ self.Minv(x) @ p.T).squeeze(-1).squeeze(-1)
        T = torch.sum(self.Minv(x)).reshape(-1,) +p*0

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        H = U * 0 + T
        dqH = dfx(H.sum(), x)
        dpH = dfx(H.sum(), p)

        v_global = self.Minv(x).matmul(p.unsqueeze(-1)).squeeze(-1)

        # Calculate the Derivative ----------------------------------------------------------------
        dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            dq_dt[:, i * self.dim:(i + 1) * self.dim] = dpH[:, i * self.dim: (i + 1) * self.dim]
            dp_dt[:, i * self.dim:(i + 1) * self.dim] = -dqH[:, i * self.dim:(i + 1) * self.dim]
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        return dz_dt

    def integrate(self, X0, t):

        def angle_forward(t, coords):
            x, p = torch.chunk(coords, 2, dim=-1)
            new_x = x % (2 * torch.pi)
            new_coords = torch.cat([new_x, p], dim=-1).clone().detach().requires_grad_(True)
            return self(t, new_coords)

        # out = ODESolver(angle_forward, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        outlist = []
        for x0 in X0:
            x0 = x0.reshape(1, -1)
            out = ODESolver(angle_forward, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
            outlist.append(out)
        out = torch.cat(outlist, dim=0)
        return out
