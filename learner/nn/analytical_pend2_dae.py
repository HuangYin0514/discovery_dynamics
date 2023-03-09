# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/8 4:53 PM
@desc:
"""
import numpy as np
import torch

from ._base_module import LossNN
from ..integrator import ODESolver
from ..utils.common_utils import enable_grad, matrix_inv, dfx


class Analytical_pend2_dae(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(Analytical_pend2_dae, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.mass = torch.nn.Linear(1, 1, bias=False)

        self.m = [10, 10]
        self.g = 10

    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        x, v = coords.chunk(2, dim=-1)

        Minv = self.Minv(x)
        V = self.potential(x)

        # 约束 -------------------------------------------------------------------------------
        phi = self.phi_fun(x)
        phi_q = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_q[:, i] = dfx(phi[:, i:i + 1], x)
        phi_qq = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_qq[:, i] = dfx(phi_q[:, i:i + 1] @ v.unsqueeze(-1), x)

        # 右端项 -------------------------------------------------------------------------------
        F = -dfx(V, x)

        # 求解 lam ----------------------------------------------------------------
        L = phi_q @ Minv @ phi_q.permute(0, 2, 1)
        R = (phi_q @ Minv @ F.unsqueeze(-1) + phi_qq @ v.unsqueeze(-1))  # (2, 1)
        lam = torch.linalg.solve(L, R)  # (2, 1)

        return torch.cat([v, F.squeeze(-1)], dim=-1)

    def Minv(self, q):
        bs, states = q.shape
        m = [10, 10]
        M = np.diag(np.array([m[0], m[0], m[1], m[1]]))
        M = np.tile(M, (bs, 1, 1))
        M = torch.tensor(M, dtype=self.Dtype, device=self.Device)

        Minv = matrix_inv(M)
        return Minv

    def phi_fun(self, x):
        bs, states_num = x.shape
        constraint_1 = x[:, 0] ** 2 + x[:, 1] ** 2 - 1 ** 2
        constraint_2 = (x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2 - 1 ** 2
        phi = torch.stack((constraint_1, constraint_2), dim=-1)
        return phi  # (bs ,2)

    def potential(self, x):
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = x[:, i * 2 + 1]
            U = U + self.m[i] * self.g * y
        return U

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
