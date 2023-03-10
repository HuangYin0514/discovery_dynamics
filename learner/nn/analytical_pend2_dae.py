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

        self.m = [1., 5.]
        self.l = [1., 1.]
        self.g = 10.

    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.shape[0]
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # 拟合 ------------------------------------------------------------------------------
        Minv = self.Minv(x)
        V = self.potential(torch.cat([x, v], dim=-1))

        # 约束 -------------------------------------------------------------------------------
        phi = self.phi_fun(x)

        phi_q = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_q[:, i] = dfx(phi[:, i], x)
        phi_qq = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_qq[:, i] = dfx(phi_q[:, i].unsqueeze(-2) @ v.unsqueeze(-1), x)

        # 右端项 -------------------------------------------------------------------------------
        F = -dfx(V, x)

        # 求解 lam ----------------------------------------------------------------
        phiq_Minv = torch.matmul(phi_q, Minv)  # (bs,2,4)
        L = torch.matmul(phiq_Minv, phi_q.permute(0, 2, 1))
        R = torch.matmul(phiq_Minv, F.unsqueeze(-1)) + torch.matmul(phi_qq, v.unsqueeze(-1))  # (2, 1)
        lam = torch.matmul(matrix_inv(L), R)

        # 求解 a ----------------------------------------------------------------
        a_R = F.unsqueeze(-1) - torch.matmul(phi_q.permute(0, 2, 1), lam)  # (4, 1)
        a = torch.matmul(Minv, a_R).squeeze(-1)  # (4, 1)
        return torch.cat([v, a], dim=-1)

    def Minv(self, q):
        bs, states = q.shape

        M = np.diag(np.array([self.m[0], self.m[0], self.m[1], self.m[1]]))
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

    def kinetic(self, coords):

        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)
        T = 0.
        for i in range(self.obj):
            vx = v[:, i * 2]
            vy = v[:, i * 2 + 1]
            T = T + 0.5 * self.m[i] * (vx ** 2 + vy ** 2)
        return T

    def potential(self, coords):
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = x[:, i * 2 + 1]
            U = U + self.m[i] * self.g * y
        return U

    def energy_fn(self, coords):
        """energy function """
        H = self.kinetic(coords) + self.potential(coords)
        return H

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
