# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/12 9:56 AM
@desc:
"""

# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import torch

from ._base_module import LossNN
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
        M = torch.zeros((x.shape[0], N, N), dtype=self.Dtype, device=self.Device)
        for i in range(N):
            for k in range(N):
                m_sum = 0
                j = i if i >= k else k
                for tmp in range(j, N):
                    m_sum += 1.0
                M[:, i, k] = torch.cos(x[:, i] - x[:, k]) * m_sum
        return M

    def Minv(self, x):
        return torch.linalg.inv(self.M(x))

    def forward(self, t, coords):
        with torch.enable_grad():

            __x, __p = torch.chunk(coords, 2, dim=-1)
            coords = torch.cat([__x % (2 * torch.pi), __p], dim=-1).clone().detach().requires_grad_(True)

            coords = coords.clone().detach().requires_grad_(True)
            bs = coords.size(0)
            q, p = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

            # Calculate the potential energy for i-th element ------------------------------------------------------------
            U = 0.
            y = 0.
            for i in range(self.obj):
                y = y - torch.cos(q[:, i])
                U = U + 9.8 * y

            # Calculate the kinetic --------------------------------------------------------------
            # T = self.dataset.kinetic(torch.cat([x, p], dim=-1).reshape(-1))

            T = 0.
            v = torch.matmul(self.Minv(q), p.unsqueeze(-1))
            T = 0.5 * torch.matmul(p.unsqueeze(1), v).squeeze(-1).squeeze(-1)

            # Calculate the Hamilton Derivative --------------------------------------------------------------
            H = U + T
            dqH = dfx(H.sum(), q)
            dpH = dfx(H.sum(), p)

            v_global = self.Minv(q).matmul(p.unsqueeze(-1)).squeeze(-1)

            # Calculate the Derivative ----------------------------------------------------------------
            # dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
            # dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
            # for i in range(self.obj):
            #     dq_dt[:, i * self.dim:(i + 1) * self.dim] = dpH[:, i * self.dim: (i + 1) * self.dim]
            #     dp_dt[:, i * self.dim:(i + 1) * self.dim] = -dqH[:, i * self.dim:(i + 1) * self.dim]
            # dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
            dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype)
            dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype)

            dq_dt = dpH
            dp_dt = -dqH

            dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
            return dz_dt

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
