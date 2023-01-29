# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import torch
from torch import nn, Tensor

from .base_module import LossNN
from .mlp import MLP
from .utils_nn import CosSinNet, ReshapeNet
from ..integrator import ODESolver
from ..utils import dfx


class HnnMod_pend2(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(HnnMod_pend2, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.global_dim = 2
        self.global_dof = int(obj * self.global_dim)

        self.mass = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.mass.weight)
    def M(self, x):
        """
        ref: Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
        Create a square mass matrix of size N x N.
        Note: the matrix is symmetric
        In the future, only half of the matrix can be considered
        """
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

    def forward(self, t, x):
        bs = x.size(0)
        x, p = x.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # position transformations ----------------------------------------------------------------
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = y - torch.cos(x[:,i])
            U = U + 9.8 * y

        dqH = dfx(U.sum(), x)

        # Calculate the velocity （dq_dt = v = Minv @ p）--------------------------------------------------------------
        v_global = self.Minv(x).matmul(p.unsqueeze(-1)).squeeze(-1)

        # Calculate the Derivative ----------------------------------------------------------------
        dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            dq_dt[:, i * self.dim:(i + 1) * self.dim] = v_global[:, i * self.dim:  (i + 1) * self.dim]
            dp_dt[:, i * self.dim:(i + 1) * self.dim] = -dqH[:, i * self.dim:(i + 1) * self.dim]
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        return dz_dt

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
