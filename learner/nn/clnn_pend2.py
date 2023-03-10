# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/10 5:14 PM
@desc:
"""
import numpy as np
import torch
from torch import nn, Tensor

from learner.integrator import ODESolver
from learner.nn import LossNN
from learner.nn.mlp import MLP
from learner.nn.utils_nn import ReshapeNet
from learner.utils.common_utils import matrix_inv, enable_grad, dfx


class MassNet(nn.Module):
    def __init__(self, q_dim, num_layers=3, hidden_dim=30):
        super(MassNet, self).__init__()

        self.net = nn.Sequential(
            MLP(input_dim=q_dim, hidden_dim=hidden_dim, output_dim=q_dim * q_dim, num_layers=num_layers,
                act=nn.Tanh),
            ReshapeNet(-1, q_dim, q_dim)
        )

    def forward(self, q):
        out = self.net(q)
        return out


class CLNN_pend2(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(CLNN_pend2, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.potential_net = MLP(input_dim=obj * dim, hidden_dim=256, output_dim=1, num_layers=3,
                                 act=nn.Tanh)

        self.mass_net = MassNet(q_dim=q_dim, num_layers=1, hidden_dim=50)


    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.shape[0]
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # 拟合 ------------------------------------------------------------------------------
        Minv = self.Minv(x)
        V = self.potential_net(x)

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
    #
    # def potential_net(self, x):
    #     self.m = [1., 5.]
    #     self.l = [1., 1.]
    #     self.g = 10.
    #     U = 0.
    #     y = 0.
    #     for i in range(self.obj):
    #         y = x[:, i * 2 + 1]
    #         U = U + self.m[i] * self.g * y
    #     return U

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

    # def Minv(self, q: Tensor, eps=1e-4) -> Tensor:
    #     """Compute the learned inverse mass matrix M^{-1}(q)
    #         M = LU
    #         M^{-1} = (LU)^{-1} = U^{-1} @ L^{-1}
    #         M^{-1}(q) = [x, 0] @ [x, x]
    #                     [x, x]   [0, x]
    #     Args:
    #         q: bs x D Tensor representing the position
    #     """
    #     assert q.ndim == 2
    #     lower_triangular = self.tril_Minv(q)
    #     assert lower_triangular.ndim == 3
    #     diag_noise = eps * torch.eye(lower_triangular.size(-1), dtype=q.dtype, device=q.device)
    #     Minv = lower_triangular.matmul(lower_triangular.transpose(-2, -1)) + diag_noise
    #     return Minv

    def Minv(self, q):
        self.m = [1., 5.]


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

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
