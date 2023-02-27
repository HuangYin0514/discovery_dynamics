# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import torch
from torch import nn

from ._base_module import LossNN
from .mlp import MLP
from ..integrator import ODESolver
from ..utils import dfx


#
# class MassNet(nn.Module):
#     def __init__(self, q_dim, num_layers=3, hidden_dim=30):
#         super(MassNet, self).__init__()
#
#         self.cos_sin_net = CosSinNet()
#         self.net = nn.Sequential(
#             MLP(input_dim=q_dim * 2, hidden_dim=hidden_dim, output_dim=q_dim * q_dim, num_layers=num_layers,
#                 act=nn.Tanh),
#             ReshapeNet(-1, q_dim, q_dim)
#         )
#
#     def forward(self, q):
#         q = self.cos_sin_net(q)
#         out = self.net(q)
#         return out

#
# class DynamicsNet(nn.Module):
#     def __init__(self, q_dim, p_dim, num_layers=3, hidden_dim=30):
#         super(DynamicsNet, self).__init__()
#         self.cos_sin_net = CosSinNet()
#
#         self.dynamics_net = nn.Sequential(
#             MLP(input_dim=q_dim * 2 + p_dim, hidden_dim=hidden_dim, output_dim=p_dim, num_layers=num_layers,
#                 act=nn.Tanh),
#             ReshapeNet(-1, p_dim)
#         )
#
#     def forward(self, q, p):
#         q = self.cos_sin_net(q)
#         x = torch.cat([q, p], dim=1)
#         out = self.dynamics_net(x)
#         return out


class GlobalPositionTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalPositionTransform, self).__init__()

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       num_layers=num_layers,
                       act=act)

    def forward(self, x, x_0):
        y = self.mlp(x) + x_0
        return y


class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       num_layers=num_layers,
                       act=act)

    def forward(self, x):
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
        __x, __p = torch.chunk(coords, 2, dim=-1)
        coords = torch.cat([__x % (2 * torch.pi), __p], dim=-1).clone().detach().requires_grad_(True)

        bs = coords.size(0)
        q, p = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # position transformations ----------------------------------------------------------------
        x_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        x_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            for j in range(i):
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += x_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
            x_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4x(
                q[:, (i) * self.dim: (i + 1) * self.dim],
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
        T = (0.5 * p.unsqueeze(1) @ self.Minv(q) @ p.unsqueeze(-1)).squeeze(-1)

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        H = U + T
        dqH = dfx(H.sum(), q)
        dpH = dfx(H.sum(), p)

        v_global = self.Minv(q).matmul(p.unsqueeze(-1)).squeeze(-1)

        # Calculate the Derivative ----------------------------------------------------------------
        dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            # dq_dt[:, i * self.dim:(i + 1) * self.dim] = v_global[:, i * self.dim: (i + 1) * self.dim]
            dq_dt[:, i * self.dim:(i + 1) * self.dim] = v_global[:, i * self.dim: (i + 1) * self.dim]
            dp_dt[:, i * self.dim:(i + 1) * self.dim] = -dqH[:, i * self.dim:(i + 1) * self.dim]
        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)
        return dz_dt

    def integrate(self, X0, t):

        coords = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4

        return coords
