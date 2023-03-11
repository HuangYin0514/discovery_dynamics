# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/11 2:24 AM
@desc:
"""
# encoding: utf-8

import torch
from torch import nn

from learner.integrator import ODESolver
from learner.nn import LossNN
from learner.nn.mlp import MLP
from learner.nn.utils_nn import Identity, Compact_Support_Activation
from learner.utils.common_utils import matrix_inv, enable_grad, dfx


class MassNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(MassNet, self).__init__()
        hidden_bock = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(5)])
        self.net = MLP(input_dim=input_dim * 6, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x):
        input_list = []
        scale_list = [2 * x, 4 * x, 8 * x, 16 * x, 32 * x]
        for idx in range(len(self.hidden_layer)):
            input = scale_list[idx]
            output = self.hidden_layer[idx](input)
            # output = input
            input_list.append(output)
        input_list.append(x)
        y = torch.cat(input_list, dim=-1)
        out = self.net(y)
        return out




class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        hidden_bock = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(5)])
        self.net = MLP(input_dim=input_dim * 6, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x):
        input_list = []
        scale_list = [2 * x, 4 * x, 8 * x, 16 * x, 32 * x]
        for idx in range(len(self.hidden_layer)):
            input = scale_list[idx]
            output = self.hidden_layer[idx](input)
            # output = input
            input_list.append(output)
        input_list.append(x)
        y = torch.cat(input_list, dim=-1)
        out = self.net(y)
        return out


class SCLNN_pend2(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(SCLNN_pend2, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        # self.potential_net = MLP(input_dim=obj * dim, hidden_dim=256, output_dim=1, num_layers=3,
        #                          act=nn.Tanh)

        self.mass_net = MassNet(input_dim=self.obj,
                                hidden_dim=10,
                                output_dim=self.obj,
                                num_layers=1, act=nn.Tanh)

        self.Potential1 = PotentialEnergyCell(input_dim=self.dim,
                                              hidden_dim=20,
                                              output_dim=1,
                                              num_layers=1, act=Identity)
        self.Potential2 = PotentialEnergyCell(input_dim=self.dim * 2,
                                              hidden_dim=20,
                                              output_dim=1,
                                              num_layers=1, act=Identity)
        self.co1 = torch.nn.Parameter(torch.ones(1, dtype=self.Dtype, device=self.Device) * 0.5)
        self.co2 = torch.nn.Parameter(torch.ones(1, dtype=self.Dtype, device=self.Device) * 0.5)

    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.shape[0]
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # 拟合 ------------------------------------------------------------------------------
        Minv = self.Minv(x)

        M = matrix_inv(Minv)
        V = 0.
        for i in range(self.obj):
            V += self.co1 * M[0, i * self.dim, i * self.dim] * self.Potential1(x[:, i * self.dim: (i + 1) * self.dim])

        for i in range(self.obj):
            for j in range(i):
                x_ij = torch.cat(
                    [x[:, i * self.dim: (i + 1) * self.dim],
                     x[:, j * self.dim: (j + 1) * self.dim]],
                    dim=1)
                x_ji = torch.cat(
                    [x[:, j * self.dim: (j + 1) * self.dim],
                     x[:, i * self.dim: (i + 1) * self.dim]],
                    dim=1)
                V += self.co2 * 0.5 * M[0, i * self.dim, i * self.dim] * (self.Potential2(x_ij) + self.Potential2(x_ji))

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
        # lam = torch.linalg.solve(L,R)

        # 求解 a ----------------------------------------------------------------
        a_R = F.unsqueeze(-1) - torch.matmul(phi_q.permute(0, 2, 1), lam)  # (4, 1)
        a = torch.matmul(Minv, a_R).squeeze(-1)  # (4, 1)
        return torch.cat([v, a], dim=-1)

    def Minv(self, q):
        bs, states = q.shape

        I = torch.ones(bs, 2, dtype=self.Dtype, device=self.Device)
        Minv = self.mass_net(I)
        Minv = torch.exp(-Minv)
        Minv = torch.cat([Minv[:, 0:1], Minv[:, 0:1], Minv[:, 1:2], Minv[:, 1:2]], dim=-1)
        Minv = torch.diag_embed(Minv)
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
