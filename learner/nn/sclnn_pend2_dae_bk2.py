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
from learner.nn.utils_nn import Identity
from learner.utils.common_utils import matrix_inv, enable_grad, dfx


class MassNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(MassNet, self).__init__()
        # hidden_bock = nn.Sequential(
        #     MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
        #         act=nn.Tanh)
        # )
        # self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(5)])

        self.net = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=nn.Tanh)

    def forward(self, x):
        y = self.net(x)
        return y


class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        hidden_bock = nn.Sequential(
            nn.Linear(input_dim, input_dim),
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(5)])
        self.net = nn.Sequential(
            nn.Linear(input_dim * 6, input_dim * 3),
            nn.Linear(input_dim * 3, output_dim),
        )

    def forward(self, x):
        input_list = []
        scale_list = [1 * x, 2 * x, 3 * x, 4 * x, 5 * x]
        for idx in range(len(self.hidden_layer)):
            input = scale_list[idx]
            output = self.hidden_layer[idx](input)
            # output = input
            input_list.append(output)
        input_list.append(x)
        y = torch.cat(input_list, dim=-1)
        out = self.net(y)
        return out


class SCLNN_pend2_dae(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(SCLNN_pend2_dae, self).__init__()

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

        self.mass1 = torch.nn.Parameter(.1 * torch.randn(1, ))
        self.mass2 = torch.nn.Parameter(.1 * torch.randn(1, ))

        self.potential_net = MLP(input_dim=obj * dim, hidden_dim=256, output_dim=1, num_layers=3,
                                 act=nn.Tanh)

    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.shape[0]
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # ?????? ------------------------------------------------------------------------------
        Minv = self.Minv(x)

        M = matrix_inv(Minv)
        V = self.potential_net(x)

        # ?????? -------------------------------------------------------------------------------
        phi = self.phi_fun(x)

        phi_q = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_q[:, i] = dfx(phi[:, i], x)
        phi_qq = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_qq[:, i] = dfx(phi_q[:, i].unsqueeze(-2) @ v.unsqueeze(-1), x)

        # ????????? -------------------------------------------------------------------------------
        F = -dfx(V, x)

        # ?????? lam ----------------------------------------------------------------
        phiq_Minv = torch.matmul(phi_q, Minv)  # (bs,2,4)
        L = torch.matmul(phiq_Minv, phi_q.permute(0, 2, 1))
        R = torch.matmul(phiq_Minv, F.unsqueeze(-1)) + torch.matmul(phi_qq, v.unsqueeze(-1))  # (2, 1)
        lam = torch.matmul(matrix_inv(L), R)
        # lam = torch.linalg.solve(L,R)

        # ?????? a ----------------------------------------------------------------
        a_R = F.unsqueeze(-1) - torch.matmul(phi_q.permute(0, 2, 1), lam)  # (4, 1)
        a = torch.matmul(Minv, a_R).squeeze(-1)  # (4, 1)
        return torch.cat([v, a], dim=-1)

    def Minv(self, q):
        bs, states = q.shape
        mass1 = torch.exp(-self.mass1)
        mass2 = torch.exp(-self.mass2)
        Minv = torch.cat([mass1, mass1, mass2, mass2], dim=0)
        Minv = torch.diag(Minv)
        Minv = Minv.repeat(bs, 1, 1)
        return Minv

    def phi_fun(self, x):
        bs, states_num = x.shape
        constraint_1 = x[:, 0] ** 2 + x[:, 1] ** 2 - 1 ** 2
        constraint_2 = (x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2 - 1 ** 2
        phi = torch.stack((constraint_1, constraint_2), dim=-1)
        return phi  # (bs ,2)

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
