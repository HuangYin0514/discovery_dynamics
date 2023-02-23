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


class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        hidden_bock = nn.Sequential(
            nn.Linear(input_dim, input_dim * 6),
            nn.LeakyReLU()
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(5)])

        self.mlp = MLP(input_dim=input_dim * 6 * 5, hidden_dim=hidden_dim, output_dim=output_dim,
                       num_layers=num_layers,
                       act=act)

    def forward(self, x):
        input_list = []
        scale_list = [1 * x, 2 * x, 3 * x, x, 4 * x, 5 * x]
        for idx in range(len(self.hidden_layer)):
            input = scale_list[idx]
            output = self.hidden_layer[idx](input)
            input_list.append(output)
        # input_list.append(torch.sin(x))
        # input_list.append(torch.cos(x))
        x = torch.cat(input_list, dim=1)
        y = self.mlp(x)
        return y


class HnnModScale_body3(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(HnnModScale_body3, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.global_dim = 2
        self.global_dof = int(obj * self.global_dim)

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

    def forward(self, t, x):
        bs = x.size(0)
        x, p = x.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        x_global = x

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

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        dqH = dfx(U.sum(), x)

        v_global = p

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
        out = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D) dopri5
        return out
