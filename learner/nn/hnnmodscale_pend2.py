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
from .utils_nn import ReshapeNet
from ..integrator import ODESolver
from ..utils import dfx


class MassNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=30, num_layers=3, act=nn.Tanh):
        super(MassNet, self).__init__()
        hidden_bock = nn.Sequential(
            nn.Linear(input_dim, input_dim * 6),
            nn.Tanh()
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(5)])

        self.net = nn.Sequential(
            MLP(input_dim=input_dim * 6 * 5 + 2 * input_dim, hidden_dim=hidden_dim, output_dim=input_dim * input_dim,
                num_layers=num_layers, act=act),
            ReshapeNet(-1, input_dim, input_dim)
        )

    def forward(self, x):
        input_list = []
        scale_list = [1 * x, 2 * x, 3 * x, 4 * x, 5 * x]
        for idx in range(len(self.hidden_layer)):
            input = scale_list[idx]
            output = self.hidden_layer[idx](input)
            input_list.append(output)
        input_list.append(torch.sin(x))
        input_list.append(torch.cos(x))
        x = torch.cat(input_list, dim=1)
        out = self.net(x)
        return out



class GlobalPositionTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalPositionTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, x_0):
        y = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) + x_0
        y = self.mlp(x) + x_0
        return y


class GlobalVelocityTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalVelocityTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, v, v0):
        y = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) * v + v0
        y = self.mlp(x) * v + v0
        return y


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
        scale_list = [x, 2 * x, 3 * x, 4 * x, 5 * x]
        for idx in range(len(self.hidden_layer)):
            input = scale_list[idx]
            output = self.hidden_layer[idx](input)
            input_list.append(output)
        # input_list.append(torch.sin(x))
        # input_list.append(torch.cos(x))
        x = torch.cat(input_list, dim=1)
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

    def forward(self, t, coords):
        __x, __p = torch.chunk(coords, 2, dim=-1)
        coords = torch.cat([__x % (2 * torch.pi), __p], dim=-1).clone().detach().requires_grad_(True)

        bs = coords.size(0)
        x, p = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # position transformations ----------------------------------------------------------------
        x_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        x_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        for i in range(self.obj):
            for j in range(i):
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += x_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
            x_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4x(
                x[:, (i) * self.dim: (i + 1) * self.dim],
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
        out = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5
        return out
