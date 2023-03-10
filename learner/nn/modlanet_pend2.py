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
from .utils_nn import Identity
from ..integrator import ODESolver
from ..utils import dfx


class GlobalPositionTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalPositionTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, x_0):
        # y = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) + x_0
        y = self.mlp(x) + x_0
        return y


class GlobalVelocityTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalVelocityTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, v, v0):
        # y = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) * v + v0
        y = self.mlp(x) * v + v0
        return y


class PotentialEnergyCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(PotentialEnergyCell, self).__init__()

        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x):
        y = self.mlp(x)
        return y


class ModLaNet_pend2(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(ModLaNet_pend2, self).__init__()

        self.obj = obj
        self.dim = dim
        self.dof = obj * dim

        self.global_dim = 2
        self.global_dof = obj * self.global_dim

        self.global4x = GlobalPositionTransform(input_dim=self.dim,
                                                hidden_dim=16,
                                                output_dim=self.global_dim,
                                                num_layers=1, act=nn.Tanh)
        self.global4v = GlobalVelocityTransform(input_dim=self.dim,
                                                hidden_dim=16,
                                                output_dim=self.global_dim,
                                                num_layers=1, act=nn.Tanh)
        self.Potential1 = PotentialEnergyCell(input_dim=self.global_dim,
                                              hidden_dim=50,
                                              output_dim=1,
                                              num_layers=1, act=Identity)
        self.Potential2 = PotentialEnergyCell(input_dim=self.global_dim * 2,
                                              hidden_dim=50,
                                              output_dim=1,
                                              num_layers=1, act=Identity)

        self.co1 = torch.nn.Parameter(torch.ones(1, dtype=self.Dtype, device=self.Device) * 0.5)
        self.co2 = torch.nn.Parameter(torch.ones(1, dtype=self.Dtype, device=self.Device) * 0.5)

        self.mass = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.mass.weight)

    def forward(self, t, coords):
        __x, __p = torch.chunk(coords, 2, dim=-1)
        coords = torch.cat([__x % (2 * torch.pi), __p], dim=-1).detach().clone().requires_grad_(True)

        bs = coords.size(0)
        x, v = torch.chunk(coords, 2, dim=-1)

        x_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        v_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)

        x_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        v_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.obj):
            for j in range(i):
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += x_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
                v_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += v_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
            x_origin[:, (i) * self.dim: (i + 1) * self.dim] += 0.0
            x_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4x(
                x[:, (i) * self.dim: (i + 1) * self.dim],
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim])
            v_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = self.global4v(
                x[:, (i) * self.dim: (i + 1) * self.dim],
                v[:, (i) * self.dim: (i + 1) * self.dim],
                v_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim])

        # Calculate the potential energy for i-th element ------------------------------------------------------------
        U = 0.
        for i in range(self.obj):
            U += self.co1 * self.mass(self.Potential1(x_global[:, i * self.global_dim: (i + 1) * self.global_dim]))

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
                U += self.co2 * (0.5 * self.mass(self.Potential2(x_ij)) + 0.5 * self.mass(self.Potential2(x_ji)))

        # Calculate the kinetic --------------------------------------------------------------
        T = 0.
        vx, vy = 0., 0.
        for i in range(self.dof):
            # vx = v_global[:, i * self.global_dim]
            # vy = v_global[:, i * self.global_dim + 1]
            vv = v_global[:, (i) * self.global_dim: (i + 1) * self.global_dim].pow(2).sum(-1, keepdim=True)
            T += 0.5 * vv

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        L = T - U

        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)
        dxdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.dof):
            dvidvL = dfx(dvL[:, i].sum(), v)
            if dvidvL is None:
                print("i am is none of dvidvL")
            dvdvL[:, i, :] += dvidvL

        for i in range(self.dof):
            dxidvL = dfx(dvL[:, i].sum(), x)
            if dxidvL is None:
                print("i am is none of dxidvL")
            dxdvL[:, i, :] += dxidvL

        dvdvL_inv = torch.linalg.inv(dvdvL)

        a = dvdvL_inv.matmul(dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))  # (bs, a_dim, 1)
        # a = dvdvL_inv @ (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))  # (bs, a_dim, 1)

        a = a.squeeze(2)
        return torch.cat([v, a], dim=-1)

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
