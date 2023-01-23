# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import torch
from torch import nn

from .base_module import LossNN
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
        y = self.mlp(x) + x_0
        return y


class GlobalVelocityTransform(nn.Module):
    """Doing coordinate transformation using a MLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(GlobalVelocityTransform, self).__init__()
        self.mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       act=act)

    def forward(self, x, v, v0):
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


class ModLaNet(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(ModLaNet, self).__init__()

        self.obj = obj
        self.dim = dim
        self.dof = obj * dim

        self.mass = torch.nn.Linear(1, 1, bias=False)

    def forward(self, t, data):
        bs = data.size(0)
        x, v = torch.chunk(data, 2, dim=1)

        L, T, U = 0., 0., torch.zeros((x.shape[0], 1), dtype=self.Dtype, device=self.Device)

        # Calculate the potential energy for i-th element
        y = 0
        for i in range(self.obj):
            y = y - torch.cos(x[:, i:i + 1])
            U += (9.8 * y)

        # Calculate the kinetic energy for i-th element
        T = 0.
        vx, vy = 0., 0.
        for i in range(self.obj):
            vx = vx + v[:,i] * torch.cos(x[:,i])
            vy = vy + v[:,i] * torch.sin(x[:,i])
            T = T + 0.5 * (torch.pow(vx, 2) + torch.pow(vy, 2))

        # Construct Lagrangian
        L = (T - U)

        dvL = dfx(L.sum(), v)  # (bs, v_dim)
        dxL = dfx(L.sum(), x)  # (bs, x_dim)

        dvdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)
        dxdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.dof):
            dvidvL = dfx(dvL[:, i].sum(), v)
            dvdvL[:, i, :] += dvidvL

        for i in range(self.dof):
            dxidvL = dfx(dvL[:, i].sum(), x)
            dxdvL[:, i, :] += dxidvL

        dvdvL_inv = torch.linalg.pinv(dvdvL)

        a = dvdvL_inv @ (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))  # (bs, a_dim, 1)
        a = a.squeeze(2)

        t1, t2, w1, w2 = x[:, 0:1], x[:, 1:2], v[:, 0:1], v[:, 1:2]
        l1, l2, m1, m2 = 1, 1, 1, 1
        g = 9.8

        a1 = (l2 / l1) * (m2 / (m1 + m2)) * torch.cos(t1 - t2)
        a2 = (l1 / l2) * torch.cos(t1 - t2)
        f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * torch.sin(t1 - t2) - (g / l1) * torch.sin(t1)
        f2 = (l1 / l2) * (w1 ** 2) * torch.sin(t1 - t2) - (g / l2) * torch.sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        return torch.cat([v, a], dim=1)

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='rk4').permute(1, 0, 2)  # (T, D)
        out[..., :int(out.shape[-1] // 2)] %= 2 * torch.pi  # TODO pendulum
        return out
