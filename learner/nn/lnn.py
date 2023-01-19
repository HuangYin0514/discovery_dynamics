# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/17 7:51 PM
@desc:
"""
import numpy as np
import torch
from torch import nn

from .base_module import LossNN, StructureNN
from .utils_nn import weights_init_xavier_normal
from ..integrator import ODESolver
from ..utils import dfx


class MLP(nn.Module):
    '''Fully connected neural networks.
    '''

    def __init__(self, ind, outd, width=200):
        super(MLP, self).__init__()
        self.ind = ind
        self.outd = outd
        self.width = width

        self.mlp = nn.Sequential(
            nn.Linear(ind, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, outd)
        )

        self.__initialize()

    def forward(self, x):
        x = self.mlp(x)
        return x

    def __initialize(self):
        self.mlp.apply(weights_init_xavier_normal)


class LNN(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim, layers=3, width=128):
        super(LNN, self).__init__()

        self.obj = obj
        self.dim = dim
        self.input_dim = obj * dim * 2

        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = MLP(self.input_dim, 1, self.width)
        return baseline

    def forward(self, t, data):

        bs = data.size(0)
        device = data.device
        _dof = int(self.input_dim / 2)

        x, v = torch.chunk(data, 2, dim=1)
        input = torch.cat([x, v], dim=1)

        L = self.baseline(input)

        dvL = dfx(L.sum(), v)  # (bs, v_dim)
        dxL = dfx(L.sum(), x)  # (bs, x_dim)

        dvdvL = torch.zeros((bs, _dof, _dof), dtype=self.Dtype, device=self.Device)
        dxdvL = torch.zeros((bs, _dof, _dof), dtype=self.Dtype, device=self.Device)

        for i in range(_dof):
            dvidvL = dfx(dvL[:, i].sum(), v)
            if dvidvL is None:
                break
            else:
                dvdvL[:, i, :] += dvidvL

        for i in range(_dof):
            dxidvL = dfx(dvL[:, i].sum(), x)
            if dxidvL is None:
                break
            else:
                dxdvL[:, i, :] += dxidvL

        dvdvL_inv = torch.linalg.pinv(dvdvL)

        a = dvdvL_inv @ (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))  # (bs, a_dim, 1)
        a = a.squeeze(2)
        return torch.cat([v, a], dim=1)

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='rk4').permute(1, 0, 2)  # (T, D)
        return out
