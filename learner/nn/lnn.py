# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/17 7:51 PM
@desc:
"""

import torch

from .base_module import LossNN
from .fnn import FNN
from ..integrator import ODESolver
from ..utils import dfx


class LNN(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, dim, layers=3, width=30):
        super(LNN, self).__init__()

        self.dim = dim
        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = FNN(self.dim, 1, self.layers, self.width)
        return baseline

    def forward(self, t, data):

        bs = data.size(0)
        _dof = int(self.dim / 2)

        x, v = torch.chunk(data, 2, dim=1)
        input = torch.cat([x, v], dim=1)

        L = self.baseline(input)

        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((bs, _dof, _dof))
        dxdvL = torch.zeros((bs, _dof, _dof))

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

        a = dvdvL_inv @ (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))
        a = a.squeeze(2)
        return torch.cat([v, a], dim=1)

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='rk4').permute(1, 0, 2)  # (T, D)
        return out
