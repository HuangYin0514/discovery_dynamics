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

    def forward(self, t, coords):

        x, v = torch.chunk(coords, 2, dim=0)

        L = self.baseline(x)

        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((self._dof, self._dof))
        dxdvL = torch.zeros((self._dof, self._dof))

        for i in range(self._dof):
            dvidvL = dfx(dvL[i].sum(), v)
            dvdvL[i] += dvidvL

        for i in range(self._dof):
            dxidvL = dfx(dvL[i].sum(), x)
            dxdvL[i] += dxidvL

        dvdvL_inv = torch.linalg.inv(dvdvL)

        res = dvdvL_inv @ (dxL - dxdvL @ v)

        return torch.cat([v, res], dim=0)

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
