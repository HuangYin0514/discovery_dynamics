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

    def __init__(self, ind, outd, layers=1, width=200):
        super(MLP, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width

        self.input_layer = nn.Sequential(
            nn.Linear(self.ind, self.width),
            nn.Softplus()
        )

        hidden_bock = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.Softplus()
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(self.layers)])

        self.output_layer = nn.Sequential(
            nn.Linear(self.width, self.outd, bias=False)
        )

        self.__initialize()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def __initialize(self):
        def input_layer_init(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:  # find the linear layer class
                # a, b = m.weight.shape
                # m.weight.data.normal_(0, 2.2 / np.sqrt(b))
                nn.init.xavier_normal_(m.weight)

        def hidden_layer_init(m):
            classname = m.__class__.__name__

            if classname.find("Linear") != -1:  # find the linear layer class
                # a, b = m.weight.shape
                # i, n = 1, 3
                # m.weight.data.normal_(0, 0.58 * (i + 1) / np.sqrt((a + b) / 2))
                nn.init.xavier_normal_(m.weight)

        def output_layer_init(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:  # find the linear layer class
                # a, b = m.weight.shape
                # m.weight.data.normal_(0, np.sqrt(a))
                nn.init.xavier_normal_(m.weight)

        self.input_layer.apply(input_layer_init)
        self.hidden_layer.apply(hidden_layer_init)
        self.output_layer.apply(output_layer_init)


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
        baseline = MLP(self.dim, 1, self.layers, self.width)
        return baseline

    def forward(self, t, data):

        bs = data.size(0)
        device = data.device
        _dof = int(self.dim / 2)

        x, v = torch.chunk(data, 2, dim=1)
        input = torch.cat([x, v], dim=1)

        L = self.baseline(input)

        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((bs, _dof, _dof), device=device)
        dxdvL = torch.zeros((bs, _dof, _dof), device=device)

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
