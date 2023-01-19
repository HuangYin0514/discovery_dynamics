# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
from .mlp import MLP

# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/17 7:51 PM
@desc:
"""
import torch
from torch import nn

from .base_module import LossNN
from ..integrator import ODESolver
from ..utils import dfx


class ModLaNet(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(ModLaNet, self).__init__()

        self.obj = obj
        self.dim = dim
        self.dof = obj * dim

        self.baseline = MLP(input_dim=obj * dim * 2, hidden_dim=200, output_dim=1, num_layers=1,
                            act=nn.Tanh)

    def forward(self, t, data):

        bs = data.size(0)

        x, v = torch.chunk(data, 2, dim=1)

        input = torch.cat([x, v], dim=1)
        L = self.baseline(input)

        dvL = dfx(L.sum(), v)  # (bs, v_dim)
        dxL = dfx(L.sum(), x)  # (bs, x_dim)

        dvdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)
        dxdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.dof):
            dvidvL = dfx(dvL[:, i].sum(), v)
            if dvidvL is None:
                break
            else:
                dvdvL[:, i, :] += dvidvL

        for i in range(self.dof):
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
