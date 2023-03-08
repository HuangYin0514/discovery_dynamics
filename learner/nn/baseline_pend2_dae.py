# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/7 8:42 PM
@desc:

利用baseline逼近dae方程

"""

import torch
from torch import nn

from ._base_module import LossNN
from .mlp import MLP
from ..integrator import ODESolver
from ..utils.common_utils import enable_grad


class Baseline_pend2_dae(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(Baseline_pend2_dae, self).__init__()

        self.obj = obj
        self.dim = dim

        self.baseline = MLP(input_dim=obj * dim * 2, hidden_dim=200, output_dim=obj * dim * 2, num_layers=1,
                            act=nn.Tanh)

    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        out = self.baseline(coords)
        return out

    def integrate(self, X0, t):
        coords = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return coords
