import numpy as np
import torch
from torch import nn

from .base_module import LossNN
from .fnn import FNN
from .mlp import MLP
from ..integrator import ODESolver
from ..utils import lazy_property, dfx


class HNN(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(HNN, self).__init__()

        self.obj = obj
        self.dim = dim

        self.baseline = MLP(input_dim=obj * dim * 2, hidden_dim=200, output_dim=1, num_layers=1, act=nn.Tanh)

    @lazy_property
    def J(self):
        # [ 0, I]
        # [-I, 0]
        d = int(self.input_dim / 2)
        res = np.eye(self.input_dim, k=d) - np.eye(self.input_dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)

    def forward(self, t, x):
        h = self.baseline(x)
        gradH = dfx(h, x)
        dy = (self.J @ gradH.T).T  # dy shape is (bs, vector)
        return dy

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
