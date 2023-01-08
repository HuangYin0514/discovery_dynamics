import numpy as np
import torch

from .base_module import LossNN
from .fnn import FNN
from ..integrator import ODESolver
from ..utils import lazy_property, dfx, ham_J


class HNN(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, dim, layers=3, width=30):
        super(HNN, self).__init__()

        self.dim = dim
        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = FNN(self.dim, 1, self.layers, self.width)
        return baseline

    @lazy_property
    def J(self):
        # [ 0, I]
        # [-I, 0]
        d = int(self.dim / 2)
        res = np.eye(self.dim, k=d) - np.eye(self.dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)

    def forward(self, t, x):
        h = self.baseline(x)
        gradH = dfx(h, x)
        # dy = self.J @ gradH.T  # dy shape is (vector, batchsize)
        # return dy.T
        return ham_J(gradH)

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
