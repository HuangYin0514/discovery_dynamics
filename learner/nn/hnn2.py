from functools import partial

import numpy as np
import torch

from .fnn import FNN
from .base_module import LossNN
from ..integrator.rungekutta import RK4, RK45
from ..utils import lazy_property, dfx
from ..criterion import L2_norm_loss


class HNN(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, dim, layers=3, width=30):
        super(LossNN, self).__init__()
        self.name = 'hnn'

        self.dim = dim
        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = FNN(self.dim, 1, self.layers, self.width)
        return baseline

    @lazy_property
    def J(self):
        # [ 0, 1]
        # [-1, 0]
        d = int(self.dim / 2)
        res = np.eye(self.dim, k=d) - np.eye(self.dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)

    def forward(self, x):
        h = self.baseline(x)
        gradH = dfx(h, x)
        dy = self.J @ gradH.T  # dqq shape is (vector, batchsize)
        return dy.T

    def criterion(self, X, y, criterion_method='MSELoss'):
        return self.__integrator_loss(X, y, criterion_method)

    def __integrator_loss(self, X, y, criterion_method):
        y_hat = self(X)
        if criterion_method == 'MSELoss':
            return torch.nn.MSELoss()(y_hat, y)
        elif criterion_method == 'L2_norm_loss':
            return L2_norm_loss(y_hat, y)
        else:
            raise NotImplementedError

    def __model_f(self, t, X, circular_motion=False):
        if circular_motion:  # Handle the case of circular motion
            position_dim = len(X) // 2
            X[:position_dim] = X[:position_dim] % (2 * np.pi)
        x = torch.tensor(X, requires_grad=True, dtype=self.Dtype, device=self.Device).view(1, -1)
        dx = self.forward(x).cpu().detach().numpy().reshape(-1)
        return dx

    def predict(self, X, h, t0, t_end, solver_method="RK45", circular_motion=False):
        assert isinstance(X, np.ndarray), "input data must be numpy types"
        model_f = partial(self.__model_f, circular_motion=circular_motion)
        if solver_method == "RK45":
            solver = RK45(model_f, t0, t_end)
        elif solver_method == "RK4":
            solver = RK4(model_f, t0, t_end)
        else:
            raise NotImplementedError
        res = solver.solve(X, h)
        return res
