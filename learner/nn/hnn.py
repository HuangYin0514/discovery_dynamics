import numpy as np
import torch
from torch import nn

from ._base_module import LossNN
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
        states_dim = self.obj * self.dim * 2
        d = int(states_dim / 2)
        res = np.eye(states_dim, k=d) - np.eye(states_dim, k=-d)
        return torch.tensor(res, dtype=self.Dtype, device=self.Device)

    def forward(self, t, x):
        h = self.baseline(x)
        gradH = dfx(h, x)
        dy = (self.J @ gradH.T).T  # dy shape is (bs, vector)
        return dy

    def integrate(self, X0, t):
        def angle_forward(t, coords):
            x, p = torch.chunk(coords, 2, dim=-1)
            new_x = x % (2 * torch.pi)
            new_coords = torch.cat([new_x, p], dim=-1).clone().detach()
            return self(t, new_coords)

        coords = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4

        x, p = torch.chunk(coords, 2, dim=-1)
        new_x = x % (2 * torch.pi)
        new_coords = torch.cat([new_x, p], dim=-1).clone().detach()

        return new_coords
