import torch
from torch import nn

from ._base_module import LossNN
from .mlp import MLP
from ..integrator import ODESolver


class Baseline_pend2(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim):
        super(Baseline_pend2, self).__init__()

        self.obj = obj
        self.dim = dim

        self.baseline = MLP(input_dim=obj * dim * 2, hidden_dim=200, output_dim=obj * dim * 2, num_layers=1,
                            act=nn.Tanh)

    def forward(self, t, coords):
        with torch.enable_grad():

            __x, __p = torch.chunk(coords, 2, dim=-1)
            coords = torch.cat([__x % (2 * torch.pi), __p], dim=-1).clone().detach().requires_grad_(True)

            out = self.baseline(coords)
        return out

    def integrate(self, X0, t):
        coords = ODESolver(self, X0, t, method='dopri5').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return coords
