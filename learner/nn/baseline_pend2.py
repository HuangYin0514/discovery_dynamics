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
        out = self.baseline(coords)
        return out

    def integrate(self, X0, t):
        def angle_forward(t, coords):
            q, p = torch.chunk(coords, 2, dim=-1)
            new_x = q % (2 * torch.pi)
            new_coords = torch.cat([new_x, p], dim=-1).clone().detach().requires_grad_(True)
            return self(t, new_coords)

        coords = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4

        return coords
