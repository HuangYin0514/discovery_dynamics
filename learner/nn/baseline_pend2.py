import torch
from torch import nn

from .base_module import LossNN
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

    def forward(self, t, x):
        data = x.clone().detach()
        data[..., :int(data.shape[-1] // 2)] %= 2 * torch.pi  # pendulum

        out = self.baseline(data)
        return out

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        out[..., :int(out.shape[-1] // 2)] %= 2 * torch.pi
        return out
