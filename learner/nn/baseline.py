from .base_module import LossNN
from .fnn import FNN
from ..integrator import ODESolver


class Baseline(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, obj, dim, layers=1, width=200):
        super(Baseline, self).__init__()

        self.obj = obj
        self.dim = dim
        self.input_dim = obj * dim * 2

        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = FNN(self.input_dim, self.input_dim, self.layers, self.width)
        return baseline

    def forward(self, t, x):
        out = self.baseline(x)
        return out

    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
