from .base_module import LossNN
from .fnn import FNN
from ..integrator import ODESolver


class Baseline(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, dim, layers=3, width=30):
        super(Baseline, self).__init__()

        self.dim = dim
        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = FNN(self.dim, self.dim, self.layers, self.width)
        return baseline

    def forward(self, t, x):
        out = self.baseline(x)
        return  out
    def integrate(self, X, t):
        out = ODESolver(self, X, t, method='dopri5').permute(1, 0, 2)  # (T, D)
        return out
