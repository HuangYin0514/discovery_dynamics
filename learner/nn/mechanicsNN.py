# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/10 10:59 AM
@desc:
"""
from .base_module import LossNN
from .fnn import FNN


class MechanicsNN(LossNN):
    '''Hamiltonian neural networks.
    '''

    def __init__(self, dim, layers=3, width=30):
        super(MechanicsNN, self).__init__()

        self.dim = dim
        self.layers = layers
        self.width = width

        self.baseline = self.__init_modules()

    def __init_modules(self):
        baseline = FNN(self.dim, self.dim, self.layers, self.width)
        return baseline

    def forward(self, t, x):
        return self.baseline(x)
