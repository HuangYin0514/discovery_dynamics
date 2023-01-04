# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import numpy as np
import torch

from learner.integrator.rungekutta import RK45
from .base_body_dataset import BaseBodyDataset


class Pendulum2(BaseBodyDataset):
    """
    Pendulum with 2 bodies
    Reference:
    # ref: Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
    # URL: https://proceedings.neurips.cc/paper/2020/file/9f655cc8884fda7ad6d8a6fb15cc001e-Paper.pdf
    Dataset statistics:
    # type: hamilton
    # obj: 2
    # dim: 1
    """

    def __init__(self, train_num, test_num, obj, dim, m=None, l=None, **kwargs):
        super(Pendulum2, self).__init__()

        self.train_num = train_num
        self.test_num = test_num
        self.dataset_url = ''

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self._m = [1 for i in range(obj)]
        self._l = [1 for i in range(obj)]
        self._g = 9.8

        self._obj = obj
        self._dim = dim
        self._dof = self._obj * self._dim  # degree of freedom

        t0 = 0
        t_end = 10
        self._h = 0.1
        self.solver = RK45(self.right_fn, t0=t0, t_end=t_end)

    def right_fn(self, t, coords):
        q1, q2, p1, p2 = coords
        l1, l2, m1, m2 = self._l[0], self._l[1], self._m[0], self._m[1]
        g = self._g
        b = l1 * l2 * (m1 + m2 * np.sin(q1 - q2) ** 2)
        dq1 = (l2 * p1 - l1 * p2 * np.cos(q1 - q2)) / (b * l1)
        dq2 = (-m2 * l2 * p1 * np.cos(q1 - q2) + (m1 + m2) * l1 * p2) / (m2 * b * l2)
        h1 = p1 * p2 * np.sin(q1 - q2) / b
        h2 = (m2 * l2 ** 2 * p1 ** 2 + (m1 + m2) * l1 ** 2 * p2 ** 2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / (
                2 * b ** 2)
        dp1 = -(m1 + m2) * g * l1 * np.sin(q1) - h1 + h2 * np.sin(2 * (q1 - q2))
        dp2 = -m2 * g * l2 * np.sin(q2) + h1 - h2 * np.sin(2 * (q1 - q2))
        return np.asarray([dq1, dq2, dp1, dp2]).reshape(-1)

    def M(self, x):
        """
        ref: Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
        Create a square mass matrix of size N x N.
        Note: the matrix is symmetric
        In the future, only half of the matrix can be considered
        """
        N = self._obj
        M = torch.zeros((N, N))
        for i in range(N):
            for k in range(N):
                m_sum = 0
                j = i if i >= k else k
                for tmp in range(j, N):
                    m_sum += self._m[tmp]
                M[i, k] += self._l[i] * self._l[k] * torch.cos(x[i] - x[k]) * m_sum
        return M.double()

    def Minv(self, x):
        return torch.inverse(self.M(x)).double()

    def kinetic(self, coords):
        """Kinetic energy"""
        assert len(coords) == self._dof * 2
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords)
        x, p = torch.split(coords, 2)
        T = torch.sum(0.5 * p @ self.Minv(x) @ p)
        return T

    def potential(self, coords):
        assert len(coords) == self._dof * 2
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords)
        g = self._g
        U = 0.
        y = 0.
        for i in range(self._obj):
            y = y - self._l[i] * torch.cos(coords[i])
            U = U + self._m[i] * self._g * y
        return U

    def energy_fn(self, coords):
        """能量函数"""
        H = self.kinetic(coords) + self.potential(coords)  # some error in this implementation
        return H

    def random_config(self):
        max_momentum = 1.
        x0 = np.zeros(self._obj * 2)
        for i in range(self._obj):
            theta = (2 * np.pi - 0) * np.random.rand() + 0  # [0, 2pi]
            momentum = (2 * np.random.rand() - 1) * max_momentum  # [-1, 1]*max_momentum
            x0[i] = theta
            x0[i + self._obj] = momentum
        return x0.reshape(-1)
