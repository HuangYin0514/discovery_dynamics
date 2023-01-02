# import autograd
# import autograd.numpy as np
import numpy as np
import torch
from .base_data import BaseDynamicsData
from ..integrator.rungekutta import RK4, RK45
from ..utils import deprecated, lazy_property, dfx


class PendulumData(BaseDynamicsData):
    def __init__(self, obj, dim, train_num, test_num, m=None, l=None, **kwargs):
        super(PendulumData, self).__init__()

        self.train_num = train_num
        self.test_num = test_num

        self.m = m
        self.l = l
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = obj * self.dim  # degree of freedom

        t0 = 0
        t_end = 10
        self.h = 0.1
        self.solver = RK45(self.hamilton_right_fn, t0=t0, t_end=t_end)

    # def hamilton_right_fn(self, t, coords):
    #     coords = torch.tensor(coords, requires_grad=True)
    #     grad_ham = dfx(self.energy_fn(coords), coords).detach().numpy()
    #     q, p = grad_ham[self.dof:], -grad_ham[:self.dof]
    #     return np.asarray([q, p]).reshape(-1)

    def hamilton_right_fn(self, t, coords):
        q1, q2, p1, p2 = coords
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        g = self.g
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
        N = self.obj
        M = torch.zeros((N, N))
        for i in range(N):
            for k in range(N):
                m_sum = 0
                j = i if i >= k else k
                for tmp in range(j, N):
                    m_sum += self.m[tmp]
                M[i, k] += self.l[i] * self.l[k] * torch.cos(x[i] - x[k]) * m_sum
        return M.double()

    def Minv(self, x):
        return torch.inverse(self.M(x)).double()

    def hamilton_kinetic(self, coords):
        """Kinetic energy"""
        assert len(coords) == self.dof * 2
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords)
        x, p = torch.split(coords, 2)
        kinetic = torch.sum(0.5 * p @ self.Minv(x) @ p)
        return kinetic

    def potential(self, coords):
        assert len(coords) == self.dof * 2
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords)
        g = self.g
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = y - self.l[i] * torch.cos(coords[i])
            U = U + self.m[i] * self.g * y
        return U

    def energy_fn(self, coords):
        """能量函数"""
        H = self.hamilton_kinetic(coords) + self.potential(coords)  # some error in this implementation
        return H

    def random_config(self, num):
        x0_list = []
        for _ in range(num):
            max_momentum = 1.
            x0 = np.zeros(self.obj * 2)
            for i in range(self.obj):
                theta = (2 * np.pi - 0) * np.random.rand() + 0  # [0, 2pi]
                momentum = (2 * np.random.rand() - 1) * max_momentum  # [-1, 1]*max_momentum
                x0[i] = theta
                x0[i + self.obj] = momentum
            x0_list.append(x0.reshape(-1))
        return np.asarray(x0_list)
