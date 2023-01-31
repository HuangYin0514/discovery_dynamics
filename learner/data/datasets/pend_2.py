# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import numpy as np
import torch
from torch import nn

from ._base_body_dataset import BaseBodyDataset
from ...utils import lazy_property, dfx


class Pendulum2(BaseBodyDataset, nn.Module):
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
        self.dataset_url = 'https://drive.google.com/file/d/1gFpZaOsaL8-ooXs6Cn-yfisT8U6S12Qk/view?usp=share_link'

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self._m = [1 for i in range(obj)]
        self._l = [1 for i in range(obj)]
        self._g = 9.8

        self._obj = obj
        self._dim = dim
        self._dof = self._obj * self._dim  # degree of freedom

        self.dt = 0.1

        t0 = 0.
        t_end = 10.
        _time_step = int((t_end - t0) / self.dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 15.
        _time_step = int((t_end - t0) / self.dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    @lazy_property
    def J(self):
        # [ 0, I]
        # [-I, 0]
        d = self._dof
        res = np.eye(self._dof * 2, k=d) - np.eye(self._dof * 2, k=-d)
        return torch.tensor(res).float()

    def forward(self, t, coords):
        assert len(coords) == self._dof * 2
        q1, q2, p1, p2 = torch.chunk(coords, 4, dim=0)
        l1, l2, m1, m2 = self._l[0], self._l[1], self._m[0], self._m[1]
        g = self._g
        b = l1 * l2 * (m1 + m2 * torch.sin(q1 - q2) ** 2)
        dq1 = (l2 * p1 - l1 * p2 * torch.cos(q1 - q2)) / (b * l1)
        dq2 = (-m2 * l2 * p1 * torch.cos(q1 - q2) + (m1 + m2) * l1 * p2) / (m2 * b * l2)
        h1 = p1 * p2 * torch.sin(q1 - q2) / b
        h2 = (m2 * l2 ** 2 * p1 ** 2 + (m1 + m2) * l1 ** 2 * p2 ** 2 - 2 * m2 * l1 * l2 * p1 * p2 * torch.cos(
            q1 - q2)) / (2 * b ** 2)
        dp1 = -(m1 + m2) * g * l1 * torch.sin(q1) - h1 + h2 * torch.sin(2 * (q1 - q2))
        dp2 = -m2 * g * l2 * torch.sin(q2) + h1 - h2 * torch.sin(2 * (q1 - q2))

        ## test #################################################################
        # coords = coords.clone().detach().requires_grad_(True)
        # h = self.energy_fn(coords)
        # gradH = dfx(h, coords)
        # dy = self.J @ gradH  # dy shape is (vector, )
        # ## return dy
        #
        # coords = coords.clone().detach().requires_grad_(True)
        # x, p = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)
        #
        # U = 0.
        # y = 0.
        # for i in range(2):
        #     y = y - torch.cos(x[i])
        #     U = U + 9.8 * y
        #
        # T = torch.sum(0.5 * p @ self.Minv(x) @ p)
        # dqH1 = dfx((T).sum(), x)
        # dqH2 = dfx((U).sum(), x)
        # dqH3 = dqH1 + dqH2
        # dqH = dfx((U + T).sum(), x)
        #
        # dpH = dfx((U + T).sum(), p)
        #
        # v = self.Minv(x) @ p
        # res1 = torch.cat([dq1, dq2, dp1, dp2], dim=0)
        # res2 = torch.cat([v, dqH], dim=0)
        ################################################################################

        return torch.cat([dq1, dq2, dp1, dp2], dim=0)

    def M(self, x):
        """
        ref: Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
        Create a square mass matrix of size N x N.
        Note: the matrix is symmetric
        In the future, only half of the matrix can be considered
        """
        N = self._obj
        M = torch.zeros((N, N), device=x.device)
        for i in range(N):
            for k in range(N):
                m_sum = 0
                j = i if i >= k else k
                for tmp in range(j, N):
                    m_sum += self._m[tmp]
                M[i, k] = self._l[i] * self._l[k] * torch.cos(x[i] - x[k]) * m_sum
        return M

    def Minv(self, x):
        return torch.inverse(self.M(x))

    def kinetic(self, coords):
        """Kinetic energy"""
        assert len(coords) == self._dof * 2
        x, p = torch.chunk(coords, 2, dim=0)
        T = torch.sum(0.5 * p @ self.Minv(x) @ p)

        return T

    def potential(self, coords):
        assert len(coords) == self._dof * 2
        U = 0.
        y = 0.
        for i in range(self._obj):
            y = y - self._l[i] * torch.cos(coords[i])
            U = U + self._m[i] * self._g * y
        return U

    def energy_fn(self, coords):
        """energy function """
        assert len(coords) == self._dof * 2
        H = self.kinetic(coords) + self.potential(coords)
        return H

    def random_config(self):
        max_momentum = .8
        x0 = torch.zeros(self._obj * 2)
        for i in range(self._obj):
            theta = (.5 * np.pi) * torch.rand(1, ) + 0  # [0, 2pi]
            momentum = (2 * torch.rand(1, ) - 1) * max_momentum  # [-1, 1]*max_momentum
            x0[i] = theta
            x0[i + self._obj] = momentum
        return x0
