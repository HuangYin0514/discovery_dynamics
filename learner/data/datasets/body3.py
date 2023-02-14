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
from ...utils import dfx


class Body3(BaseBodyDataset, nn.Module):
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
        super(Body3, self).__init__()

        self.train_num = train_num
        self.test_num = test_num

        # test time is 15s
        # self.dataset_url15 = 'https://drive.google.com/file/d/1rZDGAaeuJYnIvfvwEX1WtwlKLWjeRW1D/view?usp=share_link'
        # test time is 30s, test samples are 20
        self.dataset_url = 'https://drive.google.com/file/d/18qjMMTKB5-y8CTPuFcn2iat7UH5O_F1P/view?usp=share_link'
        # test time is 30s, test samples are 100
        # self.dataset_url = 'https://drive.google.com/file/d/1E8mdLioGy91W1XjLNEri0p3GliQVhMZB/view?usp=share_link'

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self.m = [1 for i in range(obj)]
        self.l = [1 for i in range(obj)]
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom

        self.dt = 0.05

        t0 = 0.
        t_end = 10.
        _time_step = int((t_end - t0) / self.dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        # t_end = 15.
        t_end = 30.
        _time_step = int((t_end - t0) / self.dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

        self.k = 1  # body equation parameter

    def forward(self, t, coords):
        assert len(coords) == self.dof * 2
        coords = coords.clone().detach().requires_grad_(True)
        grad_ham = dfx(self.energy_fn(coords), coords)
        dq, dp = grad_ham[self.dof:], -grad_ham[:self.dof]
        return torch.cat([dq, dp], dim=0).clone().detach()

    def kinetic(self, coords):
        s, num_states = coords.shape
        assert num_states == self.dof * 2
        x, p = torch.chunk(coords, 2, dim=1)

        T = 0.
        for i in range(self.obj):
            T = T + 0.5 * torch.sum(p[:, 2 * i:  2 * i + 2] ** 2, dim=1) / self.m[i]
        return T

    def potential(self, coords):
        s, num_states = coords.shape
        assert num_states == self.dof * 2
        q, p = torch.chunk(coords, 2, dim=1)

        k = self.k
        U = 0.
        for i in range(self.obj):
            for j in range(i):
                U = U - k * self.m[i] * self.m[j] / (
                        (q[:, 2 * i] - q[:, 2 * j]) ** 2 +
                        (q[:, 2 * i + 1] - q[:, 2 * j + 1]) ** 2) ** 0.5
        return U

    def energy_fn(self, coords):
        """能量函数"""
        assert (len(coords) == self.dof * 2)
        T, U = self.kinetic(coords), self.potential(coords)
        H = T + U
        return H

    @staticmethod
    def rotate2d(p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ p.reshape(2, 1)).squeeze()

    def random_config(self, num):
        # for n objects evenly distributed around the circle,
        # which means angle(obj_i, obj_{i+1}) = 2*pi/n
        # we made the requirement there that m is the same
        # for every obejct to simplify the formula.
        # But it can be improved.
        nu = 0.5
        min_radius = 1
        max_radius = 5
        system = 'hnn'

        x0_list = []
        for i in range(num):
            state = np.zeros(self.dof * 2)

            p0 = 2 * np.random.rand(2) - 1
            r = np.random.rand() * (max_radius - min_radius) + min_radius

            theta = 2 * np.pi / self.obj
            p0 *= r / np.sqrt(np.sum((p0 ** 2)))
            for i in range(self.obj):
                state[2 * i: 2 * i + 2] = self.rotate2d(p0, theta=i * theta)

            # # velocity that yields a circular orbit
            dirction = p0 / np.sqrt((p0 * p0).sum())
            v0 = self.rotate2d(dirction, theta=np.pi / 2)
            k = self.k / (2 * r)
            for i in range(self.obj):
                v = v0 * np.sqrt(
                    k * sum(
                        [self.m[j % self.obj] / np.sin((j - i) * theta / 2) for j in range(i + 1, self.obj + i)]))
                # make the circular orbits slightly chaotic
                if system == 'hnn':
                    v *= (1 + nu * (2 * np.random.rand(2) - 1))
                else:
                    v *= self.m[i] * (1 + nu * (2 * np.random.rand(2) - 1))
                state[self.dof + 2 * i: self.dof + 2 * i + 2] = self.rotate2d(v, theta=i * theta)
            x0_list.append(torch.tensor(state).float())
        x0 = torch.stack(x0_list)
        return x0
