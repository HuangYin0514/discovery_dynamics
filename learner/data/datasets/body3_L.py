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


class Body3_L(BaseBodyDataset, nn.Module):

    def __init__(self, train_num, test_num, obj, dim, m=None, l=None, **kwargs):
        super(Body3_L, self).__init__()

        self.train_num = train_num
        self.test_num = test_num
        self.dataset_url = ''

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

        t_end = 15.
        _time_step = int((t_end - t0) / self.dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

        self.k = 1  # body equation parameter

    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.size(0)
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # Calculate the potential energy for i-th element ------------------------------------------------------------
        U = self.potential(torch.cat([x, v], dim=-1))

        # Calculate the kinetic --------------------------------------------------------------
        T = self.kinetic(torch.cat([x, v], dim=-1))

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        L = T - U
        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)
        dxdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.dof):
            dvidvL = dfx(dvL[:, i].sum(), v)
            dvdvL[:, i, :] += dvidvL

        for i in range(self.dof):
            dxidvL = dfx(dvL[:, i].sum(), x)
            dxdvL[:, i, :] += dxidvL

        dvdvL_inv = torch.linalg.pinv(dvdvL)

        a = dvdvL_inv @ (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))  # (bs, a_dim, 1)
        a = a.squeeze(2)
        return torch.cat([v, a], dim=1)

    def kinetic(self, coords):
        """Kinetic energy"""
        s, num_states = coords.shape
        assert num_states == self.dof * 2
        x, v = torch.chunk(coords, 2, dim=1)

        T = 0.
        for i in range(self.obj):
            T = T + 0.5 * self.m[i] * torch.sum(v[:, 2 * i: 2 * i + 2] ** 2, dim=1)
        return T

    def potential(self, coords):
        s, num_states = coords.shape
        assert num_states == self.dof * 2
        x, v = torch.chunk(coords, 2, dim=1)

        k = self.k
        U = 0.
        for i in range(self.obj):
            for j in range(i):
                U = U - k * self.m[i] * self.m[j] / (
                        (x[:, 2 * i] - x[:, 2 * j]) ** 2 +
                        (x[:, 2 * i + 1] - x[:, 2 * j + 1]) ** 2) ** 0.5
        return U

    def energy_fn(self, coords):
        """energy function """
        eng = self.kinetic(coords) + self.potential(coords)
        return eng

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
