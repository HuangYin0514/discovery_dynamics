# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import math

import numpy as np
import torch
from torch import nn

from .base_body_dataset import BaseBodyDataset
from ...utils import dfx


class Pendulum2_L(BaseBodyDataset, nn.Module):
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
        super(Pendulum2_L, self).__init__()

        self.train_num = train_num
        self.test_num = test_num
        self.dataset_url = 'https://drive.google.com/file/d/1zi7l8yf5FURe4DuyuzE-T5s-kcdkBhOg/view?usp=share_link'

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
        _time_step = int((t_end - t0) / self.dt * 15)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 15.
        _time_step = int((t_end - t0) / self.dt * 15)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    def forward(self, t, coords):
        assert len(coords) == self._dof * 2

        res = self.derivative_analytical(coords)
        # res = self.derivative_lagrangian(coords)
        return res

    def derivative_analytical(self, coords):
        '''
                double pendulum dynamics from https://github.com/MilesCranmer/lagrangian_nns/blob/master/experiment_dblpend/physics.py
                :param state: the angles and angular velocities of the two masses
                :param t: dummy variable for runge-kutta calculation
                :returns: the angular velocities and accelerations of the two masses
                '''

        t1, t2, w1, w2 = torch.chunk(coords, 4, dim=0)
        l1, l2, m1, m2 = self._l[0], self._l[1], self._m[0], self._m[1]
        g = self._g

        a1 = (l2 / l1) * (m2 / (m1 + m2)) * torch.cos(t1 - t2)
        a2 = (l1 / l2) * torch.cos(t1 - t2)
        f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2 ** 2) * torch.sin(t1 - t2) - (g / l1) * torch.sin(t1)
        f2 = (l1 / l2) * (w1 ** 2) * torch.sin(t1 - t2) - (g / l2) * torch.sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        # return derivative of state
        return torch.cat([w1, w2, g1, g2], dim=0).float()

    def derivative_lagrangian(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        x, v = torch.chunk(coords, 2, dim=0)

        L = self.energy_fn(torch.cat([x, v], dim=0), L_constant=True)
        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((self._dof, self._dof))
        dxdvL = torch.zeros((self._dof, self._dof))

        for i in range(self._dof):
            dvidvL = dfx(dvL[i].sum(), v)
            dvdvL[i] += dvidvL

        for i in range(self._dof):
            dxidvL = dfx(dvL[i].sum(), x)
            dxdvL[i] += dxidvL

        dvdvL_inv = torch.linalg.inv(dvdvL)

        a = dvdvL_inv @ (dxL - dxdvL @ v)
        return torch.cat([v, a], dim=0).detach().clone()

    def kinetic(self, coords):
        """
          Coordinates consist of position and velocity -> (x, v)
        """
        T = 0.
        vx, vy = 0., 0.
        for i in range(self._dof):
            vx = vx + self._l[i] * coords[self._dof + i] * torch.cos(coords[i])
            vy = vy + self._l[i] * coords[self._dof + i] * torch.sin(coords[i])
            T = T + 0.5 * self._m[i] * (torch.pow(vx, 2) + torch.pow(vy, 2))
        return T

    def potential(self, coords):
        U = 0.
        y = 0.
        for i in range(self._obj):
            y = y - self._l[i] * torch.cos(coords[i])
            U = U + self._m[i] * self._g * y
        return U

    def energy_fn(self, coords, L_constant=False):
        """energy function """
        assert len(coords) == self._dof * 2

        if L_constant:
            L = self.kinetic(coords) - self.potential(coords)
            return L
        eng = self.kinetic(coords) + self.potential(coords)
        return eng

    def random_config(self):
        max_momentum = 10.
        x0 = torch.zeros(self._obj * 2)
        for i in range(self._obj):
            theta = (2 * np.pi - 0) * torch.rand(1, ) + 0  # [0, 2pi]
            momentum = (2 * torch.rand(1, ) - 1) * max_momentum  # [-1, 1]*max_momentum
            x0[i] = theta
            x0[i + self._obj] = momentum
        return x0
