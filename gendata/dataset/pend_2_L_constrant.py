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

from gendata.dataset._base_body_dataset import BaseBodyDataset
from learner.integrator import ODESolver
from learner.utils import dfx


class Pendulum2_L_constrant(BaseBodyDataset, nn.Module):

    def __init__(self, obj, dim, m=None, l=None, **kwargs):
        super(Pendulum2_L_constrant, self).__init__()

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
        t_end = 3.
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

        dvdvL_inv = torch.linalg.inv(dvdvL)

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
        bs, num_states = coords.shape
        assert num_states == self.dof * 2
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = coords[:, i * self.dim + 1]
            U = U + self.m[i] * self.g * y
        return U

    def energy_fn(self, coords):
        """energy function """
        eng = self.kinetic(coords) + self.potential(coords)
        return eng

    def body2globalCoords(self, coords):
        vx, vy = 0., 0.
        x, y = 0., 0.
        y0 = np.zeros(self.dof * 2)

        angle_coords_len = int(len(coords) / 2)
        for i in range(self.obj):
            x = x + self.l[i] * np.sin(coords[i])
            y = y - self.l[i] * np.cos(coords[i])
            vx = vx + self.l[i] * coords[angle_coords_len + i] * np.cos(coords[i])
            vy = vy + self.l[i] * coords[angle_coords_len + i] * np.sin(coords[i])

            y0[i * self.dim] = x
            y0[i * self.dim + 1] = y
            y0[self.dof + i * self.dim] = vx
            y0[self.dof + i * self.dim + 1] = vy
        return y0

    def random_config(self, num):
        x0_list = []
        for i in range(num):
            max_momentum = 10
            y0 = np.zeros(self.obj * 2)
            for i in range(self.obj):
                theta = (2 * np.random.rand()) * np.pi
                momentum = (2 * np.random.rand() - 1) * max_momentum
                y0[i] = theta
                y0[i + self.obj] = momentum
            # y0 = self.body2globalCoords(y0)

            y0 = [1, 0, 2, 0, 0, 0, 0, 0]
            x0_list.append(y0)
        x0 = np.stack(x0_list)
        return torch.tensor(x0, dtype=self.Dtype, device=self.Device)

    def ode_solve_traj(self, x0, t):
        x0 = x0.to(self.Device)
        t = t.to(self.Device)
        # At small step sizes, the differential equations exhibit stiffness and the rk4 solver cannot solve
        # the double pendulum task. Therefore, use dopri5 to generate training data.
        if len(t) == len(self.test_t):
            # test stages
            x = ODESolver(self, x0, t, method='dopri5').permute(1, 0, 2)  # (T, D) dopri5 rk4
        else:
            # train stages
            x = ODESolver(self, x0, t, method='dopri5').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return x
