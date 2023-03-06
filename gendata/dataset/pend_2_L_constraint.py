# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import torch
from torch import nn

from gendata.dataset._base_body_dataset import BaseBodyDataset
from learner.integrator import ODESolver


class Pendulum2_L_constraint(BaseBodyDataset, nn.Module):

    def __init__(self, obj, dim, m=None, l=None, **kwargs):
        super(Pendulum2_L_constraint, self).__init__()

        self.train_url = 'https://drive.google.com/file/d/1kTP5WtPT78rX7HBcMo2BBU9ug6RGpZ8E/view?usp=share_link'
        self.val_url = 'https://drive.google.com/file/d/1RXTP-fxTECHY6ZCP_rqkaL90k8EQd_f7/view?usp=share_link'
        self.test_url = 'https://drive.google.com/file/d/1kTP5WtPT78rX7HBcMo2BBU9ug6RGpZ8E/view?usp=share_link'

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self.m = [1 for i in range(obj)]
        self.l = [1 for i in range(obj)]
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom

        t0 = 0.

        t_end = 1.
        dt = 0.01
        _time_step = int((t_end - t0) / dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 3.
        dt = 0.01
        _time_step = int((t_end - t0) / dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    def forward(self, t, coords):
        return

    def kinetic(self, coords):

        return

    def potential(self, coords):

        return

    def energy_fn(self, coords):
        """energy function """
        H = self.kinetic(coords) + self.potential(coords)
        return H

    def random_config(self, num):
        return

    def ode_solve_traj(self, x0, t):
        x0 = x0.to(self.Device)
        t = t.to(self.Device)
        # At small step sizes, the differential equations exhibit stiffness and the rk4 solver cannot solve
        # the double pendulum task. Therefore, use dopri5 to generate training data.
        if len(t) == len(self.test_t):
            # test stages
            x = ODESolver(self, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        else:
            # train stages
            x = ODESolver(self, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return x
