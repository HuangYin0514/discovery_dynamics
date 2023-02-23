# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:22 PM
@desc:
"""
import abc

import torch
from matplotlib import pyplot as plt

from learner.data.datasets._bases import BaseDynamicsDataset
from learner.integrator import ODESolver


class BaseBodyDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDataset, self).__init__()

        self.train_num = None
        self.test_num = None

        self.t = None
        self.test_t = None

    def Init_data(self):
        train_dataset = self.generate_random(self.train_num, self.t)
        test_dataset = self.generate_random(self.test_num, self.test_t)
        self.train = train_dataset
        self.test = test_dataset

    def generate_random(self, num, t):
        x0 = self.random_config(num)  # (bs, D)
        X = self.ode_solve_traj(x0, t).reshape(-1, self.dof * 2).clone().detach()  # (bs x T, D)
        dy = self(None, X).clone().detach()  # (bs, T, D)
        E = self.energy_fn(X).reshape(num, len(t))
        dataset = (x0, t, X, dy, E)

        for i in range(num):
            plt.plot(E[i].cpu().detach().numpy())
        plt.show()
        return dataset

    def ode_solve_traj(self, x0, t):
        x0 = x0.to(self.Device)
        t = t.to(self.Device)
        x = ODESolver(self, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return x

    @abc.abstractmethod
    def random_config(self, num):
        pass

    @abc.abstractmethod
    def energy_fn(self, y):
        pass
