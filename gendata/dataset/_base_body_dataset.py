# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:22 PM
@desc:
"""
import abc
import os.path as osp

import numpy as np
import torch
from matplotlib import pyplot as plt

from learner.data.datasets._bases import BaseDynamicsDataset
from learner.integrator import ODESolver


class BaseBodyDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDataset, self).__init__()

    def gen_data(self, sample_num, t, path):
        self.generate_random(sample_num, t, path)

    def generate_random(self, num, t, filename):
        x0 = self.random_config(num).clone().detach()  # (D, )
        X = self.ode_solve_traj(x0, t).clone().detach()  # (T, D)
        dX = torch.stack(list(map(lambda x: self(None, x), X))).clone().detach()  # (T, D)
        E = torch.stack([self.energy_fn(y) for y in X]).clone().detach()

        for i in range(num):
            plt.plot(E[i].cpu().detach().numpy())
        plt.show()

        dataset = {
            'x0': x0.cpu().numpy(),
            't': t.cpu().numpy(),
            'X': X.cpu().numpy(),
            'dX': dX.cpu().numpy(),
            'E': E.cpu().numpy()
        }


        np.save(filename, dataset)

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
