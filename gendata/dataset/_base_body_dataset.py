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
from matplotlib import pyplot as plt

from learner.data.datasets._bases import BaseDynamicsDataset
from learner.integrator import ODESolver


class BaseBodyDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDataset, self).__init__()

    def gen_data(self,train_num, t, path):
        self.generate_random(train_num, t, path)

    def generate_random(self, num, t, path):
        for i in range(num):
            x0 = self.random_config(1).clone().detach()  # (D, )
            X = self.ode_solve_traj(x0, t).clone().detach()  # (T, D)
            y = self(None, X[0]).clone().detach()  # (T, D)
            E = self.energy_fn(X[0]).clone().detach()

            dataset = {
                'x0': x0.numpy(),
                't': t.numpy(),
                'X': X[0].numpy(),
                'dX': y.numpy(),
                'E': E.numpy()
            }

            num_states = X.shape[-1]
            min_t = min(t)
            max_t = max(t)
            len_t = len(t)
            filename = osp.join(path, 'dataset_{}_{}_{}_{}_{}'.format(num_states, min_t,max_t,len_t,i))
            np.save(filename, dataset)

            plt.plot(E.cpu().detach().numpy())
        plt.show()

    @abc.abstractmethod
    def random_config(self, num):
        pass

    @abc.abstractmethod
    def energy_fn(self, y):
        pass
