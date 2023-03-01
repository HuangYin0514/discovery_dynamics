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
from tqdm import tqdm

from learner.data.datasets._bases import BaseDynamicsDataset


class BaseBodyDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDataset, self).__init__()

    def gen_data(self, sample_num, t, path):
        self.generate_random(sample_num, t, path)

    def generate_random(self, num, t, path):
        dataset = []
        x0s =[]
        Xs= []
        dXs = []
        Es = []
        pbar = tqdm(range(num), desc='Processing')
        for i in pbar:
            x0 = self.random_config()  # (D, )
            X = self.ode_solve_traj(x0, t)  # (T, D)
            dX = self(None, X).clone().detach()  # (T, D)
            E = self.energy_fn(X)

            x0s.append(x0)
            Xs.append(X)
            dXs.append(dX)
            Es.append(E)

            plt.plot(E.cpu().detach().numpy())

        num_states = X.shape[-1]
        min_t = min(t)
        max_t = max(t)
        len_t = len(t)
        plt.show()

        dataset = {
            'x0': x0s,
            't': t,
            'X': Xs,
            'dX': dXs,
            'E': Es
        }

        filename = osp.join(path, 'dataset_{}_{}_{}_{}.npy'.format(num_states, min_t, max_t, len_t))
        np.save(filename, dataset)


    @abc.abstractmethod
    def random_config(self):
        pass

    @abc.abstractmethod
    def energy_fn(self, y):
        pass
