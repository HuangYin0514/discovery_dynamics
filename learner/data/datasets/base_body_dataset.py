# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:22 PM
@desc:
"""
import abc

import numpy as np
import torch

from learner.data.datasets.bases import BaseDynamicsDataset
from learner.integrator import ODESolver


class BaseBodyDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDataset, self).__init__()

        self.train_num = None
        self.test_num = None

        self.t = None

    def Init_data(self):
        train_dataset = self.__generate_random(self.train_num)
        test_dataset = self.__generate_random(self.test_num)
        self.train = train_dataset
        self.test = test_dataset

    def __generate_random(self, num):
        dataset = []
        for _ in range(num):
            x0 = self.random_config()  # (D, )
            X = self.__generate(x0, self.t)  # (T, D)
            y = torch.stack(list(map(lambda x: self(None, x), X)))  # (T, D)
            E = torch.stack([self.energy_fn(y) for y in X])
            dataset.append((x0, self.t, self.dt, X, y, E))
            # from matplotlib import pyplot as plt
            # plt.plot(E)
            # plt.show()
        return dataset

    def __generate(self, x0, t):
        x = ODESolver(self, x0, t, method='dopri5')  # (T, D)
        return x

    @abc.abstractmethod
    def random_config(self, num):
        pass

    @abc.abstractmethod
    def energy_fn(self, y):
        pass
