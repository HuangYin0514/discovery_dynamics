# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:22 PM
@desc:
"""
import abc

import numpy as np

from learner.data.datasets.bases import BaseDynamicsDataset


class BaseBodyDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDataset, self).__init__()

        self.train_num = None
        self.test_num = None

        self._h = None
        self.solver = None

    def Init_data(self):
        self.__init_data()

    def __init_data(self):
        train_dataset = self.__generate_random(self.train_num, self._h)
        test_dataset = self.__generate_random(self.test_num, self._h)
        self.train = train_dataset
        self.test = test_dataset

    def __generate_random(self, num, h):
        dataset = []
        for _ in range(num):
            x0 = self.random_config()
            t, X = self.__generate(x0, h)
            y = np.asarray(list(map(lambda x: self.right_fn(None, x), X)))
            E = np.array([self.energy_fn(y) for y in X])
            dataset.append((x0, t, h, X, y, E))
        return dataset

    def __generate(self, x0, h):
        t, x = self.solver.solve(x0, h)
        return t, x

    @abc.abstractmethod
    def random_config(self):
        pass

    @abc.abstractmethod
    def right_fn(self, t, x):
        pass

    @abc.abstractmethod
    def energy_fn(self, y):
        pass
