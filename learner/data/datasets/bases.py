# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:33 PM
@desc:
"""
import abc

import numpy as np


class BaseDataset(abc.ABC):
    """
        Base class of dynamics dataset
    """

    def __init__(self):
        super(BaseDataset, self).__init__()

    def get_dynamics_data_info(self, data):
        return 0, 0, 0

    @abc.abstractmethod
    def print_dataset_statistics(self, ds):
        raise NotImplementedError


class BaseDynamicsDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def __init__(self):
        super(BaseDynamicsDataset, self).__init__()

    def print_dataset_statistics(self, ds):
        num_train_pids, num_train_imgs, num_train_cams = self.get_dynamics_data_info(ds)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  ----------------------------------------")


class BaseBodyDynamicsDataset(BaseDynamicsDataset):
    def __init__(self):
        super(BaseBodyDynamicsDataset, self).__init__()

        self.train_num = None
        self.test_num = None

        self._h = None
        self.solver = None

    def Init_data(self):
        self.__init_data()

    def __init_data(self):
        self.X_train, self.y_train = self.__generate_random(self.train_num, self._h)
        self.X_test, self.y_test = self.__generate_random(self.test_num, self._h)

    def __generate_random(self, num, h):
        x0 = self.random_config(num)
        X = self.__generate(x0, h)
        X = np.concatenate(X)
        y = np.asarray(list(map(lambda x: self.right_fn(None, x), X)))
        E = np.array([self.energy_fn(y) for y in X])
        return X, y

    def __generate(self, X, h):
        X = np.array(list(map(lambda x: self.solver.solve(x, h), X)))
        return X

    @abc.abstractmethod
    def random_config(self, num):
        pass

    @abc.abstractmethod
    def right_fn(self, t, x):
        pass

    @abc.abstractmethod
    def energy_fn(self, y):
        pass
