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
        train_dataset = self.__generate_random(self.train_num, self._h)
        test_dataset = self.__generate_random(self.test_num, self._h)
        self.train = train_dataset
        self.test = test_dataset

    def __generate_random(self, num, h):
        dataset = []
        for _ in range(num):
            x0 = self.random_config()
            t, X = self.__generate(x0, h)
            y = self.right_fn(None, X)
            E = self.energy_fn(X)
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
