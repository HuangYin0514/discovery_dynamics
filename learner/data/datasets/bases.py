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
        num_traj = len(data)
        x0, t, h, X, y, E = data[0]
        num_t = len(t)
        num_states = len(x0)
        return num_traj, num_t, num_states

    @abc.abstractmethod
    def print_dataset_statistics(self, train_ds, test_ds):
        raise NotImplementedError


class BaseDynamicsDataset(BaseDataset):

    def __init__(self):
        super(BaseDynamicsDataset, self).__init__()

    def print_dataset_statistics(self, train_ds, test_ds):
        num_train_traj, num_train_t, num_train_states = self.get_dynamics_data_info(train_ds)
        num_test_traj, num_test_t, num_test_states = self.get_dynamics_data_info(test_ds)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # traj| # t   | # states")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_traj, num_train_t, num_train_states))
        print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_traj, num_test_t, num_test_states ))
        print("  ----------------------------------------")
