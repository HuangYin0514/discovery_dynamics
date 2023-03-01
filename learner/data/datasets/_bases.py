# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:33 PM
@desc:
"""
import abc

import torch
from torch.utils.data import Dataset


class BaseDataset(abc.ABC):
    """
        Base class of dynamics dataset
    """

    def __init__(self):
        super(BaseDataset, self).__init__()

        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'cuda':
            self.cuda()
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'cuda':
            return torch.device('cuda')

    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    def get_dynamics_data_info(self, data):
        num_traj = len(data)
        x0, t, X, y, E = data[0]
        num_t = len(t)
        min_t = min(t)
        max_t = max(t)
        num_states = len(x0)
        return num_traj, num_t, min_t, max_t, num_states

    @abc.abstractmethod
    def print_dataset_statistics(self, train_ds, test_ds):
        raise NotImplementedError


class BaseDynamicsDataset(BaseDataset):

    def __init__(self):
        super(BaseDynamicsDataset, self).__init__()

    def print_dataset_statistics(self, train_ds, test_ds):
        num_train_traj, num_train_t, min_train_t, max_train_t, num_train_states = self.get_dynamics_data_info(train_ds)
        num_test_traj, num_test_t, min_test_t, max_test_t, num_test_states = self.get_dynamics_data_info(test_ds)

        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | # traj| # t -> [t_min, t_max]  | # states")
        print("  ----------------------------------------------------")
        print("  train    | {:5d} | {:9d} -> [{:.3},{:.3}] | {:5d}".format(num_train_traj,
                                                                           num_train_t,
                                                                           min_train_t,
                                                                           max_train_t,
                                                                           num_train_states))
        print("  test     | {:5d} | {:9d} -> [{:.3},{:.3}] | {:5d}".format(num_test_traj,
                                                                           num_test_t,
                                                                           min_test_t,
                                                                           max_test_t,
                                                                           num_test_states))
        print("  ----------------------------------------------------")


class DynamicsDataset(Dataset):
    """learning dynamics Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x0, t, X, y, E = self.dataset[index]

        if self.transform is not None:
            X = self.transform(X)
            t = self.transform(t)
            x0 = self.transform(x0)
            y = self.transform(y)

        return x0, t, X, y, E
