# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:33 PM
@desc:
"""
import abc
import os.path as osp

import numpy as np
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
        _, num_train_traj,states, min_t, max_t, len_t = data[0]
        return num_train_traj, len_t, min_t, max_t, states

    @abc.abstractmethod
    def print_dataset_statistics(self, name, data):
        raise NotImplementedError


class BaseDynamicsDataset(BaseDataset):

    def __init__(self):
        super(BaseDynamicsDataset, self).__init__()

    def print_dataset_statistics(self, name, data):
        num_train_traj, num_train_t, min_train_t, max_train_t, num_train_states = self.get_dynamics_data_info(data)

        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | # traj| # t -> [t_min, t_max]  | # states")
        print("  ----------------------------------------------------")
        print("  {}    | {:5d} | {:9d} -> [{:.3},{:.3}] | {:5d}".format(name,
                                                                        num_train_traj,
                                                                        num_train_t,
                                                                        min_train_t,
                                                                        max_train_t,
                                                                        num_train_states))
        print("  ----------------------------------------------------")


def read_data(data_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(data_path):
        raise IOError("{} does not exist".format(data_path))
    while not got_img:
        try:
            loaded_data = np.load(data_path, allow_pickle=True).item()
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(data_path))
            pass
    return loaded_data


class DynamicsDataset(Dataset):
    """learning dynamics Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        data_path,num_train_traj, states, min_t, max_t, len_t = self.dataset[0]
        self.readed_data = read_data(data_path)


    def __len__(self):
        return len(self.readed_data['x0'])

    def __getitem__(self, index):
        x0 = self.readed_data['x0'][index]
        X = self.readed_data['X'][index]
        dX = self.readed_data['dX'][index]
        t = self.readed_data['t']

        if self.transform is not None:
            x0 = self.transform(x0)
            X = self.transform(X)
            dX = self.transform(dX)
            t = self.transform(t)

        return x0, X, dX, t
