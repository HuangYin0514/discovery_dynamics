# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:33 PM
@desc:
"""
import abc


class BaseDataset(abc.ABC):
    """
        Base class of dynamics dataset
    """

    def get_dynamics_data_info(self, data):
        return 0, 0, 0

    @abc.abstractmethod
    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseDynamicsDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, ds):
        num_train_pids, num_train_imgs, num_train_cams = self.get_dynamics_data_info(ds)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  ----------------------------------------")
