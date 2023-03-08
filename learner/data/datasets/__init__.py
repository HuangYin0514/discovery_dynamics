# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 11:14 PM
@desc:
"""
from .pend_2 import Pendulum2
from .pend_2_L_dae import Pendulum2_L_dae

__dataset_factory = {
    'Pendulum2': Pendulum2,
    'Pendulum2_L': Pendulum2,
    'Pendulum2_L_dae': Pendulum2_L_dae,
}


def get_dataset(dataset_name, root, **kwargs):
    if dataset_name not in __dataset_factory.keys():
        raise ValueError('Dataset \'{}\' is not implemented'.format(dataset_name))
    dataset = __dataset_factory[dataset_name](root)
    return dataset
