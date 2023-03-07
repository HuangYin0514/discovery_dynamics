# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/1 12:46 AM
@desc:
"""
from .pend_2 import Pendulum2
from .pend_2_L import Pendulum2_L
from .pend_2_L_dea import Pendulum2_L_dea

__dataset_factory = {
    'Pendulum2': Pendulum2,
    'Pendulum2_L': Pendulum2_L,
    'Pendulum2_L_dea':Pendulum2_L_dea
}


def choose_dataset(dataset_name, obj, dim, **kwargs):
    if dataset_name not in __dataset_factory.keys():
        raise ValueError('Dataset \'{}\' is not implemented'.format(dataset_name))
    dataset = __dataset_factory[dataset_name](obj, dim)
    return dataset
