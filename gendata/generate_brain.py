# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/28 8:35 PM
@desc:
"""

import os
import os.path as osp

from gendata.dataset.body3 import Body3
from gendata.dataset.body3_L import Body3_L
from gendata.dataset.pend_2 import Pendulum2
from gendata.dataset.pend_2_L import Pendulum2_L
from learner.utils import timing

__dataset_factory = {
    'Pendulum2': Pendulum2,
    'Pendulum2_L': Pendulum2_L,
    'Body3': Body3,
    'Body3_L': Body3_L,
}


def choose_dataset(dataset_name, obj, dim, **kwargs):
    if dataset_name not in __dataset_factory.keys():
        raise ValueError('Dataset \'{}\' is not implemented'.format(dataset_name))
    dataset = __dataset_factory[dataset_name](obj, dim)
    return dataset

@timing
def gen_dataset(data_name, taskname, obj, dim, train_num, val_num,test_num,
                device, dtype,
                download_data=False, generate_data=False,
                **kwargs):
    print('Start get dataset.')
    dataset = choose_dataset(data_name, obj, dim, **kwargs)
    dataset.device = device
    dataset.dtype = dtype

    # example: dataset_Pendulum2.npy
    data_path = osp.join('./outputs/', 'data', 'dataset_{}'.format(data_name))
    train_path = osp.join(data_path, 'train')
    val_path = osp.join(data_path, 'val')
    test_path = osp.join(data_path, 'test')

    os.makedirs(train_path) if not os.path.exists(train_path) else None
    os.makedirs(val_path) if not os.path.exists(val_path) else None
    os.makedirs(test_path) if not os.path.exists(test_path) else None

    print('Start generating dataset.')

    dataset.gen_data(train_num, dataset.t, train_path)
    dataset.gen_data(val_num, dataset.t, val_path)
    dataset.gen_data(test_num, dataset.test_t, test_path)
