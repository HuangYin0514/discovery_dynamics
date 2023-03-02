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
from learner.utils import timing, download_file_from_google_drive

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
def gen_dataset(data_name, taskname, obj, dim, train_num, val_num, test_num,
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

    num_states = int(obj * dim)
    min_t = min(dataset.t)
    max_t = max(dataset.t)
    len_t = len(dataset.t)
    filename = 'dataset_{}_{}_{}_{}_{}.npy'.format(train_num, num_states, min_t, max_t, len_t)
    train_filename = osp.join(train_path, filename)
    num_states = int(obj * dim)
    min_t = min(dataset.t)
    max_t = max(dataset.t)
    len_t = len(dataset.t)
    filename = 'dataset_{}_{}_{}_{}_{}.npy'.format(val_num, num_states, min_t, max_t, len_t)
    val_filename = osp.join(val_path, filename)
    min_t = min(dataset.test_t)
    max_t = max(dataset.test_t)
    len_t = len(dataset.test_t)
    filename = 'dataset_{}_{}_{}_{}_{}.npy'.format(test_num, num_states, min_t, max_t, len_t)
    test_filename = osp.join(test_path, filename)

    if download_data == 'True':
        print('Start downloading dataset.')
        download_file_from_google_drive(dataset.train_url, train_filename)
        download_file_from_google_drive(dataset.val_url, val_filename)
        download_file_from_google_drive(dataset.test_url, test_filename)
    else:
        print('Start generating dataset.')
        dataset.gen_data(train_num, dataset.t, train_filename)
        dataset.gen_data(val_num, dataset.t, val_filename)
        dataset.gen_data(test_num, dataset.test_t, test_filename)
