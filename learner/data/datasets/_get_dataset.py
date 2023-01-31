# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 11:14 PM
@desc:
"""
import os
import os.path as osp

import numpy as np

from learner.utils import download_file_from_google_drive
from .body3 import Body3
from .body3_L import Body3_L
from .pend_2 import Pendulum2
from .pend_2_L import Pendulum2_L


def choose_dataset(data_name, obj, dim, train_num, test_num):
    if data_name == 'Pendulum2':
        dataset = Pendulum2(train_num=train_num,
                            test_num=test_num,
                            obj=obj,
                            dim=dim)

    elif data_name == 'Pendulum2_L':
        dataset = Pendulum2_L(train_num=train_num,
                              test_num=test_num,
                              obj=obj,
                              dim=dim)
    elif data_name == 'Body3':
        dataset = Body3(train_num=train_num,
                        test_num=test_num,
                        obj=obj,
                        dim=dim)
    elif data_name == 'Body3_L':
        dataset = Body3_L(train_num=train_num,
                          test_num=test_num,
                          obj=obj,
                          dim=dim)
    else:
        raise NotImplementedError('{} not implemented'.format(data_name))
    return dataset


def get_dataset(data_name, taskname, obj, dim, download_data=False, **kwargs):
    print('=> Start get dataset.')
    dataset = choose_dataset(data_name, obj, dim, **kwargs)

    data_path = osp.join('./outputs/', taskname)

    # example: dataset_Pendulum2.npy
    filename = osp.join(data_path, 'dataset_{}.npy'.format(data_name))

    if download_data == 'True':
        print('=> Start downloading dataset.')
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        download_file_from_google_drive(dataset.dataset_url, filename)

    if os.path.exists(filename):
        print('=> Start loading dataset from {} .'.format(filename))
        dataset = np.load(filename, allow_pickle=True).item()
    else:
        print('=> Start generating dataset.')
        dataset.Init_data()
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        np.save(filename, dataset)

    print("=> {} loaded".format(data_name))
    dataset.print_dataset_statistics(dataset.train, dataset.test)

    return dataset
