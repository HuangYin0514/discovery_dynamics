# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/28 8:35 PM
@desc:
"""

import os
import os.path as osp

import numpy as np

from learner.data.datasets import choose_dataset
from learner.utils import download_file_from_google_drive


def gen_dataset(data_name, taskname, obj, dim, device, dtype,
                download_data=False, generate_data=False, **kwargs):
    print('Start get dataset.')
    dataset = choose_dataset(data_name, obj, dim, **kwargs)
    dataset.device = device
    dataset.dtype = dtype

    data_path = osp.join('./outputs/', 'data')

    # example: dataset_Pendulum2.npy
    filename = osp.join(data_path, 'dataset_{}'.format(data_name))

    if download_data == 'True':
        print('Start downloading dataset.')
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        download_file_from_google_drive(dataset.dataset_url, filename)

        if os.path.exists(filename):
            print('Start loading dataset from {} .'.format(filename))
            dataset = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset.Init_data()
        train_path = osp.join(filename, 'train')
        test_path = osp.join(filename, 'test')

        os.makedirs(data_path) if not os.path.exists(data_path) else None
        os.makedirs(train_path) if not os.path.exists(train_path) else None
        os.makedirs(test_path) if not os.path.exists(test_path) else None

        for data in dataset.train:
            x0, t, X, y, E = dataset

        np.save(filename, {'train': dataset.train, 'test': dataset.test})

    print("======> {} loaded".format(data_name))
    dataset.print_dataset_statistics(dataset.train, dataset.test)

    return dataset
