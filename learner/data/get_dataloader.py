# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:55 PM
@desc:
"""
import os

import numpy as np
import torch

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset
from .datasets.dynamics_dataset import DynamicsDataset
from .transforms import build_transforms
from ..utils import download_file_from_google_drive, timing

import os.path as osp


def __get_data(dataset, data_name, taskname, tasktype, obj, dim, download_data=False, **kwargs):
    data_path = osp.join('./outputs/', taskname)

    # example: dataset_pend_2_1_hamilton.npy
    filename = osp.join(data_path, 'dataset_{}_{}_{}_hamilton.npy'.format(tasktype, obj, dim))

    if download_data:
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
        # np.save(filename, dataset)

    print("=> {} loaded".format(data_name))
    dataset.print_dataset_statistics(dataset.train, dataset.test)

    return dataset


@timing
def get_dataloader(data_name, num_workers=0, **kwargs):
    num_workers = num_workers

    '''transforms'''
    train_transforms = build_transforms(is_train=True, **kwargs)
    val_transforms = build_transforms(is_train=False, **kwargs)

    dataset = init_dataset(data_name, **kwargs)
    dataset = __get_data(dataset, data_name, **kwargs)

    '''train loader'''
    train_set = DynamicsDataset(dataset.train, train_transforms)
    len_train_set = len(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=len_train_set, shuffle=True,
        num_workers=num_workers, collate_fn=train_collate_fn)

    '''test loader'''
    test_set = DynamicsDataset(dataset.test, val_transforms)
    len_test_set = len(test_set)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=len_test_set, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    return train_loader, test_loader
