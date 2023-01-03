# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:29 PM
@desc:
"""
import torch

from . import datasets
from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset
from .datasets.dataset_loader import DynamicsDataset


def getDataLoader(dataset_name, dataset_path, args):
    num_workers = 4

    '''transforms'''
    train_transforms = None
    val_transforms = None

    dataset = init_dataset(dataset_name, root=dataset_path).Init_data()

    '''train loader'''
    train_set = DynamicsDataset(dataset.train, train_transforms)
    len_train_set = len(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=train_collate_fn)

    '''test loader'''
    test_set = DynamicsDataset(dataset.test, val_transforms)
    len_test_set = len(test_set)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=len_test_set, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    return train_loader, test_loader
