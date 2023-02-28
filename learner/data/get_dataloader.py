# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:55 PM
@desc:
"""

import torch

from .collate_batch import train_collate_fn, val_collate_fn, test_collate_fn
from .datasets import get_dataset
from .datasets._bases import DynamicsDataset
from .transforms import build_transforms
from ..utils import timing


@timing
def get_dataloader(data_name, num_workers=0, **kwargs):
    num_workers = num_workers

    '''transforms'''
    train_transforms = build_transforms(is_train=True, **kwargs)
    val_transforms = build_transforms(is_train=False, **kwargs)

    dataset = get_dataset(data_name, **kwargs)

    '''dataset'''
    train_set = DynamicsDataset(dataset.train, train_transforms)
    len_train_set = len(train_set)
    val_set = DynamicsDataset(dataset.test, val_transforms)
    len_val_set = len(val_set)

    '''dataloader'''
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=len_train_set, shuffle=True,
        num_workers=num_workers, collate_fn=train_collate_fn)

    # train_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=len_val_set, shuffle=False,
    #     num_workers=num_workers, collate_fn=val_collate_fn
    # )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=len_val_set, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        val_set, batch_size=2, shuffle=False,
        num_workers=num_workers, collate_fn=test_collate_fn
    )

    return dataset, train_loader, val_loader, test_loader
