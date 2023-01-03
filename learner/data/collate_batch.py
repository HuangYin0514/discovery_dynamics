# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:20 PM
@desc:
"""
import torch


def train_collate_fn(batch):
    x0, t, h, X, y, E = zip(*batch)
    # pids = torch.tensor(X, dtype=torch.int64)
    return torch.stack(X, dim=0), y


def val_collate_fn(batch):
    x0, t, h, X, y, E = zip(*batch)
    return torch.stack(X, dim=0), y
