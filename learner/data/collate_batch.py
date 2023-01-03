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

    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)

    # todo: (N, 100, dof) -> (Nx100, dof)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    y = torch.flatten(y, start_dim=0, end_dim=1)

    X.requires_grad = True

    return X, y


def val_collate_fn(batch):
    x0, t, h, X, y, E = zip(*batch)

    X = torch.stack(X, dim=0)
    y = torch.stack(y, dim=0)

    # todo: (N, 100, dof) -> (Nx100, dof)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    y = torch.flatten(y, start_dim=0, end_dim=1)

    X.requires_grad = True
    return X, y
