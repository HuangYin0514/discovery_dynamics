# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:20 PM
@desc:
"""
import torch


def train_collate_fn(batch):
    x0, X, dX, t= zip(*batch)

    X = torch.stack(X, dim=0).float()
    t = torch.stack(t, dim=0).float()
    dX = torch.stack(dX, dim=0).float()

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    t = torch.flatten(t, start_dim=0, end_dim=1)
    dX = torch.flatten(dX, start_dim=0, end_dim=1)

    input = (X, t)
    output = dX
    return input, output


def val_collate_fn(batch):
    x0, X, dX, t = zip(*batch)

    X = torch.stack(X, dim=0).float()
    t = torch.stack(t, dim=0).float()
    dX = torch.stack(dX, dim=0).float()

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    t = torch.flatten(t, start_dim=0, end_dim=1)
    dX = torch.flatten(dX, start_dim=0, end_dim=1)

    input = (X, t)
    output = dX
    return input, output


def test_collate_fn(batch):
    x0, X, dX, t = zip(*batch)

    x0 = torch.stack(x0, dim=0).float()
    X = torch.stack(X, dim=0).float()

    # x0 = torch.flatten(x0, start_dim=0, end_dim=1)
    # X = torch.flatten(X, start_dim=0, end_dim=1)
    # t = torch.flatten(t, start_dim=0, end_dim=1)

    input = (x0, t[0])
    output = X
    return input, output
