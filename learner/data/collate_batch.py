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

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.cat(X, dim=0)
    t = torch.cat(t, dim=0)
    y = torch.cat(y, dim=0)

    # X = X[0]
    # t = t[0]
    # y = y[0]

    # X.requires_grad = True

    input = (X, t)
    output = y
    return input, output


def val_collate_fn(batch):
    x0, t, h, X, y, E = zip(*batch)

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.cat(X, dim=0)
    t = torch.cat(t, dim=0)
    y = torch.cat(y, dim=0)

    # X = X[0]
    # t = t[0]
    # y = y[0]

    # X.requires_grad = True

    input = (X, t)
    output = y
    return input, output


def test_collate_fn(batch):
    x0, t, h, X, y, E = zip(*batch)

    X = torch.stack(X, dim=0).float()

    x0 = torch.stack(x0, dim=0).float()  # (bs, D)
    x0.requires_grad = True

    t = t[0]  # (T,)
    y = torch.stack(y, dim=0).float()  # (bs, T, D)

    input = (x0, t)
    output = X
    return input, output
