# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:20 PM
@desc:
"""
import torch


def train_collate_fn(batch):
    x0, t, X, y, E = zip(*batch)

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.stack(X, dim=0)
    t = torch.stack(t, dim=0)
    y = torch.stack(y, dim=0)

    input = (X, t)
    output = y
    return input, output


def val_collate_fn(batch):
    x0, t, X, y, E = zip(*batch)

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.stack(X, dim=0)
    t = torch.stack(t, dim=0)
    y = torch.stack(y, dim=0)

    input = (X, t)
    output = y
    return input, output


def test_collate_fn(batch):
    x0, t, X, y, E = zip(*batch)

    x0 = x0[0]  # (bs, D)
    t = t[0]  # (T,)

    X = torch.stack(X, dim=0).reshape(x0.size(0), len(t), x0.size(1))

    input = (x0, t)
    output = X
    return input, output
