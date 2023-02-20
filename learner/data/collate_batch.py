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

    X = torch.stack(X, dim=0).float()
    t = torch.stack(t, dim=0).float()
    y = torch.stack(y, dim=0).float()

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    t = torch.flatten(t, start_dim=0, end_dim=1)
    y = torch.flatten(y, start_dim=0, end_dim=1)

    # X.requires_grad = True

    input = (X, t)
    output = y
    return input, output


def val_collate_fn(batch):
    x0, t, h, X, y, E = zip(*batch)

    X = torch.stack(X, dim=0).float()
    t = torch.stack(t, dim=0).float()
    y = torch.stack(y, dim=0).float()

    # (N, 100, dof) -> (Nx100, dof)
    X = torch.flatten(X, start_dim=0, end_dim=1)
    t = torch.flatten(t, start_dim=0, end_dim=1)
    y = torch.flatten(y, start_dim=0, end_dim=1)

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
