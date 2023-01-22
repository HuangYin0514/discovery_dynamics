# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/12 12:21 AM
@desc:
"""
import torch
from torch import nn, Tensor


def weights_init_xavier_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  # find the linear layer class
        nn.init.xavier_normal_(m.weight)
        # if m.bias is not None:
        #     nn.init.constant_(m.bias, 0.0)


def weights_init_orthogonal_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReshapeNet(nn.Module):
    def __init__(self, *args):
        super(ReshapeNet, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class CosSinNet(nn.Module):
    def __init__(self):
        super(CosSinNet, self).__init__()

    def forward(self, x):
        cos_ang_q, sin_ang_q = torch.cos(x), torch.sin(x)
        q = torch.cat([cos_ang_q, sin_ang_q], dim=-1)
        return q


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input
