# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/11 10:18 PM
@desc:
"""
from torch import nn


class ReshapeNet(nn.Module):
    def __init__(self, *args):
        super(ReshapeNet, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
