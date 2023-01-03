# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 11:59 PM
@desc:
"""
import torch


class To_Tensor(object):

    def __init__(self):
        super(To_Tensor, self).__init__()

    def __call__(self, x):
        x = torch.Tensor(x)
        return x
