# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/8 9:59 PM
@desc:
"""


def L1_loss(x, y):
    return (x - y).abs().mean()
