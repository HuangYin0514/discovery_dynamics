# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/7 10:39 AM
@desc:
"""
from .torchdiffeq import odeint


def ODESolver(func, y0, t, method='dopri5', rtol=1e-7, atol=1e-9, **kwargs):
    return odeint(func, y0, t, method=method)
