# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/7 10:39 AM
@desc:
"""
import torch

from .torchdiffeq import odeint_adjoint as odeint


# from .torchdiffeq import odeint


def ODESolver(func, y0, t, method='dopri5', rtol=1e-4, atol=1e-9, **kwargs):
    '''
    y0  # (bs, D)
    t   # (T,)
    out   # (T, bs, D)
    '''
    with torch.enable_grad():
        sol = odeint(func, y0, t, method=method, rtol=rtol, atol=atol, **kwargs)
    return sol
