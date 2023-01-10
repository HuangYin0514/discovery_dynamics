# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/9 2:50 PM
@desc:
"""
import torch
from torch.autograd.functional import jacobian

from ..utils import ham_J


def symplectic_prior_reg(net, z, alpha=1e8):
    """ Computes the symplectic prior regularization term
    Args:
        z: N x T x D Tensor representing the state
    Returns: ||(JDF)^T - JDF||^2
    """
    with torch.enable_grad():
        D = z.shape[-1]
        F = lambda z: net(None, z).sum(0)
        DF = jacobian(F, z.reshape(-1, D), create_graph=True, vectorize=True)  # (D,NT,D)
        # JDF = (net.J @ DF.permute(1, 0, 2)).T  # (NT,D,D)
        JDF = ham_J(DF.permute(1, 0, 2).mT).mT  # (NT,D,D)
        reg = (JDF - JDF.transpose(-1, -2)).pow(2).mean()
    return reg / alpha
