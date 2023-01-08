# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/8 11:52 PM
@desc:
"""
import torch


def ham_J(M):
    """
    applies the J matrix to another matrix M.
    input: M (*,2nd,bs)

    J ->  # [ 0, I]
          # [-I, 0]

    output: J@M (*,2nd,bs)
    """
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2:, :], -M[..., : D // 2, :]], dim=-2)
    return JM
