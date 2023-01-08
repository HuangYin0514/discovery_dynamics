# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 11:42 PM
@desc:
"""

import torchvision.transforms as T

from .to_tensor import To_Tensor


def build_transforms(is_train=True, **kwargs):
    if is_train:
        transform = T.Compose([
            To_Tensor(),
        ])
    else:
        transform = T.Compose([
            To_Tensor(),
        ])

    return transform
