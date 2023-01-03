# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:31 PM
@desc:
"""

from .pend_2 import PendulumData
from .body_3 import BodyData

__factory = {
    'PendulumData': PendulumData,
    'BodyData': BodyData,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in get_names():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
