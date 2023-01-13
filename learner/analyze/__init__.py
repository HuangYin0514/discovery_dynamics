# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/13 12:12 PM
@desc:
"""
from .plot_body import body_trajectory
from .plot_pend import pend_trajectory
from .analyze_utils import plot_energy, plot_compare_energy, plot_compare_state, plot_field

__factory = {
    'Pendulum2': pend_trajectory,
    # 'cuhk03': CUHK03,
    'Body3': body_trajectory,
}


def get_plot_trajectory(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown trajectoryfn: {}".format(name))
    return __factory[name](*args, **kwargs)
