# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/13 12:12 PM
@desc:
"""
from .analyze_utils import plot_energy, plot_compare_energy, plot_compare_state, plot_field
from .plot_body import body3_trajectory
from .plot_pend import pend_trajectory
from .plot_pend_dae import Pendulum2_L_dae_trajectory

__factory = {
    'Pendulum2': pend_trajectory,
    'Pendulum2_L': pend_trajectory,
    'Pendulum2_L_dae': Pendulum2_L_dae_trajectory,
    'Body3': body3_trajectory,
    'Body3_L': body3_trajectory,
}


def plot_trajectory(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown trajectoryfn: {}".format(name))
    return __factory[name](*args, **kwargs)
