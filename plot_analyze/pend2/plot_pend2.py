# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 2:38 PM
@desc:
"""
import os
import sys
from functools import partial

import numpy as np
from matplotlib import pyplot as plt

from fig_env_set import result_dir
from plot_analyze.pend2.plot_energy_error import plot_energy_error
from plot_analyze.pend2.plot_trajectory import plot_pend_trajectory
from plot_position_error import plot_position_error

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append(PARENT_DIR)

import learner as ln


def main(path='./data'):
    # read data --------------------------------
    # result_gt_Pendulum2 +++
    data_path = path + '/result_gt_Pendulum2.npy'
    gt_data = np.load(data_path)
    # result_Baseline_pend2 +++
    data_path = path + '/result_Baseline_pend2.npy'
    baseline_data = np.load(data_path)
    # result_HNN_pend2 +++
    data_path = path + '/result_HNN_pend2.npy'
    HNN_data = np.load(data_path)
    # result_HnnModScale_pend2 +++
    data_path = path + '/result_HnnModScale_pend2.npy'
    HnnModScale_data = np.load(data_path)
    # result_ModLaNet_pend2 +++
    # data_path = path + '/result_HnnModScale_pend2.npy'
    data_path = path + '/result_ModLaNet_pend2.npy'
    ModLaNet_data = np.load(data_path)

    # plot trajectory --------------------------------
    index = 5
    gt_sample = gt_data[index]
    baseline_sample = baseline_data[index]
    HNN_sample = HNN_data[index]
    HnnModScale_sample = HnnModScale_data[index]
    ModLaNet_sample = ModLaNet_data[index]

    true_q, _ = np.split(gt_sample, 2, axis=-1)
    baseline_q, _ = np.split(baseline_sample, 2, axis=-1)
    HNN_q, _ = np.split(HNN_sample, 2, axis=-1)
    HnnModScale_q, _ = np.split(HnnModScale_sample, 2, axis=-1)
    ModLaNet_q, _ = np.split(ModLaNet_sample, 2, axis=-1)

    plot_pend_trajectory(true_q, HnnModScale_q, baseline_q, HNN_q,ModLaNet_q)

    # plot position error --------------------------------
    error_fun = ln.metrics.accuracy.position_err_fn
    plot_position_error(error_fun, gt_data, baseline_data, HNN_data, HnnModScale_data)

    # plot position error --------------------------------
    error_fun_H = ln.metrics.accuracy.energy_err_fn

    error_fun_H = partial(error_fun_H,
                          energy_function=ln.data.datasets.Pendulum2(train_num=1, test_num=1, obj=2, dim=1).energy_fn)
    plot_energy_error(error_fun_H, gt_data, baseline_data, HNN_data, HnnModScale_data)

    plt.show()


if __name__ == '__main__':
    os.makedirs(result_dir) if not os.path.isdir(result_dir) else None
    main()
    print('done')
