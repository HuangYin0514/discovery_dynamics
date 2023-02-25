# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 2:38 PM
@desc:
"""
import os

import numpy as np
from matplotlib import pyplot as plt

from plot_analyze.pend2.fig_env_set import result_dir
from plot_trajectory import plot_pend_trajectory


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

    # plot trajectory --------------------------------
    index = 1
    gt_sample = gt_data[index]
    baseline_sample = baseline_data[index]
    HNN_sample = HNN_data[index]
    HnnModScale_sample = HnnModScale_data[index]

    true_q, _ = np.split(gt_sample, 2, axis=-1)
    baseline_q, _ = np.split(baseline_sample, 2, axis=-1)
    HNN_q, _ = np.split(HNN_sample, 2, axis=-1)
    HnnModScale_q, _ = np.split(HnnModScale_sample, 2, axis=-1)

    plot_pend_trajectory(true_q, HnnModScale_q, baseline_q, HNN_q)

    # plot position error --------------------------------
    plot_position_error(true_q, HnnModScale_q, baseline_q, HNN_q)




    plt.show()


if __name__ == '__main__':
    os.makedirs(result_dir) if not os.path.isdir(result_dir) else None
    main()
    print('done')
