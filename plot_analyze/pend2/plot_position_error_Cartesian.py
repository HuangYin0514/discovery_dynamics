# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 7:12 PM
@desc:
"""
import numpy as np
import torch

from fig_env_set import *


def position_err_fn(x, y):
    bs, times, states = x.shape
    dof = int(states // 2)

    err_list = []
    for x_, y_ in zip(x, y):
        x_position = x_[..., :dof]
        y_position = y_[..., :dof]

        rel_err = (x_position - y_position).norm(dim=1)

        err_list.append(rel_err)

    err = torch.stack(err_list)
    return err


def processData(error_fun, gt_data, SCLNN_data):
    # error_fun=position_err_fn
    # format
    gt_data = torch.from_numpy(gt_data)
    SCLNN_data = torch.from_numpy(SCLNN_data)

    # compute the error of mean and standard deviation
    SCLNN_error = error_fun(gt_data, SCLNN_data)

    # reformat
    SCLNN_error = SCLNN_error.numpy()

    return SCLNN_error


def plot_position_error_Cartesian(*args, **kwargs):
    SCLNN_error = processData(*args, **kwargs)

    legendsize = 12

    fig, ax = plt.subplots(figsize=[8, 2.5], dpi=DPI)

    epochs = np.linspace(0, len(SCLNN_error), num=SCLNN_error.shape[1])

    ax.plot(epochs, SCLNN_error.mean(0), label='Baseline', linestyle='solid')
    ax.fill_between(epochs, SCLNN_error.mean(0), SCLNN_error.mean(0) + SCLNN_error.std(0), alpha=0.3)


    ax.legend(fontsize=legendsize)
    ax.set_yscale('log')
    ax.tick_params(axis="y", direction='in')  # , length=8)
    ax.tick_params(axis="x", direction='in')  # , length=8)
    # ax.set_ylim(top=1e1)
    ax.set_xlim([-1, 32])
    # ax.set_yticks([0.01, 0.1, 1])
    ax.annotate('$t$', xy=(0.98, -0.025), ha='left', va='top', xycoords='axes fraction')
    ax.annotate('MSE', xy=(-0.07, 1.05), xytext=(-15, 2), ha='left', va='top', xycoords='axes fraction',
                textcoords='offset points')

    fig.savefig('{}/pend2-pos-err.png'.format(result_dir))
