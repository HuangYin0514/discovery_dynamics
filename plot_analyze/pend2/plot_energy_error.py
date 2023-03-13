# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 7:12 PM
@desc:
"""
from functools import partial

import numpy as np
import torch

from fig_env_set import *


def energy_err_fn(x, y, energy_function):
    err_list = []
    for x_, y_ in zip(x, y):
        eng_x = energy_function(x_).reshape(-1, 1)
        eng_y = energy_function(y_).reshape(-1, 1)
        # eng_y = eng_y[0].repeat(len(eng_y)) # 与真实的eng对比

        rel_err = (eng_x - eng_y).norm(dim=1)
        err_list.append(rel_err)

    E_err = torch.stack(err_list)
    return E_err


def processData(error_fun_H, energy_function, gt_data, baseline_data, HNN_data, HnnModScale_data):
    error_fun_H = energy_err_fn
    error_fun = partial(error_fun_H, energy_function=energy_function)
    # format
    gt_data = torch.from_numpy(gt_data)
    baseline_data = torch.from_numpy(baseline_data)
    HNN_data = torch.from_numpy(HNN_data)
    HnnModScale_data = torch.from_numpy(HnnModScale_data)

    # compute the error of mean and standard deviation
    baseline_error = error_fun(gt_data, baseline_data)
    HNN_error = error_fun(gt_data, HNN_data)
    HnnModScale_error = error_fun(gt_data, HnnModScale_data)

    # reformat
    baseline_error = baseline_error.numpy()
    HNN_error = HNN_error.numpy()
    HnnModScale_error = HnnModScale_error.numpy()

    return baseline_error, HNN_error, HnnModScale_error


def plot_energy_error(*args, **kwargs):
    baseline_error, HNN_error, HnnModScale_error = processData(*args, **kwargs)

    legendsize = 12

    fig, ax = plt.subplots(figsize=[8, 2.5], dpi=DPI)

    epochs = np.linspace(0, 30, num=baseline_error.shape[1])

    ax.plot(epochs, baseline_error.mean(0), label='Baseline', linestyle='solid')
    ax.fill_between(epochs, baseline_error.mean(0), baseline_error.mean(0) + baseline_error.std(0), alpha=0.3)

    ax.plot(epochs, HNN_error.mean(0), label='HNN', linestyle='dashed')
    ax.fill_between(epochs, HNN_error.mean(0), HNN_error.mean(0) + HNN_error.std(0), alpha=0.3)

    ax.plot(epochs, HnnModScale_error.mean(0), label='SMHNet', linestyle='dotted')
    ax.fill_between(epochs, HnnModScale_error.mean(0), HnnModScale_error.mean(0) + HnnModScale_error.std(0), alpha=0.3)

    ax.legend(fontsize=legendsize)
    ax.set_yscale('log')
    ax.tick_params(axis="y", direction='in')  # , length=8)
    ax.tick_params(axis="x", direction='in')  # , length=8)
    # ax.set_ylim(top=1e0)
    ax.set_xlim([-1, 32])
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.annotate('$t$', xy=(0.98, -0.025), ha='left', va='top', xycoords='axes fraction')
    # ax.annotate('MSE', xy=(-0.07, 1.05), xytext=(-15, 2), ha='left', va='top', xycoords='axes fraction',
    #             textcoords='offset points')

    fig.savefig('{}/pend2-eng-err.png'.format(result_dir))
