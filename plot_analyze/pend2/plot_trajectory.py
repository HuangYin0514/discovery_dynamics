# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/25 2:59 PM
@desc:
"""

import numpy as np

from fig_env_set import *


def polar2xy(x):
    """
    Convert polar coordinates to x,y coordinates.

    Parameters
    ----------
    x : float
        Polar coordinates.
    """

    pos = np.zeros([x.shape[0], x.shape[1] * 2])
    for i in range(x.shape[1]):
        if i == 0:
            pos[:, 2 * i:2 * (i + 1)] += np.concatenate([np.sin(x[:, i:i + 1]), -np.cos(x[:, i:i + 1])], 1)
        else:
            pos[:, 2 * i:2 * (i + 1)] += pos[:, 2 * (i - 1):2 * i] + np.concatenate(
                [np.sin(x[:, i:i + 1]), -np.cos(x[:, i:i + 1])], 1)
    return pos


def plot_pend_trajectory(true_q, HnnModScale_q, baseline_q, HNN_q, ModLaNet_q):
    truth_pos = polar2xy(true_q)
    HnnModScale_pos = polar2xy(HnnModScale_q)
    baseline_pos = polar2xy(baseline_q)
    HNN_pos = polar2xy(HNN_q)
    ModLaNet_pos = polar2xy(ModLaNet_q)


    fig = plt.figure(figsize=(16, 4), dpi=DPI)

    tpad = 4
    time = time = min(300, len(truth_pos) - 1)
    legendsize = 12

    # plot ground truth -----------------------------------------------------------------
    plt.subplot(1, 4, 1)
    plt.xlabel('$x$ ($m$)');
    plt.ylabel('$y$ ($m$)')
    for i in range(time - 3):
        plt.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        if i % 200 == 0:
            plt.plot([0, truth_pos[i, 0]], [0, truth_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
                     alpha=0.2 + 0.8 * (i + 1) / time)
            plt.plot([truth_pos[i, 0], truth_pos[i, 2]], [truth_pos[i, 1], truth_pos[i, 3]], 'o-', color='brown',
                     linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
            plt.scatter(truth_pos[i, 0], truth_pos[i, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
            plt.scatter(truth_pos[i, 2], truth_pos[i, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    plt.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    plt.plot([0, truth_pos[time, 0]], [0, truth_pos[time, 1]], color='brown', linewidth=2, label='_nolegend_')
    plt.plot([truth_pos[time, 0], truth_pos[time, 2]], [truth_pos[time, 1], truth_pos[time, 3]], 'o-', color='brown',
             linewidth=2, label='Pendulum')
    plt.scatter(truth_pos[time, 0], truth_pos[time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(truth_pos[time, 2], truth_pos[time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)

    plt.xlim(min(truth_pos[:, 2]) - 1, max(truth_pos[:, 2]) + 1)
    plt.ylim(min(truth_pos[:, 3]), max(truth_pos[:, 3]) + 2)
    plt.legend(fontsize=legendsize)

    # plot SMHNet  -----------------------------------------------------------------
    plt.subplot(1, 4, 2)
    plt.xlabel('$x$ ($m$)');
    plt.ylabel('$y$ ($m$)')
    for i in range(time - 3):
        plt.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        plt.plot(HnnModScale_pos[i:i + 2, 2], HnnModScale_pos[i:i + 2, 3], 'g-.', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        if i % 200 == 0:
            plt.plot([0, HnnModScale_pos[i, 0]], [0, HnnModScale_pos[i, 1]], color='brown', linewidth=2,
                     label='_nolegend_',
                     alpha=0.2 + 0.8 * (i + 1) / time)
            plt.plot([HnnModScale_pos[i, 0], HnnModScale_pos[i, 2]], [HnnModScale_pos[i, 1], HnnModScale_pos[i, 3]],
                     'o-',
                     color='brown', linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
            plt.scatter(HnnModScale_pos[i, 0], HnnModScale_pos[i, 1], s=50, linewidths=2, facecolors='gray',
                        edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
            plt.scatter(HnnModScale_pos[i, 2], HnnModScale_pos[i, 3], s=50, linewidths=2, facecolors='gray',
                        edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    plt.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    plt.plot(HnnModScale_pos[time - 2:time, 2], HnnModScale_pos[time - 2:time, 3], 'g-.', label='SMHNet (Ours)',
             linewidth=2, alpha=1)
    plt.plot([0, HnnModScale_pos[time, 0]], [0, HnnModScale_pos[time, 1]], color='brown', linewidth=2,
             label='_nolegend_')
    plt.plot([HnnModScale_pos[time, 0], HnnModScale_pos[time, 2]], [HnnModScale_pos[time, 1], HnnModScale_pos[time, 3]],
             'o-',
             color='brown', linewidth=2, label='Pendulum')
    plt.scatter(HnnModScale_pos[time, 0], HnnModScale_pos[time, 1], s=50, linewidths=2, facecolors='gray',
                edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(HnnModScale_pos[time, 2], HnnModScale_pos[time, 3], s=50, linewidths=2, facecolors='gray',
                edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)

    plt.xlim(min(truth_pos[:, 2]) - 1, max(truth_pos[:, 2]) + 1)
    plt.ylim(min(truth_pos[:, 3]), max(truth_pos[:, 3]) + 2)
    plt.legend(fontsize=legendsize)

    # plot HNN  -----------------------------------------------------------------
    plt.subplot(1, 4, 3)
    plt.xlabel('$x$ ($m$)');
    plt.ylabel('$y$ ($m$)')
    for i in range(time - 3):
        plt.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        plt.plot(HNN_pos[i:i + 2, 2], HNN_pos[i:i + 2, 3], 'b--', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        if i % 200 == 0:
            plt.plot([0, HNN_pos[i, 0]], [0, HNN_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
                     alpha=0.2 + 0.8 * (i + 1) / time)
            plt.plot([HNN_pos[i, 0], HNN_pos[i, 2]], [HNN_pos[i, 1], HNN_pos[i, 3]], 'o-', color='brown', linewidth=2,
                     label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
            plt.scatter(HNN_pos[i, 0], HNN_pos[i, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
            plt.scatter(HNN_pos[i, 2], HNN_pos[i, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    plt.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    plt.plot(HNN_pos[time - 2:time, 2], HNN_pos[time - 2:time, 3], 'b--', label='HNN', linewidth=2, alpha=1)
    plt.plot([0, HNN_pos[time, 0]], [0, HNN_pos[time, 1]], color='brown', linewidth=2, label='_nolegend_')
    plt.plot([HNN_pos[time, 0], HNN_pos[time, 2]], [HNN_pos[time, 1], HNN_pos[time, 3]], 'o-', color='brown',
             linewidth=2, label='Pendulum')
    plt.scatter(HNN_pos[time, 0], HNN_pos[time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(HNN_pos[time, 2], HNN_pos[time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)

    plt.xlim(min(truth_pos[:, 2]) - 1, max(truth_pos[:, 2]) + 1)
    plt.ylim(min(truth_pos[:, 3]), max(truth_pos[:, 3]) + 2)
    plt.legend(fontsize=legendsize)

    # # plot baseline_pos  -----------------------------------------------------------------
    # plt.subplot(1, 4, 4)
    # plt.xlabel('$x$ ($m$)');
    # plt.ylabel('$y$ ($m$)')
    # for i in range(time - 3):
    #     plt.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
    #              alpha=0.2 + 0.8 * (i + 1) / time)
    #     plt.plot(baseline_pos[i:i + 2, 2], baseline_pos[i:i + 2, 3], 'r:', label='_nolegend_', linewidth=2,
    #              alpha=0.2 + 0.8 * (i + 1) / time)
    #     if i % 200 == 0:
    #         plt.plot([0, baseline_pos[i, 0]], [0, baseline_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
    #                  alpha=0.2 + 0.8 * (i + 1) / time)
    #         plt.plot([baseline_pos[i, 0], baseline_pos[i, 2]], [baseline_pos[i, 1], baseline_pos[i, 3]], '-',
    #                  color='brown',
    #                  linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
    #         plt.scatter(baseline_pos[i, 0], baseline_pos[i, 1], s=50, linewidths=2, facecolors='gray',
    #                     edgecolors='brown',
    #                     label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    #         plt.scatter(baseline_pos[i, 2], baseline_pos[i, 3], s=50, linewidths=2, facecolors='gray',
    #                     edgecolors='brown',
    #                     label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    # plt.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    # plt.plot(baseline_pos[time - 2:time, 2], baseline_pos[time - 2:time, 3], 'r:', label='Baseline', linewidth=2,
    #          alpha=1)
    # plt.plot([0, baseline_pos[time, 0]], [0, baseline_pos[time, 1]], color='brown', linewidth=2, label='_nolegend_')
    # plt.plot([baseline_pos[time, 0], baseline_pos[time, 2]], [baseline_pos[time, 1], baseline_pos[time, 3]], 'o-',
    #          color='brown',
    #          linewidth=2, label='Pendulum')
    # plt.scatter(baseline_pos[time, 0], baseline_pos[time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
    #             label='_nolegend_', alpha=1, zorder=3)
    # plt.scatter(baseline_pos[time, 2], baseline_pos[time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
    #             label='_nolegend_', alpha=1, zorder=3)
    #
    # plt.xlim(min(truth_pos[:, 2]) - 1, max(truth_pos[:, 2]) + 1)
    # plt.ylim(min(truth_pos[:, 3]), max(truth_pos[:, 3]) + 2)
    # plt.legend(fontsize=legendsize)

    # plot modlanet  -----------------------------------------------------------------
    plt.subplot(1, 4, 4)
    plt.xlabel('$x$ ($m$)');
    plt.ylabel('$y$ ($m$)')
    for i in range(time - 3):
        plt.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        plt.plot(ModLaNet_pos[i:i + 2, 2], ModLaNet_pos[i:i + 2, 3], 'r:', label='_nolegend_', linewidth=2,
                 alpha=0.2 + 0.8 * (i + 1) / time)
        if i % 200 == 0:
            plt.plot([0, ModLaNet_pos[i, 0]], [0, ModLaNet_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
                     alpha=0.2 + 0.8 * (i + 1) / time)
            plt.plot([ModLaNet_pos[i, 0], ModLaNet_pos[i, 2]], [ModLaNet_pos[i, 1], ModLaNet_pos[i, 3]], '-',
                     color='brown',
                     linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
            plt.scatter(ModLaNet_pos[i, 0], ModLaNet_pos[i, 1], s=50, linewidths=2, facecolors='gray',
                        edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
            plt.scatter(ModLaNet_pos[i, 2], ModLaNet_pos[i, 3], s=50, linewidths=2, facecolors='gray',
                        edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    plt.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    plt.plot(ModLaNet_pos[time - 2:time, 2], ModLaNet_pos[time - 2:time, 3], 'r:', label='Baseline', linewidth=2,
             alpha=1)
    plt.plot([0, ModLaNet_pos[time, 0]], [0, ModLaNet_pos[time, 1]], color='brown', linewidth=2, label='_nolegend_')
    plt.plot([ModLaNet_pos[time, 0], ModLaNet_pos[time, 2]], [ModLaNet_pos[time, 1], ModLaNet_pos[time, 3]], 'o-',
             color='brown',
             linewidth=2, label='Pendulum')
    plt.scatter(ModLaNet_pos[time, 0], ModLaNet_pos[time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(ModLaNet_pos[time, 2], ModLaNet_pos[time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                label='_nolegend_', alpha=1, zorder=3)

    plt.xlim(min(truth_pos[:, 2]) - 1, max(truth_pos[:, 2]) + 1)
    plt.ylim(min(truth_pos[:, 3]), max(truth_pos[:, 3]) + 2)
    plt.legend(fontsize=legendsize)

    plt.tight_layout()
    fig.savefig('{}/pend-2-integration.{}'.format(result_dir, FORMAT), bbox_inches="tight")
