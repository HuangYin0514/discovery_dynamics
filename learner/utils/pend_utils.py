import numpy as np
from matplotlib import pyplot as plt


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


def plot_pend_traj(ax, truth_pos, net_pos, net_name, plot_color_marker):
    time = min(100, len(net_pos) - 1)

    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel('$y$ ($m$)')
    for i in range(time - 3):
        ax.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                alpha=0.2 + 0.8 * (i + 1) / time)
        ax.plot(net_pos[i:i + 2, 2], net_pos[i:i + 2, 3],plot_color_marker, label='_nolegend_', linewidth=2,
                alpha=0.2 + 0.8 * (i + 1) / time)
        if i % (time // 2) == 0:
            ax.plot([0, net_pos[i, 0]], [0, net_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
                    alpha=0.2 + 0.8 * (i + 1) / time)
            ax.plot([net_pos[i, 0], net_pos[i, 2]], [net_pos[i, 1], net_pos[i, 3]], 'o-',
                    color='brown', linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
            ax.scatter(net_pos[i, 0], net_pos[i, 1], s=50, linewidths=2, facecolors='gray',
                       edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
            ax.scatter(net_pos[i, 2], net_pos[i, 3], s=50, linewidths=2, facecolors='gray',
                       edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    ax.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    ax.plot(net_pos[time - 2:time, 2], net_pos[time - 2:time, 3], plot_color_marker, label=net_name.title(),
            linewidth=2, alpha=1)
    ax.plot([0, net_pos[time, 0]], [0, net_pos[time, 1]], color='brown', linewidth=2, label='_nolegend_')
    ax.plot([net_pos[time, 0], net_pos[time, 2]], [net_pos[time, 1], net_pos[time, 3]], 'o-',
            color='brown', linewidth=2, label='Pendulum')
    ax.scatter(net_pos[time, 0], net_pos[time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
               label='_nolegend_', alpha=1, zorder=3)
    ax.scatter(net_pos[time, 2], net_pos[time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
               label='_nolegend_', alpha=1, zorder=3)
    ax.legend(fontsize=12)
