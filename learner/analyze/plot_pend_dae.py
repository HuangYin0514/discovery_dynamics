# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/3/8 5:17 PM
@desc:
"""


def Pendulum2_L_dae_trajectory(ax, truth_pos, net_pos, title_label, *args, **kwargs):
    time = min(50, len(truth_pos) - 1)

    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel('$y$ ($m$)')
    for i in range(time - 2):
        ax.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                alpha=0.2 + 0.8 * (i + 1) / time)
        ax.plot(net_pos[i:i + 2, 2], net_pos[i:i + 2, 3], 'r-', label='_nolegend_', linewidth=2,
                alpha=0.2 + 0.8 * (i + 1) / time)  # net_pred
        # if i % (time // 2) == 0:  # pendulum in the middle state
        #     ax.plot([0, net_pos[i, 0]], [0, net_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
        #             alpha=0.2 + 0.8 * (i + 1) / time)
        #     ax.plot([net_pos[i, 0], net_pos[i, 2]], [net_pos[i, 1], net_pos[i, 3]], 'o-',
        #             color='brown', linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
        #     ax.scatter(net_pos[i, 0], net_pos[i, 1], s=50, linewidths=2, facecolors='gray',
        #                edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
        #     ax.scatter(net_pos[i, 2], net_pos[i, 3], s=50, linewidths=2, facecolors='gray',
        #                edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    last_time = time - 3
    ax.plot(truth_pos[last_time - 2:last_time, 2], truth_pos[last_time - 2:last_time, 3], 'k-', label='Ground truth',
            linewidth=2, alpha=1)
    ax.plot(net_pos[last_time - 2:last_time, 2], net_pos[last_time - 2:last_time, 3], 'r-', label='Net Prediction',
            linewidth=2, alpha=1)  # net_pred
    # ax.plot([0, net_pos[last_time, 0]], [0, net_pos[last_time, 1]], color='brown', linewidth=2, label='_nolegend_')
    # ax.plot([net_pos[last_time, 0], net_pos[last_time, 2]], [net_pos[last_time, 1], net_pos[last_time, 3]], 'o-',
    #         color='brown', linewidth=2, label='Pendulum')
    # ax.scatter(net_pos[last_time, 0], net_pos[last_time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
    #            label='_nolegend_', alpha=1, zorder=3)
    # ax.scatter(net_pos[last_time, 2], net_pos[last_time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
    #            label='_nolegend_', alpha=1, zorder=3)
    ax.legend(fontsize=12)
    ax.set_title(title_label)
