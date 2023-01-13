# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/13 12:25 PM
@desc:
"""


def body3_trajectory(ax, true_q, pred_q, title_label, *args, **kwargs):
    obj = 3
    dim = 2
    ax.set_xlabel('$x\;(m)$')
    ax.set_ylabel('$y\;(m)$')
    for i in range(obj):
        ax.text(true_q[0, i * 2], true_q[0, i * 2 + 1], '{}'.format(i))
        ax.plot(true_q[:, 2 * i], true_q[:, 2 * i + 1], 'g--', label='body {} path'.format(i), linewidth=2)

        ax.text(pred_q[0, i * 2], pred_q[0, i * 2 + 1], '{}'.format(i))
        ax.plot(pred_q[:, 2 * i], pred_q[:, 2 * i + 1], 'r--', label='body {} path'.format(i), linewidth=2)
    ax.axis('equal')
    ax.set_title(title_label)
