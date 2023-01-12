# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/12 1:44 PM
@desc:
"""
import argparse
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append(PARENT_DIR)

import learner as ln

parser = argparse.ArgumentParser(description=None)
# For general settings
parser.add_argument('--taskname', default='test_poor', type=str, help='Task name')
parser.add_argument('--seed', default=0, type=int, help='random seed')

# For task
parser.add_argument('--obj', default=2, type=int, help='number of objects')
parser.add_argument('--dim', default=1, type=int, help='coordinate dimension')

# data
parser.add_argument('--data_name', default='Pendulum2', type=str, help='choose dataset')
parser.add_argument('--train_num', default=0, type=int, help='the number of train sampling trajectories')
parser.add_argument('--test_num', default=0, type=int, help='the number of test sampling trajectories')
parser.add_argument('--download_data', default='False', type=str, help='Download dataset from Internet')
parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. ')

# net
parser.add_argument('--net_name', default='mechanicsNN', type=str, help='Select model to train')
parser.add_argument('--net_url', default='', type=str, help='Download net from Internet')

parser.set_defaults(feature=True)
args = parser.parse_args()

LINE_WIDTH = 2
LEGENDSIZE = 12


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


def plot_trajectory(ax, true_q, pred_q, title_label):
    truth_pos = ln.utils.polar2xy(true_q)
    net_pos = ln.utils.polar2xy(pred_q)

    time = min(400, len(truth_pos) - 1)

    ax.set_xlabel('$x$ ($m$)')
    ax.set_ylabel('$y$ ($m$)')
    for i in range(time - 3):
        ax.plot(truth_pos[i:i + 2, 2], truth_pos[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=2,
                alpha=0.2 + 0.8 * (i + 1) / time)
        ax.plot(net_pos[i:i + 2, 2], net_pos[i:i + 2, 3], 'r-', label='_nolegend_', linewidth=2,
                alpha=0.2 + 0.8 * (i + 1) / time)  # net_pred
        if i % (time // 2) == 0:  # pendulum in the middle state
            ax.plot([0, net_pos[i, 0]], [0, net_pos[i, 1]], color='brown', linewidth=2, label='_nolegend_',
                    alpha=0.2 + 0.8 * (i + 1) / time)
            ax.plot([net_pos[i, 0], net_pos[i, 2]], [net_pos[i, 1], net_pos[i, 3]], 'o-',
                    color='brown', linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / time)
            ax.scatter(net_pos[i, 0], net_pos[i, 1], s=50, linewidths=2, facecolors='gray',
                       edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
            ax.scatter(net_pos[i, 2], net_pos[i, 3], s=50, linewidths=2, facecolors='gray',
                       edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / time, 1), zorder=3)
    ax.plot(truth_pos[time - 2:time, 2], truth_pos[time - 2:time, 3], 'k-', label='Ground truth', linewidth=2, alpha=1)
    ax.plot(net_pos[time - 2:time, 2], net_pos[time - 2:time, 3], 'r-', label='Net Prediction',
            linewidth=2, alpha=1)  # net_pred
    ax.plot([0, net_pos[time, 0]], [0, net_pos[time, 1]], color='brown', linewidth=2, label='_nolegend_')
    ax.plot([net_pos[time, 0], net_pos[time, 2]], [net_pos[time, 1], net_pos[time, 3]], 'o-',
            color='brown', linewidth=2, label='Pendulum')
    ax.scatter(net_pos[time, 0], net_pos[time, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
               label='_nolegend_', alpha=1, zorder=3)
    ax.scatter(net_pos[time, 2], net_pos[time, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
               label='_nolegend_', alpha=1, zorder=3)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def plot_energy(ax, t, eng, potential_eng, kinetic_eng, title_label):
    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$E\;(J)$')
    ax.plot(t, potential_eng, 'y:', label='potential', linewidth=2)
    ax.plot(t, kinetic_eng, 'c-.', label='kinetic', linewidth=2)
    ax.plot(t, eng, 'g--', label='total', linewidth=2)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def plot_compare_energy(ax, t, true_eng, pred_eng, title_label):
    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$E\;(J)$')
    ax.plot(t, true_eng, 'g--', label='True Energy', linewidth=2)
    ax.plot(t, pred_eng, 'r--', label='Prediction Energy', linewidth=2)
    ax.plot(t, (true_eng - pred_eng) ** 2, 'b--', label='Energy Error', linewidth=2)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def plot_compare_state(ax, t, true_state, pred_state, title_label):
    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$State$')
    ax.plot(t, true_state, 'g--', label='True state', linewidth=2)
    ax.plot(t, pred_state, 'r--', label='Prediction state', linewidth=2)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def plot_field(ax, t, state_q, state_p, title_label):
    ax.set_xlabel('state $q$')
    ax.set_ylabel('state $p$')
    ax.plot(state_q, state_p, 'g--', label='state', linewidth=2)
    ax.legend(fontsize=12)
    ax.set_title(title_label)


def main():
    print('Task name: {}'.format(args.taskname))

    # seed
    ln.utils.init_random_state(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the device is:', device)

    # net ----------------------------------------------------------------
    arguments = {
        'taskname': args.taskname,
        'net_name': args.net_name,
        'obj': args.obj,
        'dim': args.dim,
        'net_url': args.net_url,
        'load_net_path': osp.join('./outputs/', args.taskname, 'model-{}.pkl'.format(args.taskname)),
        'device': device
    }
    net = ln.nn.get_model(**arguments)

    # data ----------------------------------------------------------------
    arguments = {
        'taskname': args.taskname,
        'data_name': args.data_name,
        'obj': args.obj,
        'dim': args.dim,
        'train_num': args.train_num,
        'test_num': args.test_num,
        'download_data': args.download_data,
        'num_workers': args.num_workers,
    }
    data = ln.data.get_dataloader(**arguments)

    dataset, train_loader, test_loader = data
    energy_fn = dataset.energy_fn
    kinetic_fn = dataset.kinetic
    potential_fn = dataset.potential

    test_data = next(iter(test_loader))  # get one data for test
    inputs, labels = test_data
    X, t = inputs
    X, t = X.to(device), t.to(device)
    labels = labels.to(device)

    # pred ----------------------------------------------------------------
    pred = net.integrate(X, t)  # (bs, T, states)

    # error ----------------------------------------------------------------
    err = ln.metrics.accuracy_fn(pred, labels, energy_fn)
    mse_err, rel_err, eng_err = err

    result = ('mse_err: {:.3e}'.format(mse_err)
              + '\n'
              + 'rel_err: {:.3e}'.format(rel_err)
              + '\n'
              + 'eng_err: {:.3e}'.format(eng_err))
    print(result)

    # solutions forms ----------------------------------------------------------------
    ground_true = labels[0]
    net_pred = pred[0]
    true_q, true_p = ground_true.chunk(2, dim=-1)  # (T, states)
    pred_q, pred_p = net_pred.chunk(2, dim=-1)  # (T, states)

    true_eng = torch.stack([energy_fn(i) for i in ground_true])
    true_kinetic_eng = torch.stack([kinetic_fn(i) for i in ground_true])
    true_potential_eng = torch.stack([potential_fn(i) for i in ground_true])
    pred_eng = torch.stack([energy_fn(i) for i in net_pred])
    pred_kinetic_eng = torch.stack([kinetic_fn(i) for i in net_pred])
    pred_potential_eng = torch.stack([potential_fn(i) for i in net_pred])

    t = t.detach().cpu().numpy()

    ground_true = ground_true.detach().cpu().numpy()
    true_q = true_q.detach().cpu().numpy()
    true_p = true_p.detach().cpu().numpy()
    true_eng = true_eng.detach().cpu().numpy()
    true_kinetic_eng = true_kinetic_eng.detach().cpu().numpy()
    true_potential_eng = true_potential_eng.detach().cpu().numpy()

    net_pred = net_pred.detach().cpu().numpy()
    pred_q = pred_q.detach().cpu().numpy()
    pred_p = pred_p.detach().cpu().numpy()
    pred_eng = pred_eng.detach().cpu().numpy()
    pred_kinetic_eng = pred_kinetic_eng.detach().cpu().numpy()
    pred_potential_eng = pred_potential_eng.detach().cpu().numpy()

    # plot results ----------------------------------------------------------------
    save_path = osp.join('./outputs/', args.taskname, 'fig-analyze.pdf')
    fig, ax = plt.subplots(8, 1, figsize=(6, 24), dpi=300)

    plot_trajectory(ax[0], true_q, pred_q, 'Trajectory')

    plot_energy(ax[1], t, true_eng, true_potential_eng, true_kinetic_eng, 'Ground Truth Energy')
    plot_energy(ax[2], t, pred_eng, pred_potential_eng, pred_kinetic_eng, 'Prediction Energy')
    plot_compare_energy(ax[3], t, true_eng, pred_eng, 'Compare Energy')

    plot_compare_state(ax[4], t, true_q, pred_q, 'State $q$')
    plot_compare_state(ax[5], t, true_p, pred_p, 'State $p$')

    plot_field(ax[6], t, true_q, true_p, 'True Field')
    plot_field(ax[7], t, pred_q, pred_p, 'Prediction Field')

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
