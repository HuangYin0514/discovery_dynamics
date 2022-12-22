import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import learner as ln
from task.experiment_pend_2.data_pend_2 import PendulumData

sys.path.append('.')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=None)
# For general settings
parser.add_argument('--seed', default=0, type=int, help='random seed')
# For task
parser.add_argument('--obj', default=2, type=int, help='number of elements')
parser.add_argument('--dim', default=1, type=int, help='degree of freedom')
parser.add_argument('--train_num', default=1, type=int, help='the number of sampling trajectories')
parser.add_argument('--test_num', default=1, type=int, help='the number of sampling trajectories')
# For net
parser.add_argument('--load_net_path', default='', type=str, help='The path to load the pretrained network')
# For other settings
parser.add_argument('--dtype', default='float', type=str, help='Types of data and models')
parser.set_defaults(feature=True)
args = parser.parse_args()

LINE_WIDTH = 2
LEGENDSIZE = 12


def main():
    # init state for system
    ln.utils.init_random_state(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the device is:', device)
    path = './outputs/'
    save_path = path + '/analyze/analyze-pend-2/'
    if not os.path.isdir(save_path): os.makedirs(save_path)

    # task variable
    y0 = np.array([1., 2., 0., 0.]).reshape(-1)
    t0, t_end, h = 0, 10, 0.02

    # ground truth
    pendulumData = PendulumData(obj=args.obj, dim=args.dim,
                                train_num=args.train_num,
                                test_num=args.test_num,
                                m=[1 for i in range(args.obj)],
                                l=[1 for i in range(args.obj)])
    # solver = ln.integrator.rungekutta.RK4(pendulumData.hamilton_right_fn, t0=t0, t_end=t_end)
    truth_t = np.arange(t0, t_end, h)
    solver = ln.integrator.rungekutta.RK45(pendulumData.hamilton_right_fn, t0=t0, t_end=t_end)
    traj_truth = solver.solve(y0, h)

    # net
    input_dim = args.obj * args.dim * 2
    hnn = ln.nn.HNN(dim=input_dim, layers=1, width=200)
    local = path + 'pend_2_hnn/model-pend_2_hnn.pkl'
    ln.utils.load_network(hnn, local, device)
    hnn.device = device
    hnn.dtype = args.dtype

    hnn_traj = hnn.predict(y0, h, t0, t_end, solver_method='RK45', circular_motion=True)

    method_solution = {
        'ground_truth': {
            'trajectory': traj_truth,
            'marker': 'k-',
        },
        'baseline': {
            'trajectory': hnn_traj,
            'marker': 'r--',
        },
        'hnn': {
            'trajectory': hnn_traj,
            'marker': 'b-.',
        },
        'hnn2': {
            'trajectory': hnn_traj,
            'marker': 'b-.',
        },
    }

    # plot trajectories
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), dpi=300)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.01, hspace=0)

    plot_trajectory(ax[0, 0], 'ground_truth', method_solution)  # ax[0, 1]
    plot_trajectory(ax[0, 1], 'hnn', method_solution)  # ax[0, 1]

    plot_coordinates_error(ax[1, 0], truth_t, method_solution)
    plot_energy_error(ax[1, 1], pendulumData, truth_t, method_solution)

    fig.set_tight_layout(True)
    fig.savefig(save_path + '/fig-trajectories.pdf', bbox_inches='tight')
    plt.show()

def plot_trajectory(ax, method_name, method_solution):
    truth_pos = ln.utils.polar2xy(method_solution['ground_truth']['trajectory'])

    if method_name == 'ground_truth':
        legend_name = '_nolegend_'
        net_pos = truth_pos
    else:
        legend_name = method_name
        net_pos = ln.utils.polar2xy(method_solution[method_name]['trajectory'])

    ln.utils.plot_pend_trajectory(ax, truth_pos, net_pos,
                                  legend_name,
                                  marker=method_solution[method_name]['marker'])


def plot_coordinates_error(ax, t, method_solution):
    ax.set_title("MSE between coordinates")
    ax.set_xlabel('Time step ($s$)')
    ax.set_ylabel('$x\;(m)$')
    for name, value in method_solution.items():
        if name == 'ground_truth': continue
        truth_pos = ln.utils.polar2xy(method_solution['ground_truth']['trajectory'])
        net_pos = ln.utils.polar2xy(method_solution[name]['trajectory'])
        error = ((truth_pos - net_pos) ** 2).mean(-1)
        ax.semilogy(t, error, value['marker'], label=name, linewidth=LINE_WIDTH)
    ax.legend(fontsize=LEGENDSIZE)


def plot_energy_error(ax, data, t, method_solution):
    ax.set_title("MSE of total energy")
    ax.set_xlabel('Time step ($s$)')
    ax.set_ylabel('$E\;(J)$')

    true_e = np.stack([data.hamilton_energy_fn(c) for c in method_solution['ground_truth']['trajectory']])

    for name, value in method_solution.items():
        if name == 'ground_truth': continue
        net_e = np.stack([data.hamilton_energy_fn(c) for c in value['trajectory']])
        error = (true_e - net_e) ** 2
        ax.semilogy(t, error, value['marker'], label=name, linewidth=LINE_WIDTH)

    ax.legend(fontsize=LEGENDSIZE)


if __name__ == '__main__':
    main()
    print('done!')
