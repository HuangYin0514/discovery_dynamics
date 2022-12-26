import argparse
import copy
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import learner as ln

sys.path.append('.')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=None)
# For general settings
parser.add_argument('--seed', default=0, type=int, help='random seed')
# For task
parser.add_argument('--obj', default=3, type=int, help='number of elements')
parser.add_argument('--dim', default=2, type=int, help='degree of freedom')
parser.add_argument('--train_num', default=0, type=int, help='the number of sampling trajectories')
parser.add_argument('--test_num', default=2, type=int, help='the number of sampling trajectories')
# For net
parser.add_argument('--load_net_path', default='', type=str, help='The path to load the pretrained network')
# For other settings
parser.add_argument('--dtype', default='float', type=str, help='Types of data and models')
# For test
parser.add_argument('--t0', default=0, type=int, help='number of elements')
parser.add_argument('--t_end', default=5, type=int, help='number of elements')
parser.add_argument('--h', default=0.05, type=float, help='number of elements')

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
    save_path = path + '/analyze/analyze_body_3/'
    if not os.path.isdir(save_path): os.makedirs(save_path)

    # task variable
    truth_t = np.arange(args.t0, args.t_end, args.h)
    data = ln.data.BodyData(obj=args.obj, dim=args.dim,
                            train_num=args.train_num,
                            test_num=args.test_num,
                            m=[1 for i in range(args.obj)],
                            l=[0 for i in range(args.obj)])

    # ground truth
    solver = ln.integrator.rungekutta.RK45(data.hamilton_right_fn, t0=args.t0, t_end=args.t_end)
    # solver = ln.integrator.rungekutta.RK4(data.hamilton_right_fn, t0=args.t0, t_end=args.t_end)

    # net
    input_dim = args.obj * args.dim * 2
    hnn = ln.nn.HNN(dim=input_dim, layers=1, width=200)
    local = path + 'body_3_hnn/model-body_3_hnn.pkl'
    # local_url = 'https://drive.google.com/file/d/1bMzjQvPQZW2ByRQw0I0IQ67U2p8xzkh2/view?usp=share_link'
    # ln.utils.download_file_from_google_drive(local_url, local)
    ln.utils.load_network(hnn, local, device)
    hnn.device = device
    hnn.dtype = args.dtype

    # result dict
    method_solution = {
        'ground_truth': {
            'solver': solver,
            'trajectory': [],
            'energy': [],
            'marker': 'k-',
        },
        'hnn': {
            'solver': hnn,
            'trajectory': [],
            'energy': [],
            'marker': 'r-.',
        },
    }

    # # plot the solution
    draw_one_sample_error_curve(args, data, method_solution, truth_t, save_path)
    draw_more_sample_error_curve(args, data, method_solution, truth_t, save_path)


def calculate_sample_trajectory(args, dataclass, method_solution, test_num):
    print('Starting computer trajectory')

    pbar = tqdm(range(test_num), desc='Processing')
    for i in pbar:
        np.random.seed(i)
        y0 = dataclass.random_config(1).reshape(-1)

        for name, value in method_solution.items():
            t_current = time.time()

            if name == 'ground_truth':
                solver = value['solver']
                traj = solver.solve(y0, args.h)
                eng = np.asarray(list(map(lambda x: dataclass.hamilton_energy_fn(x), traj)))
                method_solution['ground_truth']['trajectory'].append(traj)
                method_solution['ground_truth']['energy'].append(eng)
            else:
                solver = method_solution[name]['solver']
                traj = solver.predict(y0, args.h, args.t0, args.t_end, solver_method='RK45', circular_motion=False)
                eng = np.asarray(list(map(lambda x: dataclass.hamilton_energy_fn(x), traj)))
                method_solution[name]['trajectory'].append(traj)
                method_solution[name]['energy'].append(eng)

            spend_time = '{:.3}s'.format(time.time() - t_current)

            postfix = {
                'model': name,
                'spend_time': spend_time,
            }
            pbar.set_postfix(postfix)


def draw_one_sample_error_curve(args, dataclass, method_solution, truth_t, save_path):
    method_solution = copy.deepcopy(method_solution)
    calculate_sample_trajectory(args, dataclass, method_solution, test_num=1)

    # plot trajectories
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), dpi=300)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.01, hspace=0)

    plot_one_sample_trajectory(ax[0, 0], 'ground_truth', method_solution)  # ax[0, 1]
    plot_one_sample_trajectory(ax[0, 1], 'hnn', method_solution)

    plot_one_sample_energy(ax[1, 0], dataclass, 'ground_truth', method_solution, truth_t)
    plot_one_sample_energy(ax[1, 1], dataclass, 'hnn', method_solution, truth_t)

    plt.tight_layout()
    fig.savefig(save_path + '/fig-trajectories.pdf', bbox_inches='tight')
    plt.show()


def plot_one_sample_trajectory(ax, method_name, method_solution):
    y = method_solution[method_name]['trajectory'][0]
    marker = method_solution[method_name]['marker']

    ax.set_title('{} Trajectories'.format(method_name))
    for i in range(args.obj):
        ax.text(y[0, i * 2], y[0, i * 2 + 1], '{}'.format(i))
        ax.plot(y[:, 2 * i], y[:, 2 * i + 1], marker, label='body {} path'.format(i), linewidth=2)
    ax.axis('equal')
    ax.set_xlabel('$x\;(m)$')
    ax.set_ylabel('$y\;(m)$')


def plot_one_sample_energy(ax, dataclass, method_name, method_solution, dynamics_t):
    coords = method_solution[method_name]['trajectory'][0]
    marker = method_solution[method_name]['marker']

    T = np.array([dataclass.hamiltonian_kinetic(y) for y in coords])
    U = np.array([dataclass.hamiltonian_potential(y) for y in coords])

    ax.set_xlabel('Time step $(s)$')
    ax.set_ylabel('$E\;(J)$')
    ax.plot(dynamics_t, U, 'y:', label='potential', linewidth=2)
    ax.plot(dynamics_t, T, 'c-.', label='kinetic', linewidth=2)
    ax.plot(dynamics_t, U + T, 'g--', label='total', linewidth=2)
    ax.legend(fontsize=12)


def draw_more_sample_error_curve(args, dataclass, method_solution, truth_t, save_path):
    method_solution = copy.deepcopy(method_solution)

    # Compute the trajectory for each model
    calculate_sample_trajectory(args, dataclass, method_solution, test_num=args.test_num)

    # Calculate the error between each model's trajectory and the true trajectory
    calculate_more_sample_trajectory_error(args, method_solution)

    # plot trajectories
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), dpi=300)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0.01, hspace=0)

    # plot each trajectory and energy error
    plot_more_sample_trajectory_error(ax[0], args, method_solution, truth_t)
    plot_more_sample_energy_error(ax[1], args, method_solution, truth_t)

    fig.set_tight_layout(True)
    fig.savefig(save_path + '/fig-bar_error.pdf', bbox_inches='tight')
    plt.show()


def calculate_more_sample_trajectory_error(args, method_solution):
    dof = args.obj * args.dim

    for name, value in method_solution.items():
        if name == 'ground_truth': continue

        pos_error_list = []
        eng_error_list = []
        length = len(method_solution['ground_truth']['energy'][0])  # 时间步长度 (t_end - t0)/h

        for i in range(args.test_num):
            truth_traj = method_solution['ground_truth']['trajectory'][i][:, :dof]
            net_traj = method_solution[name]['trajectory'][i][:, :dof]
            pos_error = np.linalg.norm(truth_traj - net_traj) / length
            pos_error_list.append(pos_error)

            truth_energy = method_solution['ground_truth']['energy'][i]
            net_energy = method_solution[name]['energy'][i]
            eng_error = np.linalg.norm(truth_energy - net_energy) / length
            eng_error_list.append(eng_error)

        content = ('\n'
                   + 'Model name: {}'.format(name)
                   + '\n'
                   + 'Position error: {:.3e} +/- {:.3e} '.format(np.mean(pos_error_list), np.std(pos_error_list))
                   + '\n'
                   + 'Energy error: {:.3e} +/- {:.3e} '.format(np.mean(eng_error_list), np.std(eng_error_list)))
        print(content)


def plot_more_sample_trajectory_error(ax, args, method_solution, t):
    dof = args.obj * args.dim

    for name, value in method_solution.items():
        if name == 'ground_truth': continue
        truth_traj = np.asarray(list(map(lambda x: x[:, :dof], method_solution['ground_truth']['trajectory'])))
        net_traj = np.asarray(list(map(lambda x: x[:, :dof], method_solution[name]['trajectory'])))
        pos_error = np.linalg.norm(truth_traj - net_traj, axis=2)
        meanst = np.mean(pos_error, axis=0)
        sdt = np.std(pos_error, axis=0)
        ax.plot(t, meanst, method_solution[name]['marker'], label=name)
        ax.fill_between(t, meanst, meanst + sdt, alpha=0.3, )

    ax.set_yscale('log')
    ax.legend(fontsize=LEGENDSIZE)

    ax.grid('on')
    ax.set_ylabel('MSE of position ($m$)')
    ax.set_xlabel('Time ($s$)')


def plot_more_sample_energy_error(ax, args, method_solution, t):
    for name, value in method_solution.items():
        if name == 'ground_truth': continue
        truth_e = np.asarray(method_solution['ground_truth']['energy'])
        net_e = np.asarray(method_solution[name]['energy'])
        # energy_error = np.linalg.norm(truth_e - net_e, axis=2)
        energy_error = np.abs(truth_e - net_e)
        meanst = np.mean(energy_error, axis=0)
        sdt = np.std(energy_error, axis=0)
        ax.plot(t, meanst, method_solution[name]['marker'], label=name, )
        ax.fill_between(t, meanst, meanst + sdt, alpha=0.3, )
    ax.set_yscale('log')
    ax.legend(fontsize=LEGENDSIZE)

    ax.grid('on')
    ax.set_ylabel('MSE of energy ($J$)')
    ax.set_xlabel('Time ($s$)')


if __name__ == '__main__':
    main()
    print('Done!')
