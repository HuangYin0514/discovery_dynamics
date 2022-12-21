import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch

EXPERIMENT_DIR = './experiment_pend_2'
sys.path.append('.')

from task.experiment_pend_2.data_pend_2 import MyDataset
from src.models import Baseline, HNN, LNN
from src.utils import get_device

plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"]  = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

wordsize = 18
plt.rc('font', size=wordsize)  # controls default text sizes
plt.rc('axes', titlesize=wordsize)  # fontsize of the axes title
plt.rc('axes', labelsize=wordsize)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=wordsize)  # fontsize of the tick labels
plt.rc('ytick', labelsize=wordsize)  # fontsize of the tick labels
plt.rc('legend', fontsize=wordsize)  # legend fontsize
plt.rc('figure', titlesize=wordsize)  # fontsize of the figure title

DPI = 300
FORMAT = 'png'
LINE_SEGMENTS = 10
ARROW_SCALE = 30  # 100 for pend-sim, 30 for pend-real
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

TPAD = 7
LEGENDSIZE = 12

device = get_device()


def get_args():
    parser = argparse.ArgumentParser(description=None)
    # MODEL SETTINGS
    parser.add_argument('--name', default='pend', type=str, help='mode name')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--obj', default=2, type=int, help='number of elements')
    parser.add_argument('--dof', default=1, type=int, help='degree of freedom')
    parser.add_argument('--t0', default=0, type=int, help='start of time')
    parser.add_argument('--t_end', default=30, type=int, help='end of time')
    parser.add_argument('--ode_stepsize', default=0.01, type=int, help='step size of ode solver')
    parser.add_argument('--noise', default=0., type=float, help='the noise amplitude of the data')
    parser.add_argument('--samples', default=2, type=int, help='the number of sampling trajectories')
    parser.add_argument('--re_test', nargs='?', const=True, default=False, help='try to use gpu?')
    parser.add_argument('--save_dir', default=EXPERIMENT_DIR + '/data', type=str, help='dir of read results')
    parser.add_argument('--result_dir', default='../results', type=str, help='dir of save results ')
    parser.set_defaults(feature=True)
    return parser.parse_known_args()


def get_model(args, model_name, end_epoch, noise, learn_rate, hidden_dim=None, load_obj=None):
    # model_name, length, end_epoch, noise, learn_rate
    load_obj = args.obj if load_obj is None else load_obj
    path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, load_obj,
                                                                                            args.name,
                                                                                            model_name, hidden_dim,
                                                                                            end_epoch,
                                                                                            noise, learn_rate)
    checkpoint = torch.load(path, map_location=device)

    assert load_obj == args.obj
    input_dim = args.obj * args.dof * 2
    if model_name == 'baseline':
        model = Baseline(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=input_dim).to(device)
    elif model_name == 'hnn':
        model = HNN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    elif model_name == 'lnn':
        model = LNN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint['network_state_dict'])
    return model


def runge_kutta_solver(rightf, t0, t_end, t_step_size, x0, solve_method='solve_ivp'):
    """ 龙格库塔方法
    :param rightf: 方程右端项
    :param t0: 初始时间
    :param t_end: 末端时间
    :param t_step_size: 时间步长
    :param x0: 初始条件 shape->(N,)
    :return: t->时间，x->方程解
    """
    t, x = None, None
    if solve_method == 'solve_ivp':
        t = np.arange(t0, t_end, t_step_size)
        solve_result = scipy.integrate.solve_ivp(fun=rightf, t_span=[t0, t_end], y0=x0, t_eval=t, rtol=1e-12)
        x = solve_result['y'].T
    elif solve_method == 'rk4':
        num = len(x0)
        t = np.arange(t0, t_end, t_step_size)
        x = np.zeros((t.size, num))
        x[0, :] = x0
        for i in range(t.size - 1):
            s0 = rightf(t[i], x[i, :])
            s1 = rightf(t[i] + t_step_size / 2., x[i, :] + t_step_size * s0 / 2.)
            s2 = rightf(t[i] + t_step_size / 2., x[i, :] + t_step_size * s1 / 2.)
            s3 = rightf(t[i + 1], x[i, :] + t_step_size * s2)
            x[i + 1, :] = x[i, :] + t_step_size * (s0 + 2 * (s1 + s2) + s3) / 6.
    return t, x


def integrate_model(args, model, t0, t_end, ode_stepsize, y0, model_name, **kwargs):
    if model_name in ['hnn', 'baseline']:
        def fun(t, np_x):
            np_x[:args.obj * args.dof] = np_x[:args.obj * args.dof] % (2 * np.pi)
            x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32, device=device).view(1, args.obj * args.dof * 2)
            dx = model(x).cpu().data.numpy().reshape(-1)
            return dx
    elif model_name == 'lnn':
        def fun(t, np_x):
            np_x[:args.obj * args.dof] = np_x[:args.obj * args.dof] % (2 * np.pi)
            x = torch.tensor(np_x[:args.obj * args.dof], requires_grad=True, dtype=torch.float32).view(1, args.obj * args.dof)
            v = torch.tensor(np_x[args.obj * args.dof:], requires_grad=True, dtype=torch.float32).view(1, args.obj * args.dof)
            inputs = torch.cat([x, v], dim=1).to(device)
            dx = model(inputs).cpu().data.numpy().reshape(-1)
            return np.concatenate((np_x[args.obj * args.dof:], dx))
    return runge_kutta_solver(fun, t0, t_end, ode_stepsize, y0, solve_method='solve_ivp')



def polar2xy(x):
    pos = np.zeros([x.shape[0], x.shape[1] * 2])
    for i in range(x.shape[1]):
        if i == 0:
            pos[:, 2 * i:2 * (i + 1)] += np.concatenate([np.sin(x[:, i:i + 1]), -np.cos(x[:, i:i + 1])], 1)
        else:
            pos[:, 2 * i:2 * (i + 1)] += pos[:, 2 * (i - 1):2 * i] + np.concatenate(
                [np.sin(x[:, i:i + 1]), -np.cos(x[:, i:i + 1])], 1)
    return pos


def visualize_one_state(pos_truth, pos_baseline, pos_hnn):
    fig = plt.figure(figsize=(16, 4), dpi=DPI)

    t_max = min(400, len(dynamics_solution))

    # Ground truth
    plt.subplot(1, 3, 1)
    plt.xlabel('$x$ ($m$)')
    plt.ylabel('$y$ ($m$)')
    for i in range(t_max - 3):
        plt.plot(pos_truth[i:i + 2, 2], pos_truth[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=LINE_WIDTH, alpha=0.2 + 0.8 * (i + 1) / t_max)  # 轨迹
        if i % 200 == 0:  # 中间状态
            plt.plot([0, pos_truth[i, 0]], [0, pos_truth[i, 1]], color='brown', linewidth=LINE_WIDTH, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / t_max)
            plt.plot([pos_truth[i, 0], pos_truth[i, 2]], [pos_truth[i, 1], pos_truth[i, 3]], 'o-', color='brown', linewidth=LINE_WIDTH, label='_nolegend_',
                     alpha=0.2 + 0.8 * (i + 1) / t_max)
            plt.scatter(pos_truth[i, 0], pos_truth[i, 1], s=50, linewidths=LINE_WIDTH, facecolors='gray', edgecolors='brown', label='_nolegend_',
                        alpha=min(0.5 + 0.8 * (i + 1) / t_max, 1), zorder=3)
            plt.scatter(pos_truth[i, 2], pos_truth[i, 3], s=50, linewidths=LINE_WIDTH, facecolors='gray',
                        edgecolors='brown', label='_nolegend_',
                        alpha=min(0.5 + 0.8 * (i + 1) / t_max, 1), zorder=3)
    plt.plot(pos_truth[t_max - 2:t_max, 2], pos_truth[t_max - 2:t_max, 3], 'k-', label='Ground truth', linewidth=LINE_WIDTH, alpha=1)  # 轨迹
    plt.plot([0, pos_truth[t_max, 0]], [0, pos_truth[t_max, 1]], color='brown', linewidth=LINE_WIDTH, label='_nolegend_')
    plt.plot([pos_truth[t_max, 0], pos_truth[t_max, 2]], [pos_truth[t_max, 1], pos_truth[t_max, 3]], 'o-', color='brown', linewidth=LINE_WIDTH, label='Pendulum')  # 摆
    plt.scatter(pos_truth[t_max, 0], pos_truth[t_max, 1], s=50, linewidths=LINE_WIDTH, facecolors='gray', edgecolors='brown', label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(pos_truth[t_max, 2], pos_truth[t_max, 3], s=50, linewidths=LINE_WIDTH, facecolors='gray', edgecolors='brown', label='_nolegend_', alpha=1, zorder=3)  # 球
    plt.legend(fontsize=LEGENDSIZE)

    # baseline
    plt.subplot(1, 3, 2)
    plt.xlabel('$x$ ($m$)')
    plt.ylabel('$y$ ($m$)')
    for i in range(t_max - 3):
        plt.plot(pos_truth[i:i + 2, 2], pos_truth[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=LINE_WIDTH, alpha=0.2 + 0.8 * (i + 1) / t_max)
        plt.plot(pos_baseline[i:i + 2, 2], pos_baseline[i:i + 2, 3], 'r-.', label='_nolegend_', linewidth=LINE_WIDTH, alpha=0.2 + 0.8 * (i + 1) / t_max)  # 轨迹
        if i % 200 == 0:
            plt.plot([0, pos_baseline[i, 0]], [0, pos_baseline[i, 1]], color='brown', linewidth=LINE_WIDTH, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / t_max)
            plt.plot([pos_baseline[i, 0], pos_baseline[i, 2]], [pos_baseline[i, 1], pos_baseline[i, 3]], 'o-', color='brown', linewidth=2, label='_nolegend_',
                     alpha=0.2 + 0.8 * (i + 1) / t_max)
            plt.scatter(pos_baseline[i, 0], pos_baseline[i, 1], s=50, linewidths=2, facecolors='gray',
                        edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / t_max, 1),
                        zorder=3)
            plt.scatter(pos_baseline[i, 2], pos_baseline[i, 3], s=50, linewidths=2, facecolors='gray',
                        edgecolors='brown', label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / t_max, 1),
                        zorder=3)
    plt.plot(pos_truth[t_max - 2:t_max, 2], pos_truth[t_max - 2:t_max, 3], 'k-', label='Ground truth', linewidth=LINE_WIDTH, alpha=1)  # Ground truth 轨迹
    plt.plot(pos_baseline[t_max - 2:t_max, 2], pos_baseline[t_max - 2:t_max, 3], 'r-.', label='baseline', linewidth=LINE_WIDTH, alpha=1)  # hnn 轨迹
    plt.plot([0, pos_baseline[t_max, 0]], [0, pos_baseline[t_max, 1]], color='brown', linewidth=2, label='_nolegend_')
    plt.plot([pos_baseline[t_max, 0], pos_baseline[t_max, 2]], [pos_baseline[t_max, 1], pos_baseline[t_max, 3]], 'o-', color='brown', linewidth=2, label='Pendulum')  # 摆
    plt.scatter(pos_baseline[t_max, 0], pos_baseline[t_max, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown', label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(pos_baseline[t_max, 2], pos_baseline[t_max, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown', label='_nolegend_', alpha=1, zorder=3)  # 摆末端的点
    plt.legend(fontsize=LEGENDSIZE)

    # hnn
    plt.subplot(1, 3, 3)
    plt.xlabel('$x$ ($m$)')
    plt.ylabel('$y$ ($m$)')
    for i in range(t_max - 3):
        plt.plot(pos_truth[i:i + 2, 2], pos_truth[i:i + 2, 3], 'k-', label='_nolegend_', linewidth=LINE_WIDTH, alpha=0.2 + 0.8 * (i + 1) / t_max)
        plt.plot(pos_hnn[i:i + 2, 2], pos_hnn[i:i + 2, 3], 'g-.', label='_nolegend_', linewidth=LINE_WIDTH, alpha=0.2 + 0.8 * (i + 1) / t_max)  # 轨迹
        if i % 200 == 0:
            plt.plot([0, pos_hnn[i, 0]], [0, pos_hnn[i, 1]], color='brown', linewidth=LINE_WIDTH, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / t_max)
            plt.plot([pos_hnn[i, 0], pos_hnn[i, 2]], [pos_hnn[i, 1], pos_hnn[i, 3]], 'o-', color='brown', linewidth=2, label='_nolegend_', alpha=0.2 + 0.8 * (i + 1) / t_max)
            plt.scatter(pos_hnn[i, 0], pos_hnn[i, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / t_max, 1),
                        zorder=3)
            plt.scatter(pos_hnn[i, 2], pos_hnn[i, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown',
                        label='_nolegend_', alpha=min(0.5 + 0.8 * (i + 1) / t_max, 1),
                        zorder=3)
    plt.plot(pos_truth[t_max - 2:t_max, 2], pos_truth[t_max - 2:t_max, 3], 'k-', label='Ground truth', linewidth=LINE_WIDTH, alpha=1)  # Ground truth 轨迹
    plt.plot(pos_hnn[t_max - 2:t_max, 2], pos_hnn[t_max - 2:t_max, 3], 'g-.', label='HNN', linewidth=LINE_WIDTH, alpha=1)  # hnn 轨迹
    plt.plot([0, pos_hnn[t_max, 0]], [0, pos_hnn[t_max, 1]], color='brown', linewidth=2, label='_nolegend_')
    plt.plot([pos_hnn[t_max, 0], pos_hnn[t_max, 2]], [pos_hnn[t_max, 1], pos_hnn[t_max, 3]], 'o-', color='brown', linewidth=2, label='Pendulum')  # 摆
    plt.scatter(pos_hnn[t_max, 0], pos_hnn[t_max, 1], s=50, linewidths=2, facecolors='gray', edgecolors='brown', label='_nolegend_', alpha=1, zorder=3)
    plt.scatter(pos_hnn[t_max, 2], pos_hnn[t_max, 3], s=50, linewidths=2, facecolors='gray', edgecolors='brown', label='_nolegend_', alpha=1, zorder=3)  # 摆末端的点
    plt.legend(fontsize=LEGENDSIZE)

    plt.tight_layout()
    plt.show()
    fig.savefig('{}/pend-2-integration.{}'.format(args.result_dir, FORMAT), bbox_inches="tight")


def visualize_one_error(pos_truth, pos_baseline, pos_hnn):
    fig = plt.figure(figsize=(16, 4), dpi=DPI)

    t_max = min(300, len(dynamics_solution))

    plt.subplot(1, 2, 1)
    plt.title("MSE between coordinates");
    plt.xlabel('Time step ($s$)');
    plt.ylabel('$x\;(m)$')
    plt.semilogy(dynamics_t, ((pos_truth - pos_baseline) ** 2).mean(-1), 'r--', label='baseline', linewidth=LINE_WIDTH)
    plt.semilogy(dynamics_t, ((pos_truth - pos_hnn) ** 2).mean(-1), 'b-.', label='HNN', linewidth=LINE_WIDTH)
    plt.legend(fontsize=LEGENDSIZE)

    plt.subplot(1, 2, 2)
    plt.title("MSE of total energy")
    plt.xlabel('Time step ($s$)')
    plt.ylabel('$E\;(J)$')
    true_e = np.stack([ds.hamilton_energy_fn(c) for c in dynamics_solution])
    true_baseline = np.stack([ds.hamilton_energy_fn(c) for c in dynamics_solution_baseline])
    hnn_e = np.stack([ds.hamilton_energy_fn(c) for c in dynamics_solution_hnn])
    # lnn_e = np.stack([ds.lagrange_energy_fn(c) for c in dynamics_solution_lnn])
    plt.semilogy(dynamics_t, (true_e - true_baseline) ** 2, 'r-.', label='baseline', linewidth=LINE_WIDTH)
    plt.semilogy(dynamics_t, (true_e - hnn_e) ** 2, 'b-.', label='HNN', linewidth=LINE_WIDTH)
    # plt.semilogy(dynamics_t, (true_e - lnn_e) ** 2, 'r--', label='LNN', linewidth=LINE_WIDTH)
    plt.legend(fontsize=LEGENDSIZE)

    plt.tight_layout()
    plt.show()
    fig.savefig('{}/pend-2-integration-mse.{}'.format(args.result_dir, FORMAT), bbox_inches="tight")


def pred_more_sampe():
    # Be careful! It will take a while.
    # Uncomment the code below for checking.
    print('Be careful to run it! It will take a while.')

    x_list, e_list = [[] for i in range(num_models)], [[] for i in range(num_models)]
    for i in range(args.samples):
        print(i)
        np.random.seed(i)
        ds = MyDataset(args.obj, m=[1. for i in range(args.obj)], l=[1. for i in range(args.obj)])
        y0 = ds.random_config().reshape(-1)

        # ground truth
        t_current = time.time()
        x, dydt, dynamics_t, E = ds.get_trajectory(t0=t0, t_end=t_end, ode_stepsize=ode_stepsize, y0=y0, system="hnn")
        x = x[:, :args.obj]
        j = 0
        x_list[j].append(x)
        e_list[j].append(E)
        print('.1:{}'.format(time.time() - t_current))

        # baseline
        t_current = time.time()
        dynamics_t, dynamics_solution_baseline = integrate_model(args, baseline_model, t0, t_end, ode_stepsize, y0, model_name='hnn')
        x = dynamics_solution_baseline[:, :args.obj]
        E = np.array([ds.hamilton_energy_fn(y) for y in dynamics_solution_baseline])
        j = 1
        x_list[j].append(x)
        e_list[j].append(E)
        print('.2:{}'.format(time.time() - t_current))

        # hnn
        t_current = time.time()
        dynamics_t, dynamics_solution_hnn = integrate_model(args, hnn_model, t0, t_end, ode_stepsize, y0, model_name='hnn')
        x = dynamics_solution_hnn[:, :args.obj]
        E = np.array([ds.hamilton_energy_fn(y) for y in dynamics_solution_hnn])
        j = 2
        x_list[j].append(x)
        e_list[j].append(E)
        print('.3:{}'.format(time.time() - t_current))

    np.save('{}/analysis-pend-2.npy'.format(args.result_dir),
            {'x_list': np.array(x_list), 'e_list': np.asarray(e_list)})


def test_model_error():
    baseline_error = [[] for _ in range(2)]
    hnn_error = [[] for _ in range(2)]

    j = 0
    length = len(e_list[0][0])
    for i in range(args.samples):
        baseline_error[j].append(np.linalg.norm(x_list[0][i] - x_list[1][i]) / length)
        hnn_error[j].append(np.linalg.norm(x_list[0][i] - x_list[2][i]) / length)

    print('Mean error and std of position:')
    print('baseline:  {:.5f} +/- {:.5f}'.format(np.mean(baseline_error[j]), np.std(baseline_error[j])))
    print('hnn:       {:.5f} +/- {:.5f}'.format(np.mean(hnn_error[j]), np.std(hnn_error[j])))
    print('')

    j = 1
    for i in range(args.samples):
        baseline_error[j].append(np.linalg.norm(e_list[0][i] - e_list[1][i]) / length)
        hnn_error[j].append(np.linalg.norm(e_list[0][i] - e_list[2][i]) / length)

    print('Mean error and std of total energy:')
    print('baseline:  {:.5f} +/- {:.5f}'.format(np.mean(baseline_error[j]), np.std(baseline_error[j])))
    print('hnn:       {:.5f} +/- {:.5f}'.format(np.mean(hnn_error[j]), np.std(hnn_error[j])))
    print('')


def visualize_positions_error():
    # Ref: https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans

    fig, ax = plt.subplots(figsize=[8, 2.5], dpi=DPI)
    labels = ['baseline', 'HNN']
    lines = ['solid', 'dashed', 'dashdot', 'dotted',
             (0, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (5, 10)),
             (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10))]  # 'loosely dotted', 'loosely dashed', 'loosely dashdotted', 'dashdotdotted' and 'loosely dashdotdotted'

    x_array = np.array(x_list, dtype=np.float32)
    # with sns.axes_style("darkgrid"):
    epochs = np.linspace(t0, t_end, int((t_end - t0) / ode_stepsize))
    for i in range(num_models - 1):
        x_error = x_array[i + 1] - x_array[0]
        x_error_norm = np.linalg.norm(x_error, axis=2)
        meanst = np.mean(x_error_norm, axis=0)
        sdt = np.std(x_error_norm, axis=0)
        ax.plot(epochs, meanst, label=labels[i], linestyle=lines[i])
        ax.fill_between(epochs, meanst, meanst + sdt, alpha=0.3, )

    ax.legend(fontsize=LEGENDSIZE)
    ax.set_yscale('log')
    ax.tick_params(axis="y", direction='in')
    ax.tick_params(axis="x", direction='in')
    # ax.set_ylim(top=3e3)
    ax.set_xlim([0, 32])
    ax.annotate('$t$', xy=(0.98, -0.025), ha='left', va='top', xycoords='axes fraction')
    ax.annotate('MSE', xy=(-0.07, 1.05), xytext=(-15, 2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points')

    # ax.grid('on')
    ax.set_ylabel('MSE of position ($m$)')
    ax.set_xlabel('Time ($s$)')
    plt.tight_layout()
    plt.show()
    fig.savefig('{}/pend-2-100traj-pos.png'.format(args.result_dir))


def visualize_energy_error():
    # Ref: https://stackoverflow.com/questions/43064524/plotting-shaded-uncertainty-region-in-line-plot-in-matplotlib-when-data-has-nans

    fig, ax = plt.subplots(figsize=[8, 2.5], dpi=DPI)
    labels = ['baseline', 'HNN']

    e_array = np.array(e_list, dtype=np.float32)
    epochs = np.linspace(t0, t_end, int((t_end - t0) / ode_stepsize))
    for i in range(num_models - 1):
        e_error = e_array[i + 1] - e_array[0]
        e_error = np.abs(e_error).squeeze(2)
        meanst = np.mean(e_error, axis=0)
        sdt = np.std(e_error, axis=0)
        ax.plot(epochs, meanst, label=labels[i])
        ax.fill_between(epochs, meanst, meanst + sdt, alpha=0.3)
    ax.legend(fontsize=LEGENDSIZE)
    ax.set_yscale('log')
    ax.set_xlim([-1, 32])
    ax.annotate('$t$', xy=(0.98, -0.025), ha='left', va='top', xycoords='axes fraction')
    ax.annotate('MSE', xy=(-0.07, 1.05), xytext=(-15, 2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points')

    # ax.grid('on')
    ax.set_ylabel('MSE of energy ($J$)')
    ax.set_xlabel('Time ($s$)')
    plt.tight_layout()
    plt.show()
    fig.savefig('{}/pend-2-100traj-eng.png'.format(args.result_dir))


if __name__ == '__main__':
    args = get_args()[0]

    print('using the device is:', device)
    os.makedirs(args.result_dir) if not os.path.exists(args.result_dir) else None
    t0, t_end, ode_stepsize = args.t0, args.t_end, args.ode_stepsize

    # one sample ----------------------------------
    num_models = 3
    baseline_model = get_model(args, model_name='baseline', hidden_dim=200, end_epoch=3000, noise=0., learn_rate=0.001)
    hnn_model = get_model(args, model_name='hnn', hidden_dim=200, end_epoch=10000, noise=0., learn_rate=0.01)

    ds = MyDataset(args.obj, m=[1. for i in range(args.obj)], l=[1. for i in range(args.obj)])
    y0 = np.array([1., 2., 0., 0.]).reshape(-1)

    # ground thruth
    dynamics_t, dynamics_solution = runge_kutta_solver(ds.hamilton_right_fn, t0, t_end, ode_stepsize, y0)
    pos_truth = polar2xy(dynamics_solution[:, 0:args.obj])

    # baseline
    dynamics_t, dynamics_solution_baseline = integrate_model(args, baseline_model, t0, t_end, ode_stepsize, y0, model_name='baseline')
    pos_baseline = polar2xy(dynamics_solution_baseline[:, 0:args.obj])

    # # hnn
    dynamics_t, dynamics_solution_hnn = integrate_model(args, hnn_model, t0, t_end, ode_stepsize, y0, model_name='hnn')
    pos_hnn = polar2xy(dynamics_solution_hnn[:, 0:args.obj])

    # test one sample ----------------------------------
    visualize_one_state(pos_truth, pos_baseline, pos_hnn)
    visualize_one_error(pos_truth, pos_baseline, pos_hnn)

    # test more ----------------------------------
    # naming example: results/analysis-pend-2.npy
    filename = args.result_dir + '/analysis-pend-2.npy'
    if os.path.exists(filename) and args.re_test == False:
        print('Start loading result.')
    else:
        print('Start test results.')
        pred_more_sampe()
    results = np.load('{}/analysis-pend-2.npy'.format(args.result_dir), allow_pickle=True).item()
    x_list, e_list = results['x_list'], results['e_list']

    test_model_error()
    visualize_positions_error()
    visualize_energy_error()

    print('done.')
