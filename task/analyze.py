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

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('.')
sys.path.append(PARENT_DIR)

import learner as ln

parser = argparse.ArgumentParser(description=None)
# For general settings
parser.add_argument('--taskname', default='pend_2_task', type=str, help='Task name')
parser.add_argument('--seed', default=0, type=int, help='random seed')

# For task
parser.add_argument('--obj', default=2, type=int, help='number of objects')
parser.add_argument('--dim', default=2, type=int, help='coordinate dimension')

# data
parser.add_argument('--data_name', default='Pendulum2_L_dae', type=str, help='choose dataset')
parser.add_argument('--train_num', default=0, type=int, help='the number of train sampling trajectories')
parser.add_argument('--test_num', default=0, type=int, help='the number of test sampling trajectories')
parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. ')

# net
parser.add_argument('--net_name', default='Analytical_pend2_dae', type=str, help='Select model to train')
parser.add_argument('--net_url', default='', type=str, help='Download net from Internet')

# For other settings
parser.add_argument('--dtype', default='float', type=str, help='Types of data and models')

parser.set_defaults(feature=True)
args = parser.parse_args()


def main():
    print('=' * 500)
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
    net = ln.nn.get_model(**arguments).to(device)

    # data ----------------------------------------------------------------
    arguments = {
        'taskname': args.taskname,
        'data_name': args.data_name,
        'dataset_path': './outputs/data/dataset_' + args.data_name,
        'num_workers': args.num_workers,
        'dtype': args.dtype,
        'device': device
    }
    data = ln.data.get_dataloader(**arguments)

    arguments = {
        'taskname': args.taskname,
        'data': data,
        'dtype': args.dtype,
        'device': device,
        'net': net,
        'obj': args.obj,
        'dim': args.dim
    }
    ln.AnalyzeBrain.Init(**arguments)
    ln.AnalyzeBrain.Run()


if __name__ == '__main__':
    main()
