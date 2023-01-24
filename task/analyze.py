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
parser.add_argument('--dim', default=1, type=int, help='coordinate dimension')

# data
parser.add_argument('--data_name', default='Pendulum2_L', type=str, help='choose dataset')
parser.add_argument('--train_num', default=0, type=int, help='the number of train sampling trajectories')
parser.add_argument('--test_num', default=0, type=int, help='the number of test sampling trajectories')
parser.add_argument('--download_data', default='False', type=str, help='Download dataset from Internet')
parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. ')

# net
parser.add_argument('--net_name', default='ModLaNet_pend2', type=str, help='Select model to train')
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
        'obj': args.obj,
        'dim': args.dim,
        'train_num': args.train_num,
        'test_num': args.test_num,
        'download_data': args.download_data,
        'num_workers': args.num_workers,
    }
    data = ln.data.get_dataloader(**arguments)

    arguments = {
        'taskname': args.taskname,
        'data': data,
        'dtype': args.dtype,
        'device': device,
        'net': net
    }
    ln.AnalyzeBrain.Init(**arguments)
    ln.AnalyzeBrain.Run()


if __name__ == '__main__':
    main()
