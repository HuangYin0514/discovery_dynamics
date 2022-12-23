import argparse
import os
import sys

import numpy as np
import torch

import learner as ln
from data_body_3 import BodyData

sys.path.append('.')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=None)
# For general settings
parser.add_argument('--taskname', default='body_3_hnn', type=str, help='Task name')
parser.add_argument('--net', default='hnn', type=str, help='Select model to train')
parser.add_argument('--seed', default=0, type=int, help='random seed')
# For task
parser.add_argument('--tasktype', default='pend', type=str, help='Task type')
parser.add_argument('--obj', default=3, type=int, help='number of elements')
parser.add_argument('--dim', default=2, type=int, help='degree of freedom')
parser.add_argument('--train_num', default=3, type=int, help='the number of sampling trajectories')
parser.add_argument('--test_num', default=2, type=int, help='the number of sampling trajectories')
parser.add_argument('--dataset_url', default='', type=str, help='Download dataset from Internet')
# For net
parser.add_argument('--load_net_path', default='', type=str, help='The path to load the pretrained network')
# For training settings
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--criterion', default='L2_norm_loss', type=str, help='Select criterion to learn')
parser.add_argument('--optimizer', default='adam', type=str, help='Select optimizer to learn')
parser.add_argument('--scheduler', default='MultiStepLR', type=str, help='Select scheduler to learn')
parser.add_argument('--iterations', default=20, type=int, help='end of training epoch')
parser.add_argument('--print_every', default=2, type=int, help='number of gradient steps between prints')
# For other settings
parser.add_argument('--dtype', default='float', type=str, help='Types of data and models')
parser.set_defaults(feature=True)
args = parser.parse_args()


def run():
    print('Task name: {}'.format(args.taskname))

    # seed
    ln.utils.init_random_state(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the device is:', device)

    # data
    data = BodyData(obj=args.obj, dim=args.dim,
                    train_num=args.train_num,
                    test_num=args.test_num,
                    m=[1 for i in range(args.obj)],
                    l=[1 for i in range(args.obj)])
    data = ln.get_dataset(args, data)

    # net
    net = ln.get_model(args, device=device)

    arguments = {
        'taskname': args.taskname,
        'data': data,
        'criterion': args.criterion,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'lr': args.lr,
        'iterations': args.iterations,
        'batch_size': None,  # NoImplementation
        'print_every': args.print_every,
        'save': True,
        'dtype': args.dtype,
        'device': device,
        'net': net
    }

    ln.Brain.Init(**arguments)
    ln.Brain.Run()
    ln.Brain.Restore()
    ln.Brain.Output(info=arguments)
    print(arguments)


def main():
    run()


if __name__ == '__main__':
    main()
