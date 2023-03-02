import argparse
import os
import sys

import torch

from gendata.generate_brain import gen_dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../task')
sys.path.append(PARENT_DIR)

import learner as ln


parser = argparse.ArgumentParser(description=None)
# For general settings
parser.add_argument('--taskname', default='gen_data', type=str, help='Task name')
parser.add_argument('--seed', default=0, type=int, help='random seed')

# For task
parser.add_argument('--obj', default=2, type=int, help='number of objects')
parser.add_argument('--dim', default=1, type=int, help='coordinate dimension')

# data
parser.add_argument('--data_name', default='Pendulum2', type=str,
                    help='choose dataset '
                         '[Pendulum2 Pendulum2_L '
                         'Body3 Body3_L '
                         ']'
                    )
parser.add_argument('--train_num', default=3, type=int, help='the number of train sampling trajectories')
parser.add_argument('--val_num', default=1, type=int, help='the number of val sampling trajectories')
parser.add_argument('--test_num', default=2, type=int, help='the number of test sampling trajectories')
parser.add_argument('--download_data', default='True', type=str, help='Download dataset from Internet')
parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. ')

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
    arguments = {
        'taskname': args.taskname,
        'data_name': args.data_name,
        'obj': args.obj,
        'dim': args.dim,
        'train_num': args.train_num,
        'val_num': args.val_num,
        'test_num': args.test_num,
        'download_data': args.download_data,
        'num_workers': args.num_workers,
        'dtype': args.dtype,
        'device': device
    }
    gen_dataset(**arguments)

def main():
    run()


if __name__ == '__main__':
    main()
