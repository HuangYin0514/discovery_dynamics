import argparse
import os
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
parser.add_argument('--data_name', default='Pendulum2', type=str,
                    help='choose dataset '
                         '[Pendulum2 Pendulum2_L '
                         'Body3 Body3_L '
                         ']'
                    )
parser.add_argument('--train_num', default=3, type=int, help='the number of train sampling trajectories')
parser.add_argument('--test_num', default=2, type=int, help='the number of test sampling trajectories')
parser.add_argument('--num_workers', default=0, type=int, help='how many subprocesses to use for data loading. ')

# net
parser.add_argument('--net_name', default='HnnModScale_pend2', type=str,
                    help='Select model to train '
                         '[Baseline_pend2 Baseline_body3'
                         'HNN_pend2 HNN_body3'
                         'LNN_pend2 LNN_body3'
                         'MechanicsNN_pend2 MechanicsNN_body3'
                         'ModLaNet_pend2 ModLaNet_body3'
                         'HnnMod_pend2 HnnMod_body3'
                         'HnnModScale_pend2 HnnModScale_body3'
                         'Pend2_analytical body3_analytical'
                         ']'
                    )
parser.add_argument('--net_url', default='', type=str, help='Download net from Internet')
parser.add_argument('--load_net_path', default='', type=str, help='The path to load the pretrained network')

# For training settings
parser.add_argument('--lr', default=0.9, type=float, help='learning rate')
parser.add_argument('--criterion', default='L2_loss', type=str, help='Select criterion to learn')
parser.add_argument('--optimizer', default='adam', type=str, help='Select optimizer to learn')
parser.add_argument('--scheduler', default='LambdaLR', type=str, help='Select scheduler to learn')
parser.add_argument('--iterations', default=20, type=int, help='end of training epoch')
parser.add_argument('--print_every', default=10, type=int, help='number of gradient steps between prints')

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
        'dataset_path': './outputs/data/dataset_'+args.data_name,
        'num_workers': args.num_workers,
        'dtype': args.dtype,
        'device': device
    }
    data = ln.data.get_dataloader(**arguments)

    # net
    arguments = {
        'taskname': args.taskname,
        'net_name': args.net_name,
        'obj': args.obj,
        'dim': args.dim,
        'net_url': args.net_url,
        'load_net_path': args.load_net_path,
        'device': device
    }
    net = ln.nn.get_model(**arguments)

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
    # print(arguments)


def main():
    run()


if __name__ == '__main__':
    main()
