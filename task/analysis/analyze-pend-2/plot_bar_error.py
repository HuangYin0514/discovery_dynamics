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
    t0, t_end, h = 0, 5, 0.02

    # ground truth
    pendulumData = PendulumData(obj=args.obj, dim=args.dim,
                                train_num=args.train_num,
                                test_num=args.test_num,
                                m=[1 for i in range(args.obj)],
                                l=[1 for i in range(args.obj)])
    # solver = ln.integrator.rungekutta.RK4(pendulumData.hamilton_right_fn, t0=t0, t_end=t_end)
    truth_t = np.arange(t0, t_end, h)
    solver = ln.integrator.rungekutta.RK45(pendulumData.hamilton_right_fn, t0=t0, t_end=t_end)

    # net
    input_dim = args.obj * args.dim * 2
    hnn = ln.nn.HNN(dim=input_dim, layers=1, width=200)
    local = path + 'pend_2_hnn/model-pend_2_hnn.pkl'
    # local_url = 'https://drive.google.com/file/d/1bMzjQvPQZW2ByRQw0I0IQ67U2p8xzkh2/view?usp=share_link'
    # ln.utils.download_file_from_google_drive(local_url, local)
    ln.utils.load_network(hnn, local, device)
    hnn.device = device
    hnn.dtype = args.dtype


if __name__ == '__main__':
    main()
    print('Done!')