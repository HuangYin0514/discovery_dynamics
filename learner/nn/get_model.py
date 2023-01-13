import os
import os.path as osp

from learner.utils import timing, count_parameters, download_file_from_google_drive, load_network
from .baseline import Baseline
from .hnn import HNN
from .mechanicsNN import MechanicsNN


def choose_model(net_name, obj, dim):
    if net_name == 'hnn':
        input_dim = obj * dim * 2
        net = HNN(dim=input_dim, layers=1, width=200)
    elif net_name == 'baseline':
        input_dim = obj * dim * 2
        net = Baseline(dim=input_dim, layers=1, width=200)
    elif net_name == 'mechanicsNN':
        input_dim = obj * dim * 2
        net = MechanicsNN(dof=input_dim, layers=3, width=80)
    else:
        raise ValueError('Model \'{}\' is not implemented'.format(net_name))

    return net


@timing
def get_model(taskname, net_name, obj, dim, net_url, load_net_path, device):
    print('=> Start get models.')
    net = choose_model(net_name, obj, dim)
    print("=> {} loaded".format(net_name))
    print('Number of parameters in model: ', count_parameters(net))

    if len(net_url) != 0:
        print('=>Start downloading net.')
        data_path = osp.join('./outputs/', taskname)

        os.makedirs(data_path) if not os.path.exists(data_path) else None

        # example: model-pend_2_hnn.pkl
        filename = osp.join(data_path, '/model-{}.pkl'.format(taskname))

        download_file_from_google_drive(net_url, filename)

    if os.path.exists(load_net_path):
        load_network(net, load_net_path, device)
    elif len(load_net_path):
        print('Cannot find pretrained network at \'{}\''.format(load_net_path), flush=True)

    return net.to(device)
