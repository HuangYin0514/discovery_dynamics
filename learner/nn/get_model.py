import os
import os.path as osp

from learner.utils import timing, count_parameters, download_file_from_google_drive, load_network
from .baseline import Baseline
from .hnn import HNN
from .hnnmod_pend2 import HnnMod_pend2
from .lnn import LNN
from .mechanicsNN import MechanicsNN
from .modlanet_body3 import ModLaNet_body3
from .modlanet_pend2 import ModLaNet_pend2


def choose_model(net_name, obj, dim):
    if net_name == 'HNN':
        net = HNN(obj, dim)
    elif net_name == 'LNN':
        net = LNN(obj, dim)
    elif net_name == 'Baseline':
        net = Baseline(obj, dim)
    elif net_name == 'MechanicsNN':
        net = MechanicsNN(obj, dim)
    elif net_name == 'ModLaNet_pend2':
        net = ModLaNet_pend2(obj, dim)
    elif net_name == 'ModLaNet_body3':
        net = ModLaNet_body3(obj, dim)
    elif net_name == 'HnnMod_pend2':
        net = HnnMod_pend2(obj, dim)
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
