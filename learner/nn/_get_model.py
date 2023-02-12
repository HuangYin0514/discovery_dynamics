import os
import os.path as osp

from learner.utils import timing, count_parameters, download_file_from_google_drive, load_network
from .baseline import Baseline
from .baseline_pend2 import Baseline_pend2
from .hnn import HNN
from .hnnmod_body3 import HnnMod_body3
from .hnnmod_pend2 import HnnMod_pend2
from .hnnmod_pend2_anlytical import HnnMod_pend2_anlytical
from .hnnmodscale_body3 import HnnModScale_body3
from .hnnmodscale_pend2 import HnnModScale_pend2
from .lnn import LNN
from .mechanicsNN import MechanicsNN
from .modlanet_body3 import ModLaNet_body3
from .modlanet_pend2 import ModLaNet_pend2
from .pend2_analytical import Pend2_analytical

__model_factory = {
    'HNN': HNN,
    'LNN': LNN,
    'Baseline': Baseline,
    'Baseline_pend2':Baseline_pend2,
    'MechanicsNN': MechanicsNN,
    'ModLaNet_pend2': ModLaNet_pend2,
    'ModLaNet_body3': ModLaNet_body3,
    'HnnMod_pend2': HnnMod_pend2,
    'HnnMod_body3': HnnMod_body3,
    'HnnMod_pend2_anlytical': HnnMod_pend2_anlytical,
    'HnnModScale_pend2': HnnModScale_pend2,
    'HnnModScale_body3': HnnModScale_body3,
    'Pend2_analytical': Pend2_analytical,
}


def choose_model(net_name, obj, dim):
    if net_name not in __model_factory.keys():
        raise ValueError('Model \'{}\' is not implemented'.format(net_name))
    net = __model_factory[net_name](obj, dim)
    return net


@timing
def get_model(taskname, net_name, obj, dim, net_url, load_net_path, device):
    print('Start get models.')
    net = choose_model(net_name, obj, dim)
    print("======> {} loaded".format(net_name))
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
