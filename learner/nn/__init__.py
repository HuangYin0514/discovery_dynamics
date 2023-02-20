
import os
import os.path as osp

from learner.utils import timing, count_parameters, download_file_from_google_drive, load_network
from ._base_module import LossNN
from .baseline_body3 import Baseline_body3
from .baseline_pend2 import Baseline_pend2
from .body3_analytical import Body3_analytical
from .hnn_body3 import HNN_body3
from .hnn_pend2 import HNN_pend2
from .hnnmodscale_body3 import HnnModScale_body3
from .hnnmodscale_pend2 import HnnModScale_pend2
from .lnn_body3 import LNN_body3
from .lnn_pend2 import LNN_pend2
from .mechanicsNN_body3 import MechanicsNN_body3
from .mechanicsNN_pend2 import MechanicsNN_pend2
from .modlanet_body3 import ModLaNet_body3
from .modlanet_pend2 import ModLaNet_pend2
from .pend2_L_analytical import Pend2_L_analytical
from .pend2_analytical import Pend2_analytical

__model_factory = {
    'HNN_pend2': HNN_pend2,
    'HNN_body3': HNN_body3,
    'LNN_pend2': LNN_pend2,
    'LNN_body3': LNN_body3,
    'Baseline_pend2': Baseline_pend2,
    'Baseline_body3': Baseline_body3,
    'MechanicsNN_pend2': MechanicsNN_pend2,
    'MechanicsNN_body3': MechanicsNN_body3,
    'ModLaNet_pend2': ModLaNet_pend2,
    'ModLaNet_body3': ModLaNet_body3,
    'HnnModScale_pend2': HnnModScale_pend2,
    'HnnModScale_body3': HnnModScale_body3,
    'Pend2_analytical': Pend2_analytical,
    'Pend2_L_analytical': Pend2_L_analytical,
    'Body3_analytical': Body3_analytical,
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
