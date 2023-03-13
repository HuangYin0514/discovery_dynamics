import os
import os.path as osp

from learner.utils import timing, count_parameters, download_file_from_google_drive, load_network
from ._base_module import LossNN
from .analytical_body3 import Analytical_body3
from .analytical_pend2 import Analytical_pend2
from .analytical_pend2_dae import Analytical_pend2_dae
from .baseline_body3 import Baseline_body3
from .baseline_pend2 import Baseline_pend2
from .baseline_pend2_dae import Baseline_pend2_dae
from .clnn_pend2_dae import CLNN_pend2_dae
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
from .sclnn_pend2_dae import SCLNN_pend2_dae

__model_factory = {
    'HNN_pend2': HNN_pend2,
    'HNN_body3': HNN_body3,
    'LNN_pend2': LNN_pend2,
    'LNN_body3': LNN_body3,
    'Baseline_pend2': Baseline_pend2,
    'Baseline_body3': Baseline_body3,
    'Baseline_pend2_dae': Baseline_pend2_dae,
    'MechanicsNN_pend2': MechanicsNN_pend2,
    'MechanicsNN_body3': MechanicsNN_body3,
    'ModLaNet_pend2': ModLaNet_pend2,
    'ModLaNet_body3': ModLaNet_body3,
    'HnnModScale_pend2': HnnModScale_pend2,
    'HnnModScale_body3': HnnModScale_body3,
    'Analytical_pend2': Analytical_pend2,
    'Analytical_body3': Analytical_body3,
    'Analytical_pend2_dae': Analytical_pend2_dae,
    'CLNN_pend2_dae': CLNN_pend2_dae,
    'SCLNN_pend2': SCLNN_pend2_dae

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
