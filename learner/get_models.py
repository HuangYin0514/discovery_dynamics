import os

from .nn import HNN
from .utils import count_parameters, load_network, download_file_from_google_drive, timing


def choose_model(net_name, obj, dim):
    if net_name == 'hnn':
        input_dim = obj * dim * 2
        net = HNN(dim=input_dim, layers=1, width=200)
    else:
        raise ValueError('Model \'{}\' is not implemented'.format(net_name))

    return net



@timing
def get_model(taskname, net_name, obj, dim, net_url, load_net_path, device):
    net = choose_model(net_name, obj, dim)

    print('Number of parameters in model: ', count_parameters(net))

    if len(net_url) != 0:
        print('Start downloading net.')
        data_path = './outputs/' + taskname
        os.makedirs(data_path) if not os.path.exists(data_path) else None

        # example: model-pend_2_hnn.pkl
        filename = data_path + '/model_{}.pkl'.format(taskname)

        download_file_from_google_drive(net_url, filename)


    if os.path.exists(load_net_path):
        load_network(net, load_net_path, device)
    elif len(load_net_path):
        print('Cannot find pretrained network at \'{}\''.format(load_net_path), flush=True)

    return net
