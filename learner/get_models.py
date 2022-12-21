import os

from .nn import HNN
from .utils import count_parameters, load_network


def get_model(args, device):
    if args.net == 'hnn':
        input_dim = args.obj * args.dim * 2
        net = HNN(dim=input_dim, layers=args.layers, width=args.width)
    else:
        raise ValueError('Model \'{}\' is not implemented'.format(args.net))

    print('Number of parameters in model: ', count_parameters(net))

    if os.path.exists(args.load_net_path):
        load_network(net, args.load_net_path, device)
    elif len(args.load_net_path):
        print('Cannot find pretrained network at \'{}\''.format(args.load_net_path), flush=True)

    return net
