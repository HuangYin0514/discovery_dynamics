import warnings
from collections import OrderedDict

import torch


def load_network(network, file_path, device):
    #
    device = torch.device(device)

    # Original saved file with DataParallel
    state_dict = torch.load(file_path, map_location=torch.device(device))

    # state dict
    model_dict = network.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # load model state ---->{matched_layers, discarded_layers}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            print('discarded_layers {}'.format(k[:8]))
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    network.load_state_dict(model_dict)

    # assert model state
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(matched_layers)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(file_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

    return network
