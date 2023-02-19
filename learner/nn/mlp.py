from torch import nn

from .utils_nn import weights_init_xavier_normal, weights_init_orthogonal_normal


class MLP(nn.Module):
    '''Fully connected neural networks.
    '''

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, act=nn.Tanh):
        super(MLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act()
        )

        module_list = []
        for i in range(num_layers):
            module_list.append(nn.Linear(hidden_dim, hidden_dim))
            module_list.append(act())
        self.hidden_layer = nn.Sequential(*module_list)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

        self.__initialize()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out

    def __initialize(self):
        # self.input_layer.apply(weights_init_xavier_normal)
        # self.hidden_layer.apply(weights_init_xavier_normal)
        # self.output_layer.apply(weights_init_xavier_normal)

        self.input_layer.apply(weights_init_orthogonal_normal)
        self.hidden_layer.apply(weights_init_orthogonal_normal)
        self.output_layer.apply(weights_init_orthogonal_normal)

