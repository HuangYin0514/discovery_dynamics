from torch import nn

from .base_module import StructureNN
from .utils_nn import weights_init_xavier_normal


class FNN(StructureNN):
    '''Fully connected neural networks.
    '''

    def __init__(self, ind, outd, layers=1, width=200, softmax=False):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width

        self.softmax = softmax

        self.__init_modules()
        self.__initialize()

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def __init_modules(self):
        self.input_layer = nn.Sequential()
        self.input_layer.add_module('input', nn.Linear(self.ind, self.width))
        self.input_layer.add_module('act', nn.Tanh())

        # hidden_bock = nn.Sequential(OrderedDict([
        #     ('hidden', nn.Linear(self.width, self.width)),
        #     ('act', nn.Tanh()),
        # ]))
        hidden_bock = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.Tanh()
        )
        self.hidden_layer = nn.ModuleList([hidden_bock for _ in range(self.layers)])

        self.output_layer = nn.Sequential()
        self.output_layer.add_module('output', nn.Linear(self.width, self.outd, bias=False))

    def __initialize(self):
        self.input_layer.apply(weights_init_xavier_normal)
        self.hidden_layer.apply(weights_init_xavier_normal)
        self.output_layer.apply(weights_init_xavier_normal)
