import torch
import torch.nn.functional as F
from torch import nn

from .layers import SpectralConv1d
from .utils import get_device, dfx
from .utils_model import weights_init_xavier_normal

device = get_device()


class Baseline(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 output_dim,
                 init='xavier',
                 *args, **kwargs):
        super(Baseline, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

        self.nonlinearity = nn.Tanh()

        self.baseline = nn.Sequential(
            self.linear1,
            self.nonlinearity,
            self.linear2,
            self.nonlinearity,
            self.linear3)

        if init == 'xavier':
            self.baseline.apply(weights_init_xavier_normal)
        else:
            raise ValueError('Unsupported init function. Please update it by your own.')

    def forward(self, x):
        # traditional forward pass
        h = self.baseline(x)

        return h


class HNN(torch.nn.Module):

    def __init__(self, input_dim,
                 hidden_dim,
                 init='xavier',
                 *args, **kwargs):
        super(HNN, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 1, bias=False)

        self.nonlinearity = nn.Tanh()

        self.baseline = nn.Sequential(
            self.linear1,
            self.nonlinearity,
            self.linear2,
            self.nonlinearity,
            self.linear3)

        if init == 'xavier':
            self.baseline.apply(weights_init_xavier_normal)
        else:
            raise ValueError('Unsupported init function. Please update it by your own.')

        self.J = self.permutation_tensor(input_dim).to(device)  # Levi-Civita permutation tensor

    def forward(self, x):
        # traditional forward pass
        h = self.baseline(x)

        # hamiltonian_fn
        h = self.hamiltonian_fn(h, x)
        return h

    def hamiltonian_fn(self, h, x):
        dh = dfx(h, x)
        hamiltonian = self.J @ dh.T  # hamiltonian shape is (vector, batchsize)
        return hamiltonian.T

    def permutation_tensor(self, n):
        # [ 0, 1]
        # [-1, 0]
        J = torch.eye(n)
        J = torch.cat([J[n // 2:], -J[:n // 2]])
        return J


class LNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''

    def __init__(self, input_dim, hidden_dim, init='xavier',
                 *args, **kwargs):
        super(LNN, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, 2, bias=False)

        self.nonlinearity = nn.Tanh()

        self.baseline = nn.Sequential(
            self.linear1,
            self.nonlinearity,
            self.linear2,
            self.nonlinearity,
            self.linear3)

        if init == 'xavier':
            self.baseline.apply(weights_init_xavier_normal)
        else:
            raise ValueError('Unsupported init function. Please update it by your own.')

    def forward(self, x):
        # traditional forward pass
        h = self.baseline(x)

        h = self.lagrange_fn(h, x)
        return h

    def lagrange_fn(self, L, coords):
        # 初始化信息
        create_graph = True
        # x, v = torch.split(coords, 2, dim=1)
        n = coords.shape[1] // 2

        # jacobian 矩阵
        dxL = torch.autograd.grad(L.sum(), coords, create_graph=create_graph)[0][:, 0:n]
        dvL = torch.autograd.grad(L.sum(), coords, create_graph=create_graph)[0][:, n:2 * n]
        dvdvL = torch.zeros((coords.shape[0], n, n), dtype=torch.float32, device=device)
        dxdvL = torch.zeros((coords.shape[0], n, n), dtype=torch.float32, device=device)

        # hessian 矩阵
        for i in range(n):
            dxidvL = torch.autograd.grad(dvL[:, i].sum(), coords, create_graph=create_graph, allow_unused=True)[0][:, 0:n]
            if dxidvL is None:
                break
            else:
                dxdvL[:, i, :] += dxidvL

        for i in range(n):
            dvidvL = torch.autograd.grad(dvL[:, i].sum(), coords, create_graph=create_graph, allow_unused=True)[0][:, n:2 * n]
            if dvidvL is None:
                break
            else:
                dvdvL[:, i, :] += dvidvL

        # 欧拉-拉格朗日方程
        # in version 1.8.1 you can use torch.linalg.inv() to replace torch.inverse()
        inv =  torch.linalg.inv(dvdvL)
        a = torch.matmul(inv, (dxL.unsqueeze(2) - dxdvL @ coords[:, 0:n].unsqueeze(2)))
        return a.squeeze(2)


class FNO1d(nn.Module):
    def __init__(self, input_dim, hidden_dim, width=64, fc_dim=128, *args, **kwargs):
        super(FNO1d, self).__init__()

        """
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = hidden_dim
        self.width = width
        self.input_dim = input_dim
        self.output_dim = 1
        self.fc_dim = fc_dim

        self.fc0 = nn.Linear(self.input_dim, self.width)  # input channel is (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.output_dim)

    def forward(self, x):
        # FNO1d forward
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    test_input = torch.randn(20, 4, 1).to(device)
    test_input.requires_grad = True
    """    
    model = HNN(input_dim=2, hidden_dim=200).to(device)
    model = LNN(input_dim=2, hidden_dim=200).to(device)
    model = Baseline(input_dim=4, hidden_dim=200).to(device)
    """
    model = FNO1d(input_dim=1, hidden_dim=3).to(device)
    test_output = model(test_input)
    print(test_output.shape)

    print("done!")
