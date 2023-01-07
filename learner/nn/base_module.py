import abc

import torch

from ..criterion import L2_norm_loss


class BaseModule(torch.nn.Module):
    '''Standard module format. 
    '''

    def __init__(self):
        super(BaseModule, self).__init__()
        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'cuda':
            self.cuda()
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'cuda':
            return torch.device('cuda')

    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64


class StructureNN(BaseModule):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''

    def __init__(self):
        super(StructureNN, self).__init__()

    def predict(self, x, returnnp=False):
        return self(x).cpu().detach().numpy() if returnnp else self(x)


class LossNN(BaseModule, abc.ABC):
    '''Loss-oriented neural network used as an algorithm based on designing loss.
    '''

    def __init__(self):
        super(LossNN, self).__init__()

    def criterion(self, y_hat, y, criterion_method='MSELoss'):
        if criterion_method == 'MSELoss':
            return torch.nn.MSELoss()(y_hat, y)
        elif criterion_method == 'L2_norm_loss':
            return L2_norm_loss(y_hat, y)
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def integrate(self, X, t):
        pass
