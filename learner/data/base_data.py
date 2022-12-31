"""
@author: yin huang
"""
import abc

import numpy as np
import torch


class BaseData(abc.ABC):
    '''Standard data format. 
    '''

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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
            self.__to_cpu()
        elif d == 'cuda':
            self.__to_gpu()
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.__to_float()
        elif d == 'double':
            self.__to_double()
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

    @property
    def X_train_np(self):
        return BaseData.to_np(self.X_train)

    @property
    def y_train_np(self):
        return BaseData.to_np(self.y_train)

    @property
    def X_test_np(self):
        return BaseData.to_np(self.X_test)

    @property
    def y_test_np(self):
        return BaseData.to_np(self.y_test)

    @staticmethod
    def to_np(d):
        if isinstance(d, np.ndarray) or d is None:
            return d
        elif isinstance(d, torch.Tensor):
            return d.cpu().detach().numpy()
        else:
            raise ValueError

    def __to_cpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.DoubleTensor(getattr(self, d)))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cpu())

    def __to_gpu(self):
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), np.ndarray):
                setattr(self, d, torch.cuda.DoubleTensor(getattr(self, d)))
            elif isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).cuda())

    def __to_float(self):
        if self.device is None:
            raise RuntimeError('device is not set')
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).float())

    def __to_double(self):
        if self.device is None:
            raise RuntimeError('device is not set')
        for d in ['X_train', 'y_train', 'X_test', 'y_test']:
            if isinstance(getattr(self, d), torch.Tensor):
                setattr(self, d, getattr(self, d).double())


class BaseDynamicsData(BaseData):
    def __init__(self):
        super(BaseDynamicsData, self).__init__()

        self.train_num = None
        self.test_num = None

        self.h = None

        self.solver = None

    def Init_data(self):
        self.__init_data()

    def __init_data(self):
        self.X_train, self.y_train = self.__generate_random(self.train_num, self.h)
        self.X_test, self.y_test = self.__generate_random(self.test_num, self.h)

    def __generate_random(self, num, h):
        x0 = self.random_config(num)
        X = self.__generate(x0, h)
        X = np.concatenate(X)
        y = np.asarray(list(map(lambda x: self.hamilton_right_fn(None, x), X)))
        # E = np.array([self.hamilton_energy_fn(y) for y in X])
        return X, y

    def __generate(self, X, h):
        X = np.array(list(map(lambda x: self.solver.solve(x, h), X)))
        return X

    @abc.abstractmethod
    def random_config(self, num):
        pass

    @abc.abstractmethod
    def hamilton_right_fn(self, t, x):
        pass
