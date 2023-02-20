# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import numpy as np
import torch
from torch import nn

from ._base_body_dataset import BaseBodyDataset


class Pend_2_modlanet(BaseBodyDataset, nn.Module):
    """
        测试modlanet数据
    """

    def __init__(self, train_num, test_num, obj, dim, m=None, l=None, **kwargs):
        super(Pend_2_modlanet, self).__init__()

        self.train_num = 90
        self.test_num = 20
        self.dataset_url = 'https://drive.google.com/file/d/1qgAqjZDe76_GCyJqtS--8Llk-gwWl1D1/view?usp=share_link'

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self.m = [1 for i in range(obj)]
        self.l = [1 for i in range(obj)]
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom



    def Init_data(self):
        filename = '/Users/drhuang/PycharmProjects/testProject/format_datasets/dataset_2_pend_modlanet_noise_0.0.npy'
        data = np.load(filename, allow_pickle=True).item()

        x = data['x'] % (2 * np.pi)
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        v = torch.tensor(data['v'], requires_grad=True, dtype=torch.float32)
        a = torch.tensor(data['ac'], dtype=torch.float32)

        test_x = data['test_x'] % (2 * np.pi)
        test_x = torch.tensor(test_x, requires_grad=True, dtype=torch.float32)
        test_v = torch.tensor(data['test_v'], requires_grad=True, dtype=torch.float32)
        test_a = torch.tensor(data['test_ac'])

        # ----------------------------------------------------------------
        # data format (x0[i], t, self.dt, X[i], y[i], E[i])
        datalen = 101
        x0 = torch.cat([x[::datalen], v[::datalen]], dim=-1).detach().clone()
        t = torch.linspace(0, 10, 101).detach().clone()
        dt = 0.1
        X = torch.cat([x, v], dim=-1).detach().clone()
        y = torch.cat([v, a], dim=-1).detach().clone()
        E = data['E']
        train_dataset = []
        for i in range(90):
            train_dataset.append(
                (x0[i], t, dt, X[i * 101:(i + 1) * 101].detach().clone(), y[i * 101:(i + 1) * 101].detach().clone(),
                 E[i * 101:(i + 1) * 101]))

        test_x0 = torch.cat([test_x[::datalen], test_v[::datalen]], dim=-1).detach().clone()
        test_t = torch.linspace(0, 10, 101).detach().clone()
        test_dt = 0.1
        test_X = torch.cat([test_x, test_v], dim=-1).detach().clone()
        test_y = torch.cat([test_v, test_a], dim=-1).detach().clone()
        test_E = data['test_E']
        test_dataset = []
        for i in range(10):
            test_dataset.append(
                (x0[i], t, dt, X[i * 101:(i + 1) * 101].detach().clone(), y[i * 101:(i + 1) * 101].detach().clone(),
                 E[i * 101:(i + 1) * 101]))

        self.train = train_dataset
        self.test = test_dataset

    def random_config(self, num):
        pass

    def energy_fn(self, y):
        pass
