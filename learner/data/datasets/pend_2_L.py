# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from ._base_body_dataset import BaseBodyDataset
from ...integrator import ODESolver
from ...utils import dfx
from .data_modlanet import Dataset as dmodlanet

class Pendulum2_L(BaseBodyDataset, nn.Module):
    """
    Pendulum with 2 bodies
    Reference:
    # ref: Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
    # URL: https://proceedings.neurips.cc/paper/2020/file/9f655cc8884fda7ad6d8a6fb15cc001e-Paper.pdf
    Dataset statistics:
    # type: hamilton
    # obj: 2
    # dim: 1
    """

    def __init__(self, train_num, test_num, obj, dim, m=None, l=None, **kwargs):
        super(Pendulum2_L, self).__init__()

        self.train_num = train_num
        self.test_num = test_num
        self.dataset_url = ''

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self.m = [1 for i in range(obj)]
        self.l = [1 for i in range(obj)]
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom

        self.dt = 0.1

        t0 = 0.
        t_end = 10.
        _time_step = int((t_end - t0) / self.dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 30.
        dt = 0.05
        _time_step = int((t_end - t0) / dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    def Init_data(self,num_samples=100,test_split=0.9):
        dataset = dmodlanet(obj=self.obj, m=[1 for i in range(self.obj)], l=[1 for i in range(self.obj)])
        data = dataset.get_dataset(seed=0, system='modlanet', noise_std=0.0, samples=num_samples)

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
        for i in range(len(x0)):
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
        for i in range(len(test_x0)):
            test_dataset.append(
                (test_x0[i], test_t, test_dt, test_X[i * 101:(i + 1) * 101].detach().clone(), test_y[i * 101:(i + 1) * 101].detach().clone(),
                 test_E[i * 101:(i + 1) * 101]))

        self.train = train_dataset
        self.test = test_dataset

    def forward(self, t, coords):
        pass

    def kinetic(self, coords):
        pass

    def potential(self, coords):
        pass

    def energy_fn(self, coords):
        pass

    def random_config(self, num):
        pass

    def generate_random(self, num, t):
        pass

    def generate(self, x0, t):
        pass