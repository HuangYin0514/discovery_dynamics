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
from ...integrator import ODESolver
from ...utils import lazy_property, dfx


class Pendulum2(BaseBodyDataset, nn.Module):
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
        super(Pendulum2, self).__init__()

        self.train_num = train_num
        self.test_num = test_num
        self.dataset_url = 'https://drive.google.com/file/d/1gFpZaOsaL8-ooXs6Cn-yfisT8U6S12Qk/view?usp=share_link'

        self.Dtype = torch.float32
        self.Device = torch.device('cpu')

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self._m = [1 for i in range(obj)]
        self._l = [1 for i in range(obj)]
        self._g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom

        self.dt = 0.01

        t0 = 0.
        t_end = 10.
        _time_step = int((t_end - t0) / self.dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 30.
        dt = 0.05
        _time_step = int((t_end - t0) / dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    @lazy_property
    def J(self):
        # [ 0, I]
        # [-I, 0]
        d = self._dof
        res = np.eye(self._dof * 2, k=d) - np.eye(self._dof * 2, k=-d)
        return torch.tensor(res).float()

    def forward(self, t, coords):
        # __x, __p = torch.chunk(coords, 2, dim=-1)
        # __x = __x % (2 * torch.pi)
        # coords = torch.cat([__x, __p], dim=-1).clone().detach().requires_grad_(True)

        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.size(0)
        x, p = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # Calculate the potential energy for i-th element ------------------------------------------------------------
        U = self.potential(torch.cat([x, p], dim=-1))

        # Calculate the kinetic --------------------------------------------------------------
        T = self.kinetic(torch.cat([x, p], dim=-1))

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        H = U + T
        dqH = dfx(H.sum(), x)
        dpH = dfx(H.sum(), p)

        # Calculate the Derivative ----------------------------------------------------------------
        dq_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)
        dp_dt = torch.zeros((bs, self.dof), dtype=self.Dtype, device=self.Device)

        dq_dt = dpH
        dp_dt = -dqH

        dz_dt = torch.cat([dq_dt, dp_dt], dim=-1)

        return dz_dt

    def M(self, x):
        N = self.obj
        M = torch.zeros((x.shape[0], N, N), dtype=self.Dtype, device=self.Device)
        for i in range(N):
            for k in range(N):
                m_sum = 0
                j = i if i >= k else k
                for tmp in range(j, N):
                    m_sum += 1.0
                M[:, i, k] = torch.cos(x[:, i] - x[:, k]) * m_sum
        return M

    def Minv(self, x):
        return torch.linalg.inv(self.M(x))

    def kinetic(self, coords):
        """Kinetic energy"""

        s, num_states = coords.shape
        assert num_states == self.dof * 2

        x, p = torch.chunk(coords, 2, dim=1)
        T = 0.
        v = torch.matmul(self.Minv(x), p.unsqueeze(-1))
        T = 0.5 * torch.matmul(p.unsqueeze(1), v).squeeze(-1).squeeze(-1)
        return T

    def potential(self, coords):
        bs, num_states = coords.shape
        assert num_states == self.dof * 2
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = y - torch.cos(coords[:, i])
            U = U + 9.8 * y
        return U

    def energy_fn(self, coords):
        """energy function """
        H = self.kinetic(coords) + self.potential(coords)
        return H

    def random_config(self, num):
        x0_list = []
        for i in range(num):
            max_momentum = 1.
            x0 = torch.zeros((self.obj * 2))
            for i in range(self.obj):
                theta = (2 * np.pi) * torch.rand(1, ) + 0  # [0, 2pi]
                momentum = (2 * torch.rand(1, ) - 1) * max_momentum  # [-1, 1]*max_momentum
                x0[i] = theta
                x0[i + self.obj] = momentum
            x0_list.append(x0)
        x0 = torch.stack(x0_list)
        return x0

    def generate(self, x0, t):
        print("Generating for new function!")

        def angle_forward(t, coords):
            x, p = torch.chunk(coords, 2, dim=0)
            new_x = x % (2 * torch.pi)
            new_coords = torch.cat([new_x, p], dim=-1).clone().detach()
            return self(t, new_coords)

        out = ODESolver(self, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
