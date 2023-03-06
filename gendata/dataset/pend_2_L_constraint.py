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

from gendata.dataset._base_body_dataset import BaseBodyDataset
from learner.integrator import ODESolver
from learner.utils import dfx
from learner.utils.common_utils import jacobian_fx


class Pendulum2_L_constraint(BaseBodyDataset, nn.Module):

    def __init__(self, obj, dim, m=None, l=None, **kwargs):
        super(Pendulum2_L_constraint, self).__init__()

        self.train_url = 'https://drive.google.com/file/d/1kTP5WtPT78rX7HBcMo2BBU9ug6RGpZ8E/view?usp=share_link'
        self.val_url = 'https://drive.google.com/file/d/1RXTP-fxTECHY6ZCP_rqkaL90k8EQd_f7/view?usp=share_link'
        self.test_url = 'https://drive.google.com/file/d/1kTP5WtPT78rX7HBcMo2BBU9ug6RGpZ8E/view?usp=share_link'

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self.m = [10 for i in range(obj)]
        self.l = [10 for i in range(obj)]
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom

        t0 = 0.

        t_end = 10.
        dt = 0.01
        _time_step = int((t_end - t0) / dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 3.
        dt = 0.01
        _time_step = int((t_end - t0) / dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        Minv = self.Minv(x)

        phi_q = jacobian_fx(self.phi_fun, x)  # (bs, 2, 4)
        phi_q = torch.einsum("bibj->bij", phi_q)  # (bs, 2, 4)

        phi_qq = jacobian_fx(self.D_phi_fun, (x, v))[0]  # (bs, 4)
        phi_qq = torch.einsum("bibj->bij", phi_qq)  # (bs, 2, 4)

        V = self.potential(x)

        F = -dfx(V, x)

        # 求解 lam ----------------------------------------------------------------
        L = phi_q @ Minv @ phi_q.permute(0, 2, 1)
        R = (phi_q @ Minv @ F.unsqueeze(-1) + phi_qq@ v.unsqueeze(-1))  # (2, 1)
        lam = torch.linalg.pinv(L) @ R  # (2, 1)

        # 求解 vdot ----------------------------------------------------------------
        a_R = F.unsqueeze(-1) - phi_q.permute(0, 2, 1) @ lam   # (4, 1)
        a = (Minv @ a_R).squeeze(-1)  # (4, 1)
        return torch.cat([v, a], dim=-1)

    def Minv(self, q):
        bs, states = q.shape

        M = np.diag(np.array([self.m[0], self.m[0], self.m[1], self.m[1]]))
        M = np.tile(M, (bs, 1, 1))
        M = torch.tensor(M, dtype=self.Dtype, device=self.Device)

        Minv = torch.linalg.pinv(M)
        return Minv

    def phi_fun(self, x):
        bs, states_num = x.shape
        constraint_1 = x[:, 0] ** 2 + x[:, 1] ** 2 - 1 ** 2
        constraint_2 = (x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2 - 1 ** 2
        phi = torch.stack((constraint_1, constraint_2), dim=-1)
        return phi  # (bs ,2)

    def D_phi_fun(self, x, v):
        phi_q = jacobian_fx(self.phi_fun, x)  # (bs, 2, 4)
        phi_q = torch.einsum("bibj->bij", phi_q)
        return (phi_q @ v.unsqueeze(-1)).squeeze(-1)  # (bs ,2)

    def kinetic(self, v):
        T = 0.
        for i in range(self.obj):
            T = T + 0.5 * self.m[i] * torch.sum(v[:, 2 * i: 2 * i + 2] ** 2, dim=1)
        return T

    def potential(self, x):
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = x[:, i * 2 + 1]
            U = U + self.m[i] * self.g * y
        return U

    def energy_fn(self, coords):
        """energy function """
        H = self.kinetic(coords) + self.potential(coords)
        return H

    def random_config(self, num):
        x0_list = []
        for i in range(num):
            max_momentum = 1.
            y0 = np.zeros(self.obj * 2)
            for i in range(self.obj):
                theta = (2 * np.random.rand()) * np.pi
                momentum = (2 * np.random.rand() - 1) * max_momentum
                y0[i] = theta
                y0[i + self.obj] = momentum
                # ----------------------------------------------------------------
                y0 = np.array([self.l[0], 0, self.l[0] + self.l[1], 0, 0, 0, 0, 0])
                # ----------------------------------------------------------------
            x0_list.append(y0)
        x0 = np.stack(x0_list)
        return torch.tensor(x0, dtype=self.Dtype, device=self.Device)

    def ode_solve_traj(self, x0, t):
        x0 = x0.to(self.Device)
        t = t.to(self.Device)
        # At small step sizes, the differential equations exhibit stiffness and the rk4 solver cannot solve
        # the double pendulum task. Therefore, use dopri5 to generate training data.
        if len(t) == len(self.test_t):
            # test stages
            x = ODESolver(self, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        else:
            # train stages
            x = ODESolver(self, x0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return x
