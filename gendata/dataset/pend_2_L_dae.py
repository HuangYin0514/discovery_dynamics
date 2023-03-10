# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:

利用 微分代数方程DAE来生成数据

初始化角度坐标-》转换成笛卡尔坐标-》构造DAE形式-》利用ode求解器进行求解

任务存在问题，不能单样本训练测试，故选取参数时，应避免出现单样本情况
"""
import numpy as np
import torch
from torch import nn

from gendata.dataset._base_body_dataset import BaseBodyDataset
from learner.integrator import ODESolver
from learner.utils import dfx
from learner.utils.common_utils import enable_grad, matrix_inv


class Pendulum2_L_dae(BaseBodyDataset, nn.Module):

    train_url = 'https://drive.google.com/file/d/14qzrrWUjaagEt8591DASUazhchLTRo0e/view?usp=share_link'
    val_url = 'https://drive.google.com/file/d/1q7Q7pZnLeouZNIKnfTRR_4VwKLOSRrd6/view?usp=share_link'
    test_url = 'https://drive.google.com/file/d/1FcUSk61-QUuwWMTJF2kxaD2SfVz0b3EL/view?usp=share_link'

    def __init__(self, obj, dim, m=None, l=None, **kwargs):
        super(Pendulum2_L_dae, self).__init__()

        self.__init_dynamic_variable(obj, dim)

    def __init_dynamic_variable(self, obj, dim):
        self.m = [1., 50.]
        self.l = [1., 1.]
        self.g = 10.

        self.obj = obj
        self.dim = dim
        self.dof = self.obj * self.dim  # degree of freedom

        t0 = 0.

        t_end = 3.0
        dt = 0.001
        _time_step = int((t_end - t0) / dt)
        self.t = torch.linspace(t0, t_end, _time_step)

        t_end = 5.0
        dt = 0.001
        _time_step = int((t_end - t0) / dt)
        self.test_t = torch.linspace(t0, t_end, _time_step)

    @enable_grad
    def forward(self, t, coords):
        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.shape[0]
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        # 拟合 ------------------------------------------------------------------------------
        Minv = self.Minv(x)
        V = self.potential(torch.cat([x, v], dim=-1))

        # 约束 -------------------------------------------------------------------------------
        phi = self.phi_fun(x)

        phi_q = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device)  # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_q[:, i] = dfx(phi[:, i], x)
        phi_qq = torch.zeros(phi.shape[0], phi.shape[1], x.shape[1], dtype=self.Dtype, device=self.Device) # (bs, 2, 4)
        for i in range(phi.shape[1]):
            phi_qq[:, i] = dfx(phi_q[:, i].unsqueeze(-2) @ v.unsqueeze(-1), x)

        # 右端项 -------------------------------------------------------------------------------
        F = -dfx(V, x)

        # 求解 lam ----------------------------------------------------------------
        phiq_Minv = torch.matmul(phi_q, Minv)  # (bs,2,4)
        L = torch.matmul(phiq_Minv, phi_q.permute(0, 2, 1))
        R = torch.matmul(phiq_Minv, F.unsqueeze(-1)) + torch.matmul(phi_qq, v.unsqueeze(-1))  # (2, 1)
        lam = torch.matmul(matrix_inv(L), R)

        # 求解 a ----------------------------------------------------------------
        a_R = F.unsqueeze(-1) - torch.matmul(phi_q.permute(0, 2, 1), lam)  # (4, 1)
        a = torch.matmul(Minv, a_R).squeeze(-1)  # (4, 1)
        return torch.cat([v, a], dim=-1)

    def Minv(self, q):
        bs, states = q.shape

        M = np.diag(np.array([self.m[0], self.m[0], self.m[1], self.m[1]]))
        M = np.tile(M, (bs, 1, 1))
        M = torch.tensor(M, dtype=self.Dtype, device=self.Device)

        Minv = matrix_inv(M)
        return Minv

    def phi_fun(self, x):
        bs, states_num = x.shape
        constraint_1 = x[:, 0] ** 2 + x[:, 1] ** 2 - 1 ** 2
        constraint_2 = (x[:, 0] - x[:, 2]) ** 2 + (x[:, 1] - x[:, 3]) ** 2 - 1 ** 2
        phi = torch.stack((constraint_1, constraint_2), dim=-1)
        return phi  # (bs ,2)

    def kinetic(self, coords):
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)
        T = 0.
        for i in range(self.obj):
            vx = v[:, i * 2]
            vy = v[:, i * 2 + 1]
            T = T + 0.5 * self.m[i] * (vx ** 2 + vy ** 2)
        return T

    def potential(self, coords):
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)
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

    def angle2cartesian(self, angles):
        pos = np.zeros([angles.shape[0], angles.shape[1] * 2])
        num_angles_dim = int(angles.shape[1] / 2)
        for i in range(self.obj):
            if i == 0:
                pos[:, self.dim * i:self.dim * (i + 1)] += np.concatenate(
                    [self.l[i] * np.sin(angles[:, i:i + 1]), -self.l[i] * np.cos(angles[:, i:i + 1])],
                    1)
                pos[:, self.dof + self.dim * i:self.dof + self.dim * (i + 1)] += np.concatenate(
                    [self.l[i] *
                     np.cos(angles[:, i:i + 1]) *
                     angles[:, num_angles_dim + i:num_angles_dim + i + 1],
                     self.l[i] *
                     np.sin(angles[:, i:i + 1]) *
                     angles[:, num_angles_dim + i:num_angles_dim + i + 1]],
                    1)
            else:
                pos[:, self.dim * i:self.dim * (i + 1)] += pos[:, self.dim * (i - 1):self.dim * i] + np.concatenate(
                    [self.l[i] * np.sin(angles[:, i:i + 1]), -self.l[i] * np.cos(angles[:, i:i + 1])],
                    1)
                pos[:, self.dof + self.dim * i:self.dof + self.dim * (i + 1)] += np.concatenate(
                    [self.l[i] *
                     np.cos(angles[:, i:i + 1]) *
                     angles[:, num_angles_dim + i:num_angles_dim + i + 1],
                     self.l[i] *
                     np.sin(angles[:, i:i + 1]) *
                     angles[:, num_angles_dim + i:num_angles_dim + i + 1]],
                    1)
        return pos

    def random_config(self, num):
        x0_list = []
        for i in range(num):
            max_momentum = 0.0
            y0 = np.zeros(self.obj * 2)
            for i in range(self.obj):
                theta = (2 * np.random.rand()) * np.pi
                momentum = (2 * np.random.rand() - 1) * max_momentum
                # momentum = np.zeros(1)
                y0[i] = theta
                y0[i + self.obj] = momentum
            # ----------------------------------------------------------------
            # y0 = np.array([np.pi / 2, np.pi / 2, 0., 0.])
            # y0 = np.array([self.l[0], 0, self.l[0] + self.l[1], 0, 0, 0, 0, 0])
            # ----------------------------------------------------------------
            x0_list.append(y0)
        x0 = np.stack(x0_list)
        x0 = self.angle2cartesian(x0)
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
