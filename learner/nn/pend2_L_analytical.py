# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/2/12 9:56 AM
@desc:
"""

# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/18 4:41 PM
@desc:
"""
import torch

from ._base_module import LossNN
from ..integrator import ODESolver
from ..utils import dfx


class Pend2_L_analytical(LossNN):
    """
    Mechanics neural networks.
    """

    def __init__(self, obj, dim, num_layers=None, hidden_dim=None):
        super(Pend2_L_analytical, self).__init__()

        q_dim = int(obj * dim)
        p_dim = int(obj * dim)

        self.obj = obj
        self.dim = dim
        self.dof = int(obj * dim)

        self.global_dim = 2
        self.global_dof = int(obj * self.global_dim)

        self.mass = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.mass.weight)

    def forward(self, t, coords):
        __x, __p = torch.chunk(coords, 2, dim=-1)
        coords = torch.cat([__x % (2 * torch.pi), __p], dim=-1).clone().detach().requires_grad_(True)

        coords = coords.clone().detach().requires_grad_(True)
        bs = coords.size(0)
        x, v = coords.chunk(2, dim=-1)  # (bs, q_dim) / (bs, p_dim)

        x_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        v_global = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)

        x_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)
        v_origin = torch.zeros((bs, self.global_dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.obj):
            for j in range(i):
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += x_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
                v_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim] += v_global[:, (j) * self.global_dim:
                                                                                             (j + 1) * self.global_dim]
            x_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = \
                torch.cat(
                    [torch.sin(x[:, (i) * self.dim: (i + 1) * self.dim]),
                     torch.cos(x[:, (i) * self.dim: (i + 1) * self.dim])],
                    dim=-1) + \
                x_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim]

            v_global[:, (i) * self.global_dim: (i + 1) * self.global_dim] = \
                torch.cat(
                    [torch.sin(x[:, (i) * self.dim: (i + 1) * self.dim]),
                     torch.cos(x[:, (i) * self.dim: (i + 1) * self.dim])],
                    dim=-1) * v[:, (i) * self.dim: (i + 1) * self.dim] + \
                v_origin[:, (i) * self.global_dim: (i + 1) * self.global_dim]

        # Calculate the potential energy for i-th element ------------------------------------------------------------
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = -x_global[:, i * self.global_dim + 1]
            U = U + 9.8 * y

        # Calculate the kinetic --------------------------------------------------------------
        T = 0.
        vx, vy = 0., 0.
        for i in range(self.dof):
            vx = v_global[:, i * self.global_dim]
            vy = v_global[:, i * self.global_dim + 1]
            vv = v_global[:, (i) * self.global_dim: (i + 1) * self.global_dim].pow(2).sum(-1)
            T += 0.5 * vv

        # Calculate the Hamilton Derivative --------------------------------------------------------------
        L = T - U
        dvL = dfx(L.sum(), v)
        dxL = dfx(L.sum(), x)

        dvdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)
        dxdvL = torch.zeros((bs, self.dof, self.dof), dtype=self.Dtype, device=self.Device)

        for i in range(self.dof):
            dvidvL = dfx(dvL[:, i].sum(), v)
            if dvidvL is None:
                print("i am is none of dvidvL")
            dvdvL[:, i, :] += dvidvL

        for i in range(self.dof):
            dxidvL = dfx(dvL[:, i].sum(), x)
            if dxidvL is None:
                print("i am is none of dxidvL")
            dxdvL[:, i, :] += dxidvL

        dvdvL_inv = torch.linalg.pinv(dvdvL)

        a = dvdvL_inv @ (dxL.unsqueeze(2) - dxdvL @ v.unsqueeze(2))  # (bs, a_dim, 1)
        a = a.squeeze(2)
        return torch.cat([v, a], dim=-1)

    def integrate(self, X0, t):
        out = ODESolver(self, X0, t, method='rk4').permute(1, 0, 2)  # (T, D) dopri5 rk4
        return out
