import autograd
import autograd.numpy as np

from .base_data import BaseData
from learner.integrator.rungekutta import RK4, RK45


class BodyData(BaseData):
    def __init__(self, obj, dim, train_num, test_num, m=None, l=None, **kwargs):
        super(BodyData, self).__init__()

        self.train_num = train_num
        self.test_num = test_num

        self.m = m
        self.l = l
        self.g = 9.8

        self.obj = obj
        self.dim = dim
        self.dof = obj * self.dim  # degree of freedom

        t0 = 0
        t_end = 10
        self.h = 0.05
        self.solver = RK45(self.hamilton_right_fn, t0=t0, t_end=t_end)

        self.k = 1  # body equation parameter

    def Init_data(self):
        self.__init_data()

    def hamiltonian_kinetic(self, coords):
        T = 0.
        for i in range(self.obj):
            T = T + 0.5 * np.sum(coords[self.dof + 2 * i: self.dof + 2 * i + 2] ** 2, axis=0) / self.m[i]
        return T

    def hamiltonian_potential(self, coords):
        k = self.k
        U = 0.
        for i in range(self.obj):
            for j in range(i):
                U = U - k * self.m[i] * self.m[j] / (
                        (coords[2 * i] - coords[2 * j]) ** 2 +
                        (coords[2 * i + 1] - coords[2 * j + 1]) ** 2) ** 0.5
        return U

    def hamilton_energy_fn(self, coords):
        """能量函数"""
        assert (len(coords) == self.dof * 2)
        T, U = self.hamiltonian_kinetic(coords), self.hamiltonian_potential(coords)
        # NOT STANDARD
        H = T + U
        return H

    def hamilton_right_fn(self, t, coords):
        """方程右端项"""
        grad_ham = autograd.grad(self.hamilton_energy_fn)
        grad = grad_ham(coords)
        dq = grad[self.dof:]
        dp = -grad[:self.dof]
        dy = np.asarray([dq, dp]).reshape(-1)
        return dy

    @staticmethod
    def rotate2d(p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ p.reshape(2, 1)).squeeze()

    def random_config(self, num):
        x0_list = []
        for _ in range(num):
            # for n objects evenly distributed around the circle,
            # which means angle(obj_i, obj_{i+1}) = 2*pi/n
            # we made the requirement there that m is the same
            # for every obejct to simplify the formula.
            # But it can be improved.
            nu = 0.5
            min_radius = 1
            max_radius = 5
            system = 'hnn'

            state = np.zeros(self.dof * 2)

            p0 = 2 * np.random.rand(2) - 1
            r = np.random.rand() * (max_radius - min_radius) + min_radius

            theta = 2 * np.pi / self.obj
            p0 *= r / np.sqrt(np.sum((p0 ** 2)))
            for i in range(self.obj):
                state[2 * i: 2 * i + 2] = self.rotate2d(p0, theta=i * theta)

            # # velocity that yields a circular orbit
            dirction = p0 / np.sqrt((p0 * p0).sum())
            v0 = self.rotate2d(dirction, theta=np.pi / 2)
            k = self.k / (2 * r)
            for i in range(self.obj):
                v = v0 * np.sqrt(
                    k * sum([self.m[j % self.obj] / np.sin((j - i) * theta / 2) for j in range(i + 1, self.obj + i)]))
                # make the circular orbits slightly chaotic
                if system == 'hnn':
                    v *= (1 + nu * (2 * np.random.rand(2) - 1))
                else:
                    v *= self.m[i] * (1 + nu * (2 * np.random.rand(2) - 1))
                state[self.dof + 2 * i: self.dof + 2 * i + 2] = self.rotate2d(v, theta=i * theta)

            x0_list.append(state.reshape(-1))
        return np.asarray(x0_list)

    def __init_data(self):
        self.X_train, self.y_train = self.__generate_random(self.train_num, self.h)
        self.X_test, self.y_test = self.__generate_random(self.test_num, self.h)

    def __generate_random(self, num, h):
        x0 = self.random_config(num)
        X = self.__generate(x0, h)
        X = np.concatenate(X, axis=0)
        y = np.asarray(list(map(lambda x: self.hamilton_right_fn(None, x), X)))
        # E = np.array([self.hamilton_energy_fn(y) for y in X])
        return X, y

    def __generate(self, X, h):
        X = np.array(list(map(lambda x: self.solver.solve(x, h), X)))
        return X
