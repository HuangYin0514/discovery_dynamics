import autograd
import autograd.numpy as np

import learner as ln
from learner.integrator.rungekutta import RK4,RK45


class PendulumData(ln.Data):
    def __init__(self, obj, dim, train_num, test_num, m=None, l=None, **kwargs):
        super(PendulumData, self).__init__()

        self.train_num = train_num
        self.test_num = test_num

        self.m = m
        self.l = l
        self.g = 9.8

        self.space = [0, 2 * np.pi, -1, 1]  # value range of q and p

        self.obj = obj
        self.dim = dim
        self.dof = obj * self.dim  # degree of freedom

        t0 = 0
        t_end = 10
        self.h = 0.01
        self.solver = RK4(self.hamilton_right_fn, t0=t0, t_end=t_end)

        self.__init_data()

    # def hamilton_right_fn(self, t, coords):
    #     grad_ham = autograd.grad(self.hamilton_energy_fn)
    #     grad = grad_ham(coords)
    #     q, p = grad[self.dof:], -grad[:self.dof]
    #     return np.asarray([q, p]).reshape(-1)

    def hamilton_right_fn(self, t, coords):
        """获取导数"""
        q1, q2, p1, p2 = coords
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        g = self.g
        b = l1 * l2 * (m1 + m2 * np.sin(q1 - q2) ** 2)
        dq1 = (l2 * p1 - l1 * p2 * np.cos(q1 - q2)) / (b * l1)
        dq2 = (-m2 * l2 * p1 * np.cos(q1 - q2) + (m1 + m2) * l1 * p2) / (m2 * b * l2)
        h1 = p1 * p2 * np.sin(q1 - q2) / b
        h2 = (m2 * l2 ** 2 * p1 ** 2 + (m1 + m2) * l1 ** 2 * p2 ** 2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / (
                2 * b ** 2)
        dp1 = -(m1 + m2) * g * l1 * np.sin(q1) - h1 + h2 * np.sin(2 * (q1 - q2))
        dp2 = -m2 * g * l2 * np.sin(q2) + h1 - h2 * np.sin(2 * (q1 - q2))
        return np.asarray([ dq1, dq2, dp1, dp2]).reshape(-1)

    def hamiltonian_kinetic(self, coords):
        assert (len(coords) == self.dof * 2)
        T = 0.
        vx, vy = 0., 0.
        for i in range(self.obj):
            vx = vx + self.l[i] * coords[self.dof + i] * np.cos(coords[i])
            vy = vy + self.l[i] * coords[self.dof + i] * np.sin(coords[i])
            T = T + 0.5 * self.m[i] * (np.power(vx, 2) + np.power(vy, 2))
        return T

    def hamiltonian_potential(self, coords):
        assert (len(coords) == self.dof * 2)
        g = self.g
        U = 0.
        y = 0.
        for i in range(self.obj):
            y = y - self.l[i] * np.cos(coords[i])
            U = U + self.m[i] * g * y
        return U

    def hamilton_energy_fn(self, coords):
        """能量函数"""
        # From "The double pendulum: Hamiltonian formulation"
        # https://diego.assencio.com/?index=e5ac36fcb129ce95a61f8e8ce0572dbf
        # H = self.hamiltonian_kinetic(coords) + self.hamiltonian_potential(coords)  # some error in this implementation
        q1, q2, p1, p2 = np.split(coords, 4)  # q is angle, p is angular momentum.
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        H = (m1 + m2) * self.g * l1 * (-np.cos(q1)) + m2 * self.g * l2 * (-np.cos(q2)) \
            + ((m1 + m2) * l1 ** 2 * p2 ** 2 + m2 * l2 ** 2 * p1 ** 2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / \
            (2 * m2 * (l1 ** 2) * (l2 ** 2) * (m1 + m2 * np.sin(q1 - q2) ** 2))
        return H

    # def random_config(self, space, num):
    #     qmin, qmax, pmin, pmax = space[0], space[1], space[2], space[3]
    #     x0 = np.zeros((num, self.obj * 2))
    #     for i in range(self.obj):
    #         q = np.random.rand(num) * (qmax - qmin) + qmin
    #         p = np.random.rand(num) * (pmax - pmin) + pmin
    #         x0[..., i] = q
    #         x0[..., i + self.obj] = p
    #     return x0
    def random_config(self, space, num):
        y0_list = []
        for i in range(num):
            max_momentum = 1.
            y0 = np.zeros(self.obj * 2)
            for i in range(self.obj):
                theta = (2 * np.pi - 0) * np.random.rand() + 0
                momentum = (2 * np.random.rand() - 1) * max_momentum
                y0[i] = theta
                y0[i + self.obj] = momentum
            y0_list.append(y0.reshape(-1))
        return y0_list

    def __init_data(self):
        self.X_train, self.y_train = self.__generate_random(self.space, self.train_num, self.h)
        self.X_test, self.y_test = self.__generate_random(self.space, self.test_num, self.h)

    def __generate_random(self, space, num, h):
        x0 = self.random_config(space, num)
        X = self.__generate(x0, h)
        X = np.concatenate(X)
        # y = list(map(lambda x: self.hamilton_right_fn(None, x), X))
        # y = np.asarray(y)
        dydt = [self.hamilton_right_fn(None, y) for y in X]
        y = np.stack(dydt)
        # E = np.array([self.hamilton_energy_fn(y) for y in X])
        return X, y

    def __generate(self, X, h):
        X = np.array(list(map(lambda x: self.solver.solve(x, h), X)))
        return X
