import numpy as np
import scipy.integrate


class MyDataset:
    def __init__(self, obj, m=None, l=None, **kwargs):
        self.m = m
        self.g = 9.8
        self.l = l
        self.obj = obj
        self.dim = 1
        self.dof = obj * self.dim  # degree of freedom
        self.samples = 1

    def random_config(self, system='hnn'):
        max_momentum = 1.
        y0 = np.zeros(self.obj * 2)
        for i in range(self.obj):
            theta = (2 * np.pi - 0) * np.random.rand() + 0
            momentum = (2 * np.random.rand() - 1) * max_momentum
            y0[i] = theta
            y0[i + self.obj] = momentum
        return y0.reshape(-1)

    def runge_kutta_solver(self, rightf, t0, t_end, t_step_size, x0, solve_method='solve_ivp'):
        """ 龙格库塔方法
        :param rightf: 方程右端项
        :param t0: 初始时间
        :param t_end: 末端时间
        :param t_step_size: 时间步长
        :param x0: 初始条件 shape->(N,)
        :return: t->时间，x->方程解
        """
        t, x = None, None
        if solve_method == 'solve_ivp':
            t = np.arange(t0, t_end, t_step_size)
            solve_result = scipy.integrate.solve_ivp(fun=rightf, t_span=[t0, t_end], y0=x0, t_eval=t, rtol=1e-12)
            x = solve_result['y'].T
        elif solve_method == 'rk4':
            num = len(x0)
            t = np.arange(t0, t_end, t_step_size)
            x = np.zeros((t.size, num))
            x[0, :] = x0
            for i in range(t.size - 1):
                s0 = rightf(t[i], x[i, :])
                s1 = rightf(t[i] + t_step_size / 2., x[i, :] + t_step_size * s0 / 2.)
                s2 = rightf(t[i] + t_step_size / 2., x[i, :] + t_step_size * s1 / 2.)
                s3 = rightf(t[i + 1], x[i, :] + t_step_size * s2)
                x[i + 1, :] = x[i, :] + t_step_size * (s0 + 2 * (s1 + s2) + s3) / 6.
        return t, x

    def hamilton_get_ddq(self, coords):
        """获取导数"""
        q1, q2, p1, p2 = coords
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        g = self.g
        b = l1 * l2 * (m1 + m2 * np.sin(q1 - q2) ** 2)
        dq1 = (l2 * p1 - l1 * p2 * np.cos(q1 - q2)) / (b * l1)
        dq2 = (-m2 * l2 * p1 * np.cos(q1 - q2) + (m1 + m2) * l1 * p2) / (m2 * b * l2)
        h1 = p1 * p2 * np.sin(q1 - q2) / b
        h2 = (m2 * l2 ** 2 * p1 ** 2 + (m1 + m2) * l1 ** 2 * p2 ** 2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / (2 * b ** 2)
        dp1 = -(m1 + m2) * g * l1 * np.sin(q1) - h1 + h2 * np.sin(2 * (q1 - q2))
        dp2 = -m2 * g * l2 * np.sin(q2) + h1 - h2 * np.sin(2 * (q1 - q2))
        return dq1, dq2, dp1, dp2

    def hamilton_right_fn(self, t, coords):
        """方程右端项"""
        dq1, dq2, dp1, dp2 = self.hamilton_get_ddq(coords)
        dy = np.asarray([dq1, dq2, dp1, dp2]).reshape(-1)
        return dy

    def hamilton_energy_fn(self, coords):
        """能量函数"""
        q1, q2, p1, p2 = np.split(coords, 4)  # q is angle, p is angular momentum.
        l1, l2, m1, m2 = self.l[0], self.l[1], self.m[0], self.m[1]
        H = (m1 + m2) * self.g * l1 * (-np.cos(q1)) + m2 * self.g * l2 * (-np.cos(q2)) \
            + ((m1 + m2) * l1 ** 2 * p2 ** 2 + m2 * l2 ** 2 * p1 ** 2 - 2 * m2 * l1 * l2 * p1 * p2 * np.cos(q1 - q2)) / \
            (2 * m2 * (l1 ** 2) * (l2 ** 2) * (m1 + m2 * np.sin(q1 - q2) ** 2))
        # double pendulum hamiltonian
        return H

    def lagrangian_get_ddq(self, coords):
        th1, th2, dth1, dth2 = coords
        A = np.asarray([
            [self.l[0] ** 2 * (self.m[0] + self.m[1]), self.l[0] * self.l[1] * self.m[1] * np.cos(th1 - th2)],
            [self.l[0] * self.l[1] * self.m[1] * np.cos(th1 - th2), self.l[1] ** 2 * self.m[1]]
        ])
        A = np.linalg.inv(A)
        B = np.asarray([
            [- self.l[0] * (self.l[1] * self.m[1] * np.sin(th1 - th2) * dth1 * dth2 + self.g * self.m[0] * np.sin(th1) + +self.g * self.m[1] * np.sin(th1)),
             self.l[1] * self.m[1] * (self.l[0] * np.sin(th1 - th2) * dth1 * dth2 - self.g * np.sin(th2))]])
        C = np.asarray([
            [-1 * self.l[0] * self.l[1] * self.m[1] * np.sin(th1 - th2) * dth2, self.l[0] * self.l[1] * self.m[1] * np.sin(th1 - th2) * dth2],
            [-1 * self.l[0] * self.l[1] * self.m[1] * np.sin(th1 - th2) * dth1, self.l[0] * self.l[1] * self.m[1] * np.sin(th1 - th2) * dth1]
        ])
        dy = np.asarray([dth1, dth2]).reshape(-1, 1)
        ddy = A @ (B.T - C @ dy)
        return dy, ddy

    def lagrangian_right_fn(self, t, coords):
        size = int(len(coords) / 2)
        dy, ddy = self.lagrangian_get_ddq(coords)
        return np.asarray([dy, ddy]).reshape(-1)

    def lagrangian_energy_fn(self, coords, eng=True):
        assert (len(coords) == self.dof * 2)
        g = self.g
        U, T = 0., 0.
        vx, vy = 0., 0.
        y = 0.
        for i in range(self.obj):
            vx = vx + self.l[i] * coords[self.dof + i] * np.cos(coords[i])
            vy = vy + self.l[i] * coords[self.dof + i] * np.sin(coords[i])
            T = T + 0.5 * self.m[i] * (np.power(vx, 2) + np.power(vy, 2))
            y = y - self.l[i] * np.cos(coords[i])
            U = U + self.m[i] * g * y
        L = T - U if not eng else T + U
        return L

    def position_transformation_L2H(self, coords):
        """
        将拉格朗日坐标dth转换为哈密顿坐标p
        """
        th1, th2, dth1, dth2 = coords
        dq1 = self.l[0] * (self.l[0] * self.m[0] * dth1 + self.m[1] * (self.l[0] * dth1 + self.l[1] * np.cos(th1 - th2) * dth2))
        dq2 = self.l[1] * self.m[1] * (self.l[0] * np.cos(th1 - th2) * dth1 + self.l[1] * dth2)
        return np.asarray([th1, th2, dq1, dq2]).reshape(-1)

    def position_transformation_H2L(self, coords):
        """
        将哈密顿坐标p转换为拉格朗日坐标dth
        """
        th1, th2, p1, p2 = coords
        denominator1 = self.l[0] ** 2 * self.l[1] * (self.m[0] + self.m[1] * np.sin(th1 - th2) ** 2)
        denominator2 = self.m[1] * self.l[0] * self.l[1] ** 2 * (self.m[0] + self.m[1] * np.sin(th1 - th2) ** 2)
        dth1 = (self.l[1] * p1 - self.l[0] * p2 * np.cos(th1 - th2)) / denominator1
        dth2 = (-1 * self.m[1] * self.l[1] * p1 * np.cos(th1 - th2) + (self.m[0] + self.m[1]) * self.l[0] * p2) / denominator2
        return np.asarray([th1, th2, dth1, dth2]).reshape(-1)

    def get_trajectory(self, t0=0, t_end=10, ode_stepsize=None, y0=None, noise_std=0., system="hnn", **kwargs):
        # get initial state
        self.m = kwargs['m'] if 'm' in kwargs else [1. for i in range(self.obj)]
        self.l = kwargs['l'] if 'l' in kwargs else [1. for i in range(self.obj)]
        self.g = kwargs['g'] if 'g' in kwargs else 9.8
        if y0 is None:
            y0 = self.random_config(system)

        if system == "hnn":
            dynamics_t, dynamics_solution = self.runge_kutta_solver(self.hamilton_right_fn, t0, t_end, ode_stepsize, y0)
            dydt = [self.hamilton_right_fn(None, y) for y in dynamics_solution]
            dydt = np.stack(dydt)
            E = np.array([self.hamilton_energy_fn(y) for y in dynamics_solution])

            # add noise
            x = dynamics_solution + np.random.randn(*dynamics_solution.shape) * noise_std
            return x, dydt, dynamics_t, E

        elif system == "lnn":

            dynamics_t, dynamics_solution = self.runge_kutta_solver(self.lagrangian_right_fn, t0, t_end, ode_stepsize, y0)
            dydt = [self.lagrangian_right_fn(None, y) for y in dynamics_solution]
            dydt = np.stack(dydt)
            E = np.array([self.lagrangian_energy_fn(y) for y in dynamics_solution])

            x = dynamics_solution[:, :self.dof]
            v = dydt[:, :self.dof]
            a = dydt[:, self.dof:]

            # add noise
            x += np.random.randn(*x.shape) * noise_std
            v += np.random.randn(*v.shape) * noise_std

            return x, v, a, dynamics_t, E

        else:
            raise ValueError('Unsupported system system, choose'
                             ' system = \'hnn\' or \'modlanet\' instead.')

    def get_dataset(self, seed=0, samples=100, ode_stepsize=0.1, test_split=0.9, system='hnn', **kwargs):
        data = {'meta': locals()}
        self.samples = samples

        if system == 'hnn':
            # randomly sample inputs
            np.random.seed(seed)
            xs, dxs, ts, Es = [], [], [], []
            for s in range(self.samples):
                x, dx, t, E = self.get_trajectory(system=system, ode_stepsize=ode_stepsize, **kwargs)
                xs.append(x)
                dxs.append(dx)
                ts.append(t)
                Es.append(E)

            data['x'] = np.concatenate(xs)
            data['dx'] = np.concatenate(dxs)
            data['t'] = np.concatenate(ts).reshape(-1, 1)
            data['E'] = np.concatenate(Es).reshape(-1, 1)

            # make a train/test split
            split_ix = int(len(data['x']) * test_split)
            split_data = {}
            for k in ['x', 'dx', 't', 'E']:
                split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
            data = split_data

        elif system == 'lnn':
            # randomly sample inputs
            xs, vs, acs, ts, Es = [], [], [], [], []
            for s in range(samples):
                x, v, ac, t, E = self.get_trajectory(system=system, ode_stepsize=ode_stepsize, **kwargs)
                xs.append(x)
                vs.append(v)
                acs.append(ac)
                ts.append(t)
                Es.append(E)

            data['x'] = np.concatenate(xs)
            data['v'] = np.concatenate(vs)
            data['ac'] = np.concatenate(acs)
            data['t'] = np.concatenate(ts).reshape(-1, 1)
            data['E'] = np.concatenate(Es).reshape(-1, 1)

            # make a train/test split
            split_ix = int((data['x'].shape[0]) * test_split)
            split_data = {}
            for k in ['x', 'v', 'ac', 't', 'E']:
                split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
            data = split_data

        else:
            raise ValueError('Unsupported dynamic system, choose'
                             'system = \'hnn\' or \'modlanet\' instead.')
        return data
