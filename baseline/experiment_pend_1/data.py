import numpy as np


class Dataset:
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

    def runge_kutta4_solver(self, rightf, t0, t_end, t_step_size, x0):
        """ 龙格库塔方法
        :param rightf: 方程右端项
        :param t0: 初始时间
        :param t_end: 末端时间
        :param t_step_size: 时间步长
        :param x0: 初始条件 shape->(N,)
        :return: t->时间，x->方程解
        """
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

    def lagrange_get_ddq(self, coords, g=9.8):
        """获取导数"""
        q, p = np.split(coords, 2)
        ddq = np.zeros_like(q)
        ddq[0] = -3 / 2 * np.sin(q)
        return ddq

    def lagrange_right_fn(self, t, coords):
        """方程右端项"""
        q, p = np.split(coords, 2)
        dq = p
        ddq = self.lagrange_get_ddq(coords)
        dy = np.concatenate([dq, ddq], axis=0).reshape(-1)
        return dy

    def lagrange_energy_fn(self, coords):
        """能量函数"""
        q, p = np.split(coords, 2)
        H = 3 * (1 - np.cos(q)) + p ** 2
        return H

    def hamilton_get_ddq(self, coords, g=9.8):
        """获取导数"""
        # 获得初始化位置
        q, p = np.split(coords, 2)
        # 计算2阶导数
        dy = np.zeros_like(coords)
        dq = 2 * p
        dp = -3 * np.sin(q)
        return dq, dp

    def hamilton_right_fn(self, t, coords):
        """方程右端项"""
        q, p = np.split(coords, 2)
        dq, dp = self.hamilton_get_ddq(coords)
        dy = np.concatenate([dq, dp], axis=0).reshape(-1)
        return dy

    def hamilton_energy_fn(self, coords):
        """能量函数"""
        q, p = np.split(coords, 2)
        # q, p = coords
        H = 3 * (1 - np.cos(q)) + p ** 2
        return H

    def get_trajectory(self, t0=0, t_end=3, ode_stepsize=None, y0=None, noise_std=0., system="modlanet", **kwargs):
        # get initial state
        self.m = kwargs['m'] if 'm' in kwargs else [1. for i in range(self.obj)]
        self.l = kwargs['l'] if 'l' in kwargs else [1. for i in range(self.obj)]
        self.g = kwargs['g'] if 'g' in kwargs else 9.8
        if y0 is None:
            y0 = self.random_config(system)

        if system == "hnn":
            dynamics_t, dynamics_solution = self.runge_kutta4_solver(self.hamilton_right_fn, t0, t_end, ode_stepsize, y0)
            dydt = [self.hamilton_right_fn(None, y) for y in dynamics_solution]
            dydt = np.stack(dydt)
            E = np.array([self.hamilton_energy_fn(y) for y in dynamics_solution])

            # add noise
            x = dynamics_solution + np.random.randn(*dynamics_solution.shape) * noise_std
            return x, dydt, dynamics_t, E

        elif system == "lnn":

            dynamics_t, dynamics_solution = self.runge_kutta4_solver(self.lagrange_right_fn, t0, t_end, ode_stepsize, y0)
            dydt = [self.lagrange_right_fn(None, y) for y in dynamics_solution]
            dydt = np.stack(dydt)
            E = np.array([self.lagrange_energy_fn(y) for y in dynamics_solution])

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

    def get_dataset(self, seed=0, samples=100, ode_stepsize=0.06, test_split=0.9, system='modlanet', **kwargs):
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
            data['E'] = np.concatenate(ts).reshape(-1, 1)

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
            data['E'] = np.concatenate(ts).reshape(-1, 1)

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
