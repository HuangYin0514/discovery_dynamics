# import numpy as np
import autograd
import autograd.numpy as np
import scipy.integrate


class MyDataset:
    def __init__(self, obj, m=None, l=None, **kwargs):
        self.m = m
        self.g = 9.8
        self.l = l
        self.obj = obj
        self.dim = 2
        self.dof = obj * self.dim  # degree of freedom
        self.samples = 1
        self.k = 1

    @staticmethod
    def rotate2d(p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ p.reshape(2, 1)).squeeze()

    def random_config(self, nu=0.5, min_radius=1, max_radius=5, system="modlanet"):
        # for n objects evenly distributed around the circle,
        # which means angle(obj_i, obj_{i+1}) = 2*pi/n
        # we made the requirement there that m is the same
        # for every obejct to simplify the formula.
        # But it can be improved.
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
            if system == 'modlanet':
                v *= (1 + nu * (2 * np.random.rand(2) - 1))
            else:
                v *= self.m[i] * (1 + nu * (2 * np.random.rand(2) - 1))
            state[self.dof + 2 * i: self.dof + 2 * i + 2] = self.rotate2d(v, theta=i * theta)

        return state

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

    def hamilton_energy_fn(self, coords, eng=True):
        """能量函数"""
        assert (len(coords) == self.dof * 2)
        T, U = self.hamiltonian_kinetic(coords), self.hamiltonian_potential(coords)
        # NOT STANDARD
        H = T + U
        return H

    def hamilton_get_ddq(self, coords):
        grad_ham = autograd.grad(self.hamilton_energy_fn)
        grad = grad_ham(coords)
        return grad[self.dof:], -grad[:self.dof]

    def hamilton_right_fn(self, t, coords):
        """方程右端项"""
        dq, dp = self.hamilton_get_ddq(coords)
        dy = np.asarray([dq, dp]).reshape(-1)
        return dy

    def lagrangian_kinetic(self, coords):
        T = 0.
        for i in range(self.obj):
            T = T + 0.5 * self.m[i] * np.sum(coords[self.dof + 2 * i: self.dof + 2 * i + 2] ** 2, axis=0)
        return T

    def lagrangian_potential(self, coords):
        k = self.k
        U = 0.
        for i in range(self.obj):
            for j in range(i):
                U = U - k * self.m[i] * self.m[j] / (
                        (coords[2 * i] - coords[2 * j]) ** 2 +
                        (coords[2 * i + 1] - coords[2 * j + 1]) ** 2) ** 0.5
        return U

    def lagrangian_energy_fn(self, coords, eng=False):
        assert (len(coords) == self.dof * 2)
        T, U = self.lagrangian_kinetic(coords), self.lagrangian_potential(coords)
        L = T - U if not eng else T + U
        return L

    def lagrangian_get_ddq(self, coords):
        grad_lag = autograd.grad(self.lagrangian_energy_fn)
        jaco_lag = autograd.jacobian(grad_lag)
        grad = grad_lag(coords)
        jaco = jaco_lag(coords)
        size = self.dof
        dy = coords[size:]
        ddy = np.linalg.inv(jaco[size:, size:]) @ (grad[:size] - jaco[size:, :size] @ coords[size:])
        return dy, ddy

    def lagrangian_right_fn(self, t, coords):
        size = int(len(coords) / 2)
        dy, ddy = self.lagrangian_get_ddq(coords)
        return np.asarray([dy, ddy]).reshape(-1)

    def get_trajectory(self, t0=0, t_end=10, ode_stepsize=None, y0=None, noise_std=0., system="hnn", **kwargs):
        # get initial state
        self.m = kwargs['m'] if 'm' in kwargs else [1. for i in range(self.obj)]
        self.g = kwargs['g'] if 'g' in kwargs else 9.8
        if y0 is None:
            y0 = self.random_config(system=system)

        if system == "hnn":
            dynamics_t, dynamics_solution = self.runge_kutta_solver(self.hamilton_right_fn, t0, t_end, ode_stepsize, y0)
            dydt = [self.hamilton_right_fn(None, y) for y in dynamics_solution]
            dydt = np.stack(dydt)
            E = np.array([self.hamilton_energy_fn(y) for y in dynamics_solution])

            # add noise
            x = dynamics_solution + np.random.randn(*dynamics_solution.shape) * noise_std
            return x, dydt, dynamics_t, E

        elif system == "lnn":
            """
                为了提速，使用哈密顿系统
            """
            # dynamics_t, dynamics_solution = self.runge_kutta_solver(self.lagrangian_right_fn, t0, t_end, ode_stepsize, y0)
            # dydt = [self.lagrangian_right_fn(None, y) for y in dynamics_solution]
            # dydt = np.stack(dydt)
            # E = np.array([self.lagrangian_energy_fn(y) for y in dynamics_solution])

            dynamics_t, dynamics_solution = self.runge_kutta_solver(self.hamilton_right_fn, t0, t_end, ode_stepsize, y0)
            dydt = [self.hamilton_right_fn(None, y) for y in dynamics_solution]
            dydt = np.stack(dydt)
            E = np.array([self.hamilton_energy_fn(y) for y in dynamics_solution])

            x = dynamics_solution[:, :self.dof]
            v = dydt[:, :self.dof]
            a = dydt[:, self.dof:]
            for i in range(self.obj):
                a[:, self.dim * i:self.dim * (i + 1)] /= self.m[i]

            # add noise
            x += np.random.randn(*x.shape) * noise_std
            v += np.random.randn(*v.shape) * noise_std

            return x, v, a, dynamics_t, E

        else:
            raise ValueError('Unsupported system system, choose'
                             ' system = \'hnn\' or \'modlanet\' instead.')

    def get_dataset(self, seed=0, samples=5, ode_stepsize=0.05, test_split=0.9, system='hnn', **kwargs):
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
