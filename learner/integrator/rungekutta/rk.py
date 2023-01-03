# -*- coding: utf-8 -*-
import abc

import numpy as np
from scipy.integrate import solve_ivp


class RK(abc.ABC):
    '''Runge-Kutta method. '''

    def __init__(self):
        pass

    @abc.abstractmethod
    def solver(self, y0, h):
        pass

    def solve(self, y0, h):
        t, y = self.solver(y0, h)
        return t,y


class RK4(RK):

    def __init__(self, f, t0, t_end):
        super(RK4, self).__init__()
        self.f = f
        self.t0 = t0
        self.t_end = t_end
        # print("Waring! you using the RK4 solver!!!!!!!!!!!!!!!!")

    def solver(self, y0, h):
        num = len(y0)
        t = np.arange(self.t0, self.t_end, h)
        x = np.zeros((t.size, num))
        x[0, :] = y0
        for i in range(t.size - 1):
            s0 = self.f(t[i], x[i, :])
            s1 = self.f(t[i] + h / 2., x[i, :] + h * s0 / 2.)
            s2 = self.f(t[i] + h / 2., x[i, :] + h * s1 / 2.)
            s3 = self.f(t[i + 1], x[i, :] + h * s2)
            x[i + 1, :] = x[i, :] + h * (s0 + 2 * (s1 + s2) + s3) / 6.
        return t, x


class RK45(RK):

    def __init__(self, f, t0, t_end):
        super(RK45, self).__init__()
        self.f = f
        self.t0 = t0
        self.t_end = t_end

    def solver(self, y0, h):
        t = np.arange(self.t0, self.t_end, h)
        solve_result = solve_ivp(fun=self.f, t_span=[self.t0, self.t_end], y0=y0, t_eval=t, rtol=1e-12, method='RK45')
        x = solve_result['y'].T
        return t, x
