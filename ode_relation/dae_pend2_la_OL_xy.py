"""
双摆任务表现为常微分方程组形式

利用autograd，欧拉拉格朗日方程的形式去求解双摆任务

坐标形式(x1, y1, x2, y2, dx1, dy1, dx2, dy2)
"""
import autograd
import autograd.numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# constants
l = [10.0, 10.0]
m = [10.0, 10.0]
g = 10


def equations(t, coords):
    def La(coords):
        x1, y1, x2, y2, dx1, dy1, dx2, dy2 = coords
        x = np.array([x1, y1, x2, y2])
        dx = np.array([dx1, dy1, dx2, dy2])
        T = 0.
        for i in range(2):
            T = T + 0.5 * m[i] * np.sum(dx[2 * i: 2 * i + 2] ** 2)

        U = 0.
        y = 0.
        for i in range(2):
            y = x[i + 1]
            U = U + m[i] * g * y

        L = T - U
        return L

    grad_lag = autograd.grad(La)
    jaco_lag = autograd.jacobian(grad_lag)
    grad = grad_lag(coords)
    jaco = jaco_lag(coords)
    size = int(len(coords) / 2)
    a = np.linalg.inv(jaco[size:, size:]) @ (grad[:size] - jaco[size:, :size] @ coords[size:])
    return np.append(coords[size:], a)


coords = np.array([l[0], 0, l[0] + l[1], 0, 0, 0, 0, 0])
t_eval = np.linspace(0, 10, num=10000)

sol = solve_ivp(equations, t_span=[0, 10], t_eval=t_eval, y0=coords)

x1 = sol.y[0]
y1 = sol.y[1]
x2 = sol.y[2]
y2 = sol.y[3]

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x1, y1, 'b', label='m1')
ax.plot(x2, y2, 'r', label='m2')
axis_max = 1.2
ax.set_xlim(-(l[0] + l[1]) * axis_max, (l[0] + l[1]) * axis_max)
ax.set_ylim(-(l[0] + l[1]) * axis_max, (l[0] + l[1]) * axis_max)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()
