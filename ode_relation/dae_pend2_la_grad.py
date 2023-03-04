"""
双摆任务表现为微分代数方程形式

利用解析的形式去求解双摆任务
faiq、faiqq 利用自动求导给出
"""
import autograd.numpy as np
from autograd import jacobian
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# constants
l1 = 10.0
l2 = 10.0
m1 = 1.0
m2 = 1.0
g = 10

x1 = l1
x2 = 1
y1 = l1 + l2
y2 = 2
dx1, dx2, dy1, dy2 = 1, 1, 1, 1
lam1, lam2 = 1, 1


def equations(t, coords):
    dx1, dx2, dy1, dy2, x1, x2, y1, y2, lam1, lam2 = coords

    dx = np.array([dx1, dx2, dy1, dy2])

    M = np.array([[m1, 0, 0, 0],
                  [0, m1, 0, 0],
                  [0, 0, m2, 0],
                  [0, 0, 0, m2]
                  ])

    F = np.array([[0],
                  [-m1 * g],
                  [0],
                  [-m2 * g]
                  ])

    def phi(coords):
        return np.array([
            coords[4] ** 2 + coords[5] ** 2 - l1 ** 2,
            (coords[4] - coords[6]) ** 2 + (coords[5] - coords[7]) ** 2 - l1 ** 2,
        ]).reshape(-1)

    def phi_plus_q(coords):
        return jacobian(phi)(coords)[:, 4: 8] @ coords[0:4]

    faiq = jacobian(phi)(coords)[:, 4: 8]
    faiqq = jacobian(phi_plus_q)(coords)[:, 4: 8]

    L = np.block([[M, np.zeros((4, 4)), faiq.T],
                  [np.zeros((4, 4)), np.diag(np.ones(4)), np.zeros((4, 2))],
                  [faiq, np.zeros((2, 4)), np.zeros((2, 2))]])

    R = np.concatenate([F, dx.reshape(-1, 1), (-faiqq @ dx.reshape(-1, 1)).reshape(-1, 1)], axis=0)

    L_inv = np.linalg.pinv(L)

    return (L_inv @ R).reshape(-1)


coords = np.array([0, 0, 0, 0, l1, 0, l1 + l2, 0, 1, 1])
t_eval = np.linspace(0, 10, num=1000)

sol = solve_ivp(equations, t_span=[0, 10], t_eval=t_eval, y0=coords,method='RK23')

x1 = sol.y[4]
y1 = sol.y[5]
x2 = sol.y[6]
y2 = sol.y[7]

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(x1, y1, 'b', label='m1')
ax.plot(x2, y2, 'r', label='m2')
ax.set_xlim(-22, 22)
ax.set_ylim(-22, 22)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()
