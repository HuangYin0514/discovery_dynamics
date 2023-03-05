"""
双摆任务表现为微分代数方程形式

利用解析的形式去求解双摆任务
faiq、faiqq给出了具体的表达式

坐标形式(x1,y1,x2,y2)
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# constants
l1 = 10.0
l2 = 10.0
m1 = 10.0
m2 = 10.0
g = 10.0


def equations(t, coords):
    dx1, dx2, dy1, dy2, x1, x2, y1, y2, lam1, lam2 = coords

    M = np.array([[m1, 0, 0, 0],
                  [0, m1, 0, 0],
                  [0, 0, m2, 0],
                  [0, 0, 0, m2]
                  ])

    F = np.array([[0],
                  [m1 * g],
                  [0],
                  [m2 * g]
                  ])

    faiq = np.array([
        [2 * coords[4], 2 * coords[5], 0, 0],
        [2 * (coords[4] - coords[6]), 2 * (coords[5] - coords[7]), 2 * (coords[6] - coords[4]),
         2 * (coords[7] - coords[5])]
    ])
    faiqq = np.array([
        [2 * coords[0], 2 * coords[1], 0, 0],
        [2 * (coords[0] - coords[2]), 2 * (coords[1] - coords[3]), 2 * (coords[2] - coords[0]),
         2 * (coords[3] - coords[1])]
    ])

    L = np.block([[M, np.zeros((4, 4)), faiq.T],
                  [faiq, np.zeros((2, 4)), np.zeros((2, 2))],
                  [np.zeros((4, 4)), np.diag(np.ones(4)), np.zeros((4, 2))]])

    R = np.concatenate([F, (-faiqq @ coords[0:4].reshape(-1, 1)).reshape(-1, 1), coords[0:4].reshape(-1, 1)], axis=0)

    L_inv = np.linalg.inv(L)

    return (L_inv @ R).reshape(-1)


coords = np.array([0, 0, 0, 0, l1, 0, l1 + l2, 0, 1, 1])
t0 = 0.
t_end = 10.
dt=0.005
_time_step = int((t_end - t0) / dt)
t = np.linspace(t0, t_end, _time_step)
sol = solve_ivp(equations, t_span=[0, t_end], t_eval=t, y0=coords,method='RK23')

x1 = sol.y[4]
y1 = sol.y[5]
x2 = sol.y[6]
y2 = sol.y[7]

fig, ax = plt.subplots(figsize=(5, 5))

ax.plot(x1, -y1, 'b', label='m1')
ax.plot(x2, -y2, 'r', label='m2')
ax.set_xlim(-22, 22)
ax.set_ylim(-22, 22)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
ax.legend()
plt.show()
