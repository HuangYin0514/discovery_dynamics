"""
双摆任务表现为微分代数方程形式

利用autograd的形式去求解双摆任务
方法利用等式关系，求解拉格朗日乘子lambda
ref：

坐标形式(x1,y1,x2,y2)
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

l = [10.0, 10.0]
m = [10.0, 10.0]
g = 10


def equations(t, coords):
    dx1, dx2, dy1, dy2, x1, x2, y1, y2 = coords

    # dx = np.array([dx1, dx2, dy1, dy2])

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

    phi_q = jacobian(phi)(coords)[:, 4: 8]  # (2, 4)
    phi_qq = jacobian(phi_plus_q)(coords)[:, 4: 8]

    Minv = np.linalg.inv(M)  # (4, 4)

    # 求解 lam
    L = phi_q @ Minv @ phi_q.T  # (2, 2)
    R = phi_q @ Minv @ F + phi_qq @ coords[0: 4].reshape(-1, 1)  # (2, 1)
    lam = np.linalg.inv(L) @ R

    # 求解 vdot
    vdot_L = M
    vdot_R = F - phi_q.T @ lam
    vdot = np.linalg.inv(vdot_L) @ vdot_R

    return np.concatenate([vdot, coords[0: 4].reshape(-1, 1)], axis=0).reshape(-1)


coords = np.array([0, 0, 0, 0, l1, 0, l1 + l2, 0])
t_eval = np.linspace(0, 10, num=10000)

sol = solve_ivp(equations, t_span=[0, 10], t_eval=t_eval, y0=coords, method='RK23')

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
