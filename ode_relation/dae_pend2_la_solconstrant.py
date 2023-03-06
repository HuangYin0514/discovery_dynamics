"""
双摆任务表现为微分代数方程形式

利用autograd的形式去求解双摆任务
方法利用等式关系，求解拉格朗日乘子lambda
ref：

坐标形式(x1,y1,x2,y2)
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
    # dx1, dy1, dx1, dy2, x1, y1, x2, y2
    dx, x = np.split(coords, 2)

    M = np.array([[m[0], 0, 0, 0],
                  [0, m[0], 0, 0],
                  [0, 0, m[1], 0],
                  [0, 0, 0, m[1]]
                  ])

    def la_V(x):
        U = 0.
        y = 0.
        for i in range(2):
            y = x[i * 2 + 1]
            U = U + m[i] * g * y
        return U

    def phi(x):
        return np.array([
            x[0] ** 2 + x[1] ** 2 - l[0] ** 2,
            (x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2 - l[1] ** 2,
        ]).reshape(-1)

    def phi_plus_q(x, dx, D_phi_q_fun):
        return D_phi_q_fun(x) @ dx

    D_phi_q_fun = autograd.jacobian(phi)
    D_phi_qq_fun = autograd.jacobian(phi_plus_q, argnum=0)
    phi_q = D_phi_q_fun(x)  # (2, 4)
    phi_qq = D_phi_qq_fun(x, dx, D_phi_q_fun)

    Minv = np.linalg.inv(M)  # (4, 4)
    F = -autograd.jacobian(la_V)(x).reshape(-1, 1)  # (4, 1)

    # 求解 lam ----------------------------------------------------------------
    L = phi_q @ Minv @ phi_q.T  # (2, 2)
    R = phi_q @ Minv @ F + phi_qq @ dx.reshape(-1, 1)  # (2, 1)
    lam = np.linalg.inv(L) @ R  # (2, 1)

    # 求解 vdot ----------------------------------------------------------------
    vdot_R = F - phi_q.T @ lam  # (4, 1)
    vdot = Minv @ vdot_R  # (4, 1)

    return np.concatenate([vdot, dx.reshape(-1, 1)], axis=0).reshape(-1)


coords = np.array([0, 0, 0, 0, l[0], 0, l[0] + l[1], 0])
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
