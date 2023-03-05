"""
双摆任务表现为常微分方程组形式

利用autograd，欧拉拉格朗日方程的形式去求解双摆任务

坐标形式(theta1, theta2, dtheta1, dtheta2)
"""
import autograd
import autograd.numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

# constants
l = [1.0, 1.0]
m = [1.0, 1.0]
g = 9.8


def equations(t, coords):
    def La(coords):
        # T = 0.
        # vx, vy = 0., 0.
        # for i in range(2):
        #     vx = vx + l[i] * coords[2 + i] * np.cos(coords[i])
        #     vy = vy + l[i] * coords[2 + i] * np.sin(coords[i])
        #     T = T + 0.5 * m[i] * (np.power(vx, 2) + np.power(vy, 2))
        #
        # U = 0.
        # y = 0.
        # for i in range(2):
        #     y = y - l[i] * np.cos(coords[i])
        #     U = U + m[i] * g * y
        #
        # L = T - U
        # return L
        U, T = 0., 0.
        vx, vy = 0., 0.
        y = 0.
        for i in range(2):
            vx = vx + l[i] * coords[2 + i] * np.cos(coords[i])
            vy = vy + l[i] * coords[2 + i] * np.sin(coords[i])
            T = T + 0.5 * m[i] * (np.power(vx, 2) + np.power(vy, 2))
            y = y - l[i] * np.cos(coords[i])
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


coords = np.array([1, 2, 0, 0])

t_eval = np.linspace(0, 3, num=1000)
sol = solve_ivp(equations, t_span=[0, 3], t_eval=t_eval, y0=coords)

traj = sol.y.T  # (bs,states)

x1 = l[0] * np.sin(traj[:, 0])
y1 = -l[0] * np.cos(traj[:, 0])
x2 = x1 + l[1] * np.sin(traj[:, 1])
y2 = y1 - l[1] * np.cos(traj[:, 1])

# def polar2xy(x):
#     pos = np.zeros([x.shape[0], x.shape[1] * 2])
#     for i in range(x.shape[1]):
#         if i == 0:
#             pos[:, 2 * i:2 * (i + 1)] += np.concatenate([np.sin(x[:, i:i + 1]), -np.cos(x[:, i:i + 1])], 1)
#         else:
#             pos[:, 2 * i:2 * (i + 1)] += pos[:, 2 * (i - 1):2 * i] + np.concatenate(
#                 [np.sin(x[:, i:i + 1]), -np.cos(x[:, i:i + 1])], 1)
#     return pos
# x1 = polar2xy(traj)[:, 0]
# y1 = polar2xy(traj)[:, 1]
# x2 = polar2xy(traj)[:, 2]
# y2 = polar2xy(traj)[:, 3]

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
