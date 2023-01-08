# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/7 10:34 PM
@desc:
"""

import torch


def square_err_fn(x, y):
    return ((x - y) ** 2).mean((-1, -2))


def rel_err_fn(x, y):
    #  This function is  sqrt（a-b）** 2/ (a**2 + b**2)
    square_err = square_err_fn(x, y)
    rel_err = torch.sqrt(square_err) / \
              (torch.sqrt((x ** 2).mean((-1, -2))) + torch.sqrt((y ** 2).mean((-1, -2))))
    loggeomean_rel_err = torch.log(torch.clamp(rel_err, min=1e-7))
    return loggeomean_rel_err


def energy_err_fn(x, y, energy_function):
    err_list = []
    for x_, y_ in zip(x, y):
        eng_x = torch.stack([energy_function(i) for i in x_])
        eng_y = torch.stack([energy_function(i) for i in y_])
        H_err = torch.abs(eng_x - eng_y) / (torch.abs(eng_x) + torch.abs(eng_y))
        log_H_err = torch.log(torch.clamp(H_err, min=1e-7))
        err_list.append(log_H_err)
        # from matplotlib import pyplot as plt
        # plt.plot(H_err.detach().numpy())
        # plt.plot(log_H_err.detach().numpy())
        # plt.show()
    return torch.stack(err_list)


def accuracy_fn(output_traj, target_traj, energy_function):
    """
     Args:
         output_traj: output of the model (bs,T,states)
         target_traj: ground truth
         energy_function: energy function

     Returns:
         accuracy
     """
    mse_err = square_err_fn(output_traj, target_traj).mean()
    rel_err = rel_err_fn(output_traj, target_traj).mean()
    eng_err = energy_err_fn(output_traj, target_traj, energy_function).mean()
    return mse_err, torch.exp(rel_err), torch.exp(eng_err)
