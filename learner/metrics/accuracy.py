# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/7 10:34 PM
@desc:
"""
import torch


def position_MSE_err_fn(x, y):
    bs, times, states = x.shape
    dof = int(states // 2)

    err_list = []
    for x_, y_ in zip(x, y):
        x_position = x_[..., :dof]
        y_position = y_[..., :dof]

        error_norm = (x_position - y_position).norm()
        err_list.append(error_norm)

    position_err = torch.stack(err_list)
    return position_err


def penergy_MSE_err_fn(x, y, energy_function):
    err_list = []
    for x_, y_ in zip(x, y):
        eng_x = energy_function(x_).reshape(-1, 1)
        eng_y = energy_function(y_).reshape(-1, 1)
        # eng_y = eng_y[0].repeat(len(eng_y)) # 与真实的eng对比

        rel_err = (eng_x - eng_y).norm()
        print(eng_x.shape, eng_y.shape)
        err_list.append(rel_err)

    E_err = torch.stack(err_list)
    return E_err


def position_BE_err_fn(x, y):
    bs, times, states = x.shape
    dof = int(states // 2)

    err_list = []
    for x_, y_ in zip(x, y):
        x_position = x_[..., :dof]
        y_position = y_[..., :dof]

        rel_err = (x_position - y_position).norm(dim=1) / (x_position.norm(dim=1) + y_position.norm(dim=1))
        # rel_err = (x_position - y_position).norm() / (x_position.norm() + y_position.norm())

        err_list.append(rel_err)

    err = torch.stack(err_list)
    return err


def energy_BE_err_fn(x, y, energy_function):
    err_list = []
    for x_, y_ in zip(x, y):
        eng_x = energy_function(x_).reshape(-1, 1)
        eng_y = energy_function(y_).reshape(-1, 1)
        # eng_y = eng_y[0].repeat(len(eng_y)) # 与真实的eng对比

        # rel_err = (eng_x - eng_y).norm() / (eng_x.norm() + eng_y.norm())
        rel_err = (eng_x - eng_y).norm(dim=1) / (eng_x.norm(dim=1) + eng_y.norm(dim=1))
        err_list.append(rel_err)

    E_err = torch.stack(err_list)
    return E_err


def accuracy_fn(output_traj, target_traj, energy_function):
    """
     Args:
         output_traj: output of the model (bs,T,states)
         target_traj: ground truth
         energy_function: energy function

     Returns:
         accuracy
     """
    pos_err = position_MSE_err_fn(output_traj, target_traj)
    eng_err = penergy_MSE_err_fn(output_traj, target_traj, energy_function)
    # pos_err = position_BE_err_fn(output_traj, target_traj)
    # eng_err = energy_BE_err_fn(output_traj, target_traj, energy_function)
    return pos_err, eng_err
