import argparse
import os
import sys
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# matplotlib.use('TKAgg')

sys.path.append('')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from src.models import Baseline, HNN, LNN
from data import Dataset
from src.utils import L2_loss, get_device, init_random_state, count_parameters

device = get_device()


def get_args():
    parser = argparse.ArgumentParser(description=None)
    # MODEL SETTINGS
    parser.add_argument('--model', default='hnn', type=str,
                        help='Select model to train, either \'modlanet\', \'lnn\', \'hnn\', or \'baseline\' currently')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--obj', default=1, type=int, help='number of elements')
    parser.add_argument('--dof', default=1, type=int, help='degree of freedom')
    parser.add_argument('--dim', default=1, type=int, help='space dimension, 2D or 3D')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')

    # For HNN
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')

    # TRAINING SETTINGS
    parser.add_argument('--gpu', nargs='?', const=True, default=True, help='try to use gpu?')
    parser.add_argument('--overwrite', nargs='?', const=True, default=False, help='overwrite the saved model.')
    parser.add_argument('--load_epoch', default=0000, type=int, help='load saved model at steps k.')
    parser.add_argument('--end_epoch', default=2000, type=int, help='end of training epoch')
    parser.add_argument('--use_lr_scheduler', default=True, help='whether to use lr_scheduler.')
    parser.add_argument('--samples', default=20, type=int, help='the number of sampling trajectories')
    parser.add_argument('--noise', default=0., type=float, help='the noise amplitude of the data')

    # GENERAL SETTINGS
    parser.add_argument('--save_dir', default=THIS_DIR + '/data', type=str, help='where to save the trained model')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--plot', default=False, action='store_true', help='plot training and testing loss?')
    parser.add_argument('--verbose', dest='verbose', default=True, action='store_true', help='verbose?')
    parser.add_argument('--print_every', default=100, type=int, help='number of gradient steps between prints')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.set_defaults(feature=True)
    return parser.parse_known_args()


def train_Baseline(args):
    t0 = time.time()

    if args.verbose:
        print("Training Baseline model:")
        print('using the device is:', device)

    # init model and optimizer
    input_dim = args.obj * args.dof * 2
    model = Baseline(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=input_dim).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.learn_rate,
        weight_decay=1e-4,  # 1e-8
        betas=(0.9, 0.999))
    scheduler_enc = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[3000, 9000, 15000],
                                                         gamma=0.5)  # StepLR(optim, step_size=700, gamma=1)
    print('number of parameters in model: ', count_parameters(model))

    # load trained models if possible
    # naming example: model-2-pend-hnn-hidden_dim-200-end_epoch-10000-noise-0.0-learn_rate-0.001.tar
    start_epoch = 0
    if args.load_epoch > 0:
        path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj,
                                                                                                args.name, args.model,
                                                                                                args.hidden_dim,
                                                                                                args.load_epoch,
                                                                                                args.noise,
                                                                                                args.learn_rate)
        if os.path.exists(path):
            print('load model: {}'.format(path))
            checkpoint = torch.load(path)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_state_dict'])
            scheduler_enc.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.load_epoch
        else:
            raise ValueError('Trained model \'{}\' not exists. please check the path.'.format(path))

    # naming example: dataset_2_pend_hnn_noise_0.0.npy
    filename = args.save_dir + '/dataset_' + str(args.obj) + '_' + args.name + '_hnn_noise_' + str(args.noise) + '.npy'
    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset = Dataset(obj=args.obj, m=[1 for i in range(args.obj)], l=[1 for i in range(args.obj)])
        data = dataset.get_dataset(seed=args.seed, system='hnn', noise_std=args.noise, samples=args.samples)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        np.save(filename, data)

    data_train_x = torch.tensor(data['x'], dtype=torch.float32, device=device)
    data_train_dx = torch.tensor(data['dx'], dtype=torch.float32, device=device)
    data_test_x = torch.tensor(data['test_x'], dtype=torch.float32, device=device)
    data_test_dx = torch.tensor(data['test_dx'], dtype=torch.float32, device=device)

    dataset_train = torch.utils.data.TensorDataset(data_train_x, data_train_dx)
    dataset_test = torch.utils.data.TensorDataset(data_test_x, data_test_dx)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    print('number of samples in train dataset : ', len(dataset_train))
    print('number of samples in test dataset : ', len(dataset_test))

    if args.verbose:
        print('Data Obtaining Time: {}'.format(time.time() - t0))

    # epochs开始
    stats = {'loss_train': [], 'loss_test': []}
    for epoch in tqdm(range(start_epoch, args.end_epoch), desc='Processing'):
        # 训练
        model.train()
        for x, y in dataloader_train:
            x, y = x.to(device), y.to(device)
            x.requires_grad = True
            optim.zero_grad()
            pred = model(x).reshape(y.shape)
            loss_train = L2_loss(pred, y)
            loss_train.backward()
            optim.step()

        # 推断阶段
        for x, y in dataloader_test:
            x.requires_grad = True
            x, y = x.to(device), y.to(device)
            pred = model(x).reshape(y.shape)
            loss_test = L2_loss(pred, y)

        # 调整学习率
        if args.use_lr_scheduler:
            scheduler_enc.step()

        stats['loss_train'].append(loss_train.item())
        stats['loss_test'].append(loss_test.item())

        if args.verbose and epoch % args.print_every == 0:
            print("epoch {}, train_loss {:.4e}, test_loss {:.4e}".format(epoch, loss_train.item(), loss_test.item()))

    # 计算模型距离
    data_train_x.requires_grad = True
    data_train_dx_hat = model(data_train_x).reshape(data_train_dx.shape)
    dist_train = (data_train_dx - data_train_dx_hat) ** 2
    data_test_x.requires_grad = True
    data_test_dx_hat = model(data_test_x).reshape(data_test_dx.shape)
    dist_test = (data_test_dx - data_test_dx_hat) ** 2
    print('Final train error {:.4e} +/- {:.4e}\nFinal test error {:.4e} +/- {:.4e}'
          .format(dist_train.mean().item(), dist_train.std().item() / np.sqrt(dist_train.shape[0]),
                  dist_test.mean().item(), dist_test.std().item() / np.sqrt(dist_test.shape[0])))

    if args.plot:
        fig = plt.figure()
        plt.semilogy((stats['loss_train']), 'b')
        plt.semilogy((stats['loss_test']), 'r')
        # plt.show()
        path = '{}/fig-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.{}'.format(
            args.save_dir, args.obj, args.name,
            args.model, args.hidden_dim, args.load_epoch, args.end_epoch, args.noise,
            args.learn_rate, 'png')
        fig.savefig(path)

    return model, stats, optim, scheduler_enc


def train_HNN(args):
    t0 = time.time()

    if args.verbose:
        print("Training HNN model:")
        print('using the device is:', device)

    # init model and optimizer
    input_dim = args.obj * args.dof * 2
    model = HNN(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.learn_rate,
        weight_decay=1e-8,
        betas=(0.9, 0.999))
    scheduler_enc = torch.optim.lr_scheduler.StepLR(optim, step_size=700, gamma=1)
    print('number of parameters in model: ', count_parameters(model))

    # load trained models if possible
    # naming example: model-2-pend-hnn-hidden_dim-200-end_epoch-10000-noise-0.0-learn_rate-0.001.tar
    start_epoch = 0
    if args.load_epoch > 0:
        path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj,
                                                                                                args.name, args.model,
                                                                                                args.hidden_dim,
                                                                                                args.load_epoch,
                                                                                                args.noise,
                                                                                                args.learn_rate)
        if os.path.exists(path):
            print('load model: {}'.format(path))
            checkpoint = torch.load(path)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_state_dict'])
            scheduler_enc.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.load_epoch
        else:
            raise ValueError('Trained model \'{}\' not exists. please check the path.'.format(path))

    # naming example: dataset_2_pend_hnn_noise_0.0.npy
    filename = args.save_dir + '/dataset_' + str(args.obj) + '_' + args.name + '_hnn_noise_' + str(args.noise) + '.npy'
    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset = Dataset(obj=args.obj, m=[1 for i in range(args.obj)], l=[1 for i in range(args.obj)])
        data = dataset.get_dataset(seed=args.seed, system='hnn', noise_std=args.noise, samples=args.samples)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        np.save(filename, data)

    data_train_x = torch.tensor(data['x'], dtype=torch.float32, device=device)
    data_train_dx = torch.tensor(data['dx'], dtype=torch.float32, device=device)
    data_test_x = torch.tensor(data['test_x'], dtype=torch.float32, device=device)
    data_test_dx = torch.tensor(data['test_dx'], dtype=torch.float32, device=device)

    dataset_train = torch.utils.data.TensorDataset(data_train_x, data_train_dx)
    dataset_test = torch.utils.data.TensorDataset(data_test_x, data_test_dx)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    print('number of samples in train dataset : ', len(dataset_train))
    print('number of samples in test dataset : ', len(dataset_test))

    if args.verbose:
        print('Data Obtaining Time: {}'.format(time.time() - t0))

    # epochs开始
    stats = {'loss_train': [], 'loss_test': []}
    for epoch in tqdm(range(start_epoch, args.end_epoch), desc='Processing'):
        # 训练
        model.train()
        for x, y in dataloader_train:
            x, y = x.to(device), y.to(device)
            x.requires_grad = True
            optim.zero_grad()
            pred = model(x).reshape(y.shape)
            loss_train = L2_loss(pred, y)
            loss_train.backward()
            optim.step()

        # 推断阶段
        for x, y in dataloader_test:
            x.requires_grad = True
            x, y = x.to(device), y.to(device)
            pred = model(x).reshape(y.shape)
            loss_test = L2_loss(pred, y)

        # 调整学习率
        if args.use_lr_scheduler:
            scheduler_enc.step()

        stats['loss_train'].append(loss_train.item())
        stats['loss_test'].append(loss_test.item())

        if args.verbose and epoch % args.print_every == 0:
            print("epoch {}, train_loss {:.4e}, test_loss {:.4e}".format(epoch, loss_train.item(), loss_test.item()))

    # 计算模型距离
    data_train_x.requires_grad = True
    data_train_dx_hat = model(data_train_x).reshape(data_train_dx.shape)
    dist_train = (data_train_dx - data_train_dx_hat) ** 2
    data_test_x.requires_grad = True
    data_test_dx_hat = model(data_test_x).reshape(data_test_dx.shape)
    dist_test = (data_test_dx - data_test_dx_hat) ** 2
    print('Final train error {:.4e} +/- {:.4e}\nFinal test error {:.4e} +/- {:.4e}'
          .format(dist_train.mean().item(), dist_train.std().item() / np.sqrt(dist_train.shape[0]),
                  dist_test.mean().item(), dist_test.std().item() / np.sqrt(dist_test.shape[0])))

    if args.plot:
        fig = plt.figure()
        plt.semilogy((stats['loss_train']), 'b')
        plt.semilogy((stats['loss_test']), 'r')
        # plt.show()
        path = '{}/fig-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.{}'.format(
            args.save_dir, args.obj, args.name,
            args.model, args.hidden_dim, args.load_epoch, args.end_epoch, args.noise,
            args.learn_rate, 'png')
        fig.savefig(path)

    return model, stats, optim, scheduler_enc


def train_LNN(args):
    t0 = time.time()

    if args.verbose:
        print("Training LNN model:")
        print('using the device is:', device)

    # init model and optimizer
    input_dim = args.obj * args.dof * 2
    model = LNN(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.learn_rate,
        weight_decay=1e-8,
        betas=(0.9, 0.999))
    scheduler_enc = torch.optim.lr_scheduler.StepLR(optim, step_size=700, gamma=1)
    print('number of parameters in model: ', count_parameters(model))

    # load trained models if possible
    # naming example: model-2-pend-hnn-hidden_dim-200-end_epoch-10000-noise-0.0-learn_rate-0.001.tar
    start_epoch = 0
    if args.load_epoch > 0:
        path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj,
                                                                                                args.name, args.model,
                                                                                                args.hidden_dim,
                                                                                                args.load_epoch,
                                                                                                args.noise,
                                                                                                args.learn_rate)
        if os.path.exists(path):
            print('load model: {}'.format(path))
            checkpoint = torch.load(path)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_state_dict'])
            scheduler_enc.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.load_epoch
        else:
            raise ValueError('Trained model \'{}\' not exists. please check the path.'.format(path))

    # naming example: dataset_2_pend_hnn_noise_0.0.npy
    filename = args.save_dir + '/dataset_' + str(args.obj) + '_' + args.name + '_lnn_noise_' + str(args.noise) + '.npy'
    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        dataset = Dataset(obj=args.obj, m=[1 for i in range(args.obj)], l=[1 for i in range(args.obj)])
        data = dataset.get_dataset(seed=args.seed, system='lnn', noise_std=args.noise, samples=args.samples)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        np.save(filename, data)

    data_train_x = torch.tensor(data['x'], dtype=torch.float32, device=device)
    data_train_v = torch.tensor(data['v'], dtype=torch.float32, device=device)
    data_train_ac = torch.tensor(data['ac'], dtype=torch.float32, device=device)

    data_test_x = torch.tensor(data['test_x'], dtype=torch.float32, device=device)
    data_test_v = torch.tensor(data['test_v'], dtype=torch.float32, device=device)
    data_test_ac = torch.tensor(data['test_ac'], dtype=torch.float32, device=device)

    data_train_input = torch.cat([data_train_x, data_train_v], dim=1)
    data_test_input = torch.cat([data_test_x, data_test_v], dim=1)

    dataset_train = torch.utils.data.TensorDataset(data_train_input, data_train_ac)
    dataset_test = torch.utils.data.TensorDataset(data_test_input, data_test_ac)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    print('number of samples in train dataset : ', len(dataset_train))
    print('number of samples in test dataset : ', len(dataset_test))

    if args.verbose:
        print('Data Obtaining Time: {}'.format(time.time() - t0))

    # epochs开始
    stats = {'loss_train': [], 'loss_test': []}
    for epoch in tqdm(range(start_epoch, args.end_epoch), desc='Processing'):
        # 训练
        model.train()
        for x, y in dataloader_train:
            x, y = x.to(device), y.to(device)
            x.requires_grad = True
            optim.zero_grad()
            pred = model(x).reshape(y.shape)
            loss_train = L2_loss(pred, y)
            loss_train.backward()
            optim.step()

        # 推断阶段
        for x, y in dataloader_test:
            x.requires_grad = True
            x, y = x.to(device), y.to(device)
            pred = model(x).reshape(y.shape)
            loss_test = L2_loss(pred, y)

        # 调整学习率
        if args.use_lr_scheduler:
            scheduler_enc.step()

        stats['loss_train'].append(loss_train.item())
        stats['loss_test'].append(loss_test.item())

        if args.verbose and epoch % args.print_every == 0:
            print("epoch {}, train_loss {:.4e}, test_loss {:.4e}".format(epoch, loss_train.item(), loss_test.item()))

    # 计算模型距离
    data_train_input.requires_grad = True
    data_train_ac_hat = model(data_train_input).reshape(data_train_ac.shape)
    dist_train = (data_train_ac - data_train_ac_hat) ** 2
    data_test_input.requires_grad = True
    data_test_ac_hat = model(data_test_input).reshape(data_test_ac.shape)
    dist_test = (data_test_ac - data_test_ac_hat) ** 2
    print('Final train error {:.4e} +/- {:.4e}\nFinal test error {:.4e} +/- {:.4e}'
          .format(dist_train.mean().item(), dist_train.std().item() / np.sqrt(dist_train.shape[0]),
                  dist_test.mean().item(), dist_test.std().item() / np.sqrt(dist_test.shape[0])))

    if args.plot:
        fig = plt.figure()
        plt.semilogy((stats['loss_train']), 'b')
        plt.semilogy((stats['loss_test']), 'r')
        # plt.show()
        path = '{}/fig-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.{}'.format(
            args.save_dir, args.obj, args.name,
            args.model, args.hidden_dim, args.load_epoch, args.end_epoch, args.noise,
            args.learn_rate, 'png')
        fig.savefig(path)

    return model, stats, optim, scheduler_enc


def main():
    args = get_args()[0]
    init_random_state(args.seed)

    model, stats = None, None

    # Check whether model exists
    # Example: model-2-pend-modlanet-hidden_dim-50-end_epoch-3000-noise-0.0-learn_rate-0.001.tar
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    args.hidden_dim = args.energy_hidden_dim if args.model == 'modlanet' else args.hidden_dim
    path = '{}/model-{}-{}-{}-hidden_dim-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar'.format(args.save_dir, args.obj,
                                                                                            args.name,
                                                                                            args.model, args.hidden_dim,
                                                                                            args.end_epoch,
                                                                                            args.noise, args.learn_rate)
    if os.path.exists(path):
        if args.overwrite:
            print('Model already exist, overwrite it.')
        else:
            raise ValueError('Trained model \'{}\' already exists. '
                             'For overwrite, please use --overwrite'.format(path))

    if args.model == 'baseline':
        model, states, optim, scheduler = train_Baseline(args)
    elif args.model == 'hnn':
        model, states, optim, scheduler = train_HNN(args)
    elif args.model == 'lnn':
        model, states, optim, scheduler = train_LNN(args)
    else:
        raise ValueError('Model \'{}\' is not implemented'.format(args.model))

    # save
    torch.save({'network_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, path)
    training_process = '{}/train_process-{}-{}-{}-hidden_dim-{}-start_epoch-{}-end_epoch-{}-noise-{}-learn_rate-{}.tar.npy'.format(
        args.save_dir, args.obj, args.name,
        args.model, args.hidden_dim, args.load_epoch, args.end_epoch,
        args.noise, args.learn_rate)
    np.save(training_process, states)


if __name__ == "__main__":
    main()
