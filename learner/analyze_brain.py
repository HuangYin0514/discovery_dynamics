import os
import os.path as osp

import time
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from tqdm import tqdm

from .metrics import accuracy_fn
from .nn.base_module import LossNN
from .regularization import symplectic_prior_reg
from .scheduler import lr_decay_scheduler, no_scheduler
from .utils import timing
from .analyze import plot_energy, plot_compare_energy, plot_compare_state, plot_field, plot_trajectory


class AnalyzeBrain:
    '''Runner based on torch.
    '''
    analyze_brain = None

    @classmethod
    def Init(cls, **kwargs):
        cls.analyze_brain = cls(**kwargs)

    @classmethod
    def Run(cls):
        cls.analyze_brain.run()

    def __init__(self, taskname, data, net, dtype, device):
        self.taskname = taskname
        self.data = data
        self.net = net
        self.dtype = dtype
        self.device = device

        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None

        self.__optimizer = None
        self.__scheduler = None
        self.__criterion = None

    @timing
    def run(self):
        self.__init_brain()
        print('analyze...', flush=True)

        test_data = next(iter(self.test_loader))  # get one data for test
        inputs, labels = test_data
        X, t = inputs
        X, t = X.to(self.device), t.to(self.device)
        labels = labels.to(self.device)

        # pred ----------------------------------------------------------------
        pred = self.net.integrate(X, t)  # (bs, T, states)

        # error ----------------------------------------------------------------
        err = accuracy_fn(pred, labels, self.energy_fn)
        mse_err, rel_err, eng_err = err

        result = ('mse_err: {:.3e}'.format(mse_err)
                  + '\n'
                  + 'rel_err: {:.3e}'.format(rel_err)
                  + '\n'
                  + 'eng_err: {:.3e}'.format(eng_err))
        print(result)

        # solutions forms ----------------------------------------------------------------
        check_index = 0
        ground_true = labels[check_index]
        net_pred = pred[check_index]
        true_q, true_p = ground_true.chunk(2, dim=-1)  # (T, states)
        pred_q, pred_p = net_pred.chunk(2, dim=-1)  # (T, states)

        true_eng = torch.stack([self.energy_fn(i) for i in ground_true])
        true_kinetic_eng = torch.stack([self.kinetic_fn(i) for i in ground_true])
        true_potential_eng = torch.stack([self.potential_fn(i) for i in ground_true])
        pred_eng = torch.stack([self.energy_fn(i) for i in net_pred])
        pred_kinetic_eng = torch.stack([self.kinetic_fn(i) for i in net_pred])
        pred_potential_eng = torch.stack([self.potential_fn(i) for i in net_pred])

        t = t.detach().cpu().numpy()

        ground_true = ground_true.detach().cpu().numpy()
        true_q = true_q.detach().cpu().numpy()
        true_p = true_p.detach().cpu().numpy()
        true_eng = true_eng.detach().cpu().numpy()
        true_kinetic_eng = true_kinetic_eng.detach().cpu().numpy()
        true_potential_eng = true_potential_eng.detach().cpu().numpy()

        net_pred = net_pred.detach().cpu().numpy()
        pred_q = pred_q.detach().cpu().numpy()
        pred_p = pred_p.detach().cpu().numpy()
        pred_eng = pred_eng.detach().cpu().numpy()
        pred_kinetic_eng = pred_kinetic_eng.detach().cpu().numpy()
        pred_potential_eng = pred_potential_eng.detach().cpu().numpy()

        # plot results ----------------------------------------------------------------
        save_path = osp.join('./outputs/', self.taskname, 'fig-analyze.pdf')
        fig, ax = plt.subplots(8, 1, figsize=(6, 24), dpi=100)

        plot_trajectory(self.data_name, ax[0], true_q, pred_q, 'Trajectory')

        plot_energy(ax[1], t, true_eng, true_potential_eng, true_kinetic_eng, 'Ground Truth Energy')
        plot_energy(ax[2], t, pred_eng, pred_potential_eng, pred_kinetic_eng, 'Prediction Energy')
        plot_compare_energy(ax[3], t, true_eng, pred_eng, 'Compare Energy')

        plot_compare_state(ax[4], t, true_q, pred_q, 'State $q$')
        plot_compare_state(ax[5], t, true_p, pred_p, 'State $p$')

        plot_field(ax[6], t, true_q, true_p, 'True Field')
        plot_field(ax[7], t, pred_q, pred_p, 'Prediction Field')

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches='tight')
        plt.show()

    def __init_brain(self):
        self.__init_data()
        self.__init_net()

    def __init_data(self):
        dataset, train_loader, val_loader, test_loader = self.data

        self.data_name = dataset.__class__.__name__
        # dataloader
        self.test_loader = test_loader

        # energy function
        self.energy_fn = dataset.energy_fn
        self.kinetic_fn = dataset.kinetic
        self.potential_fn = dataset.potential

    def __init_net(self):
        self.net.device = self.device
        self.net.dtype = self.dtype
