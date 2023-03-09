import os.path as osp

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import gendata
from .analyze import plot_energy, plot_compare_energy, plot_compare_state, plot_field, plot_trajectory
from .metrics import accuracy_fn
from .utils import timing


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

    def __init__(self, taskname, obj, dim, data, net, dtype, device):
        self.taskname = taskname
        self.obj = obj
        self.dim = dim
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

        pred_list = []
        labels_list = []

        pbar = tqdm(self.test_loader, desc='Processing')
        for test_data in pbar:
            inputs, true_traj = test_data
            X, t = inputs
            X, t = X.to(self.device), t.to(self.device)
            true_traj = true_traj.to(self.device)

            # pred ----------------------------------------------------------------
            net_pred = self.net.integrate(X, t).clone().detach()  # (bs, T, states)

            pred_list.append(net_pred)
            labels_list.append(true_traj)

        # error ----------------------------------------------------------------
        preds = torch.cat(pred_list, dim=0)
        labels = torch.cat(labels_list, dim=0)


        err = accuracy_fn(preds, labels, self.energy_fn)
        pos_err, eng_err = err
        result = ('net: {}'.format(self.net.__class__.__name__)
                  + '\n'
                  + 'pos_err: {:.3e} +/- {:.3e}'.format(pos_err.mean(), pos_err.std())
                  + '\n'
                  + 'eng_err: {:.3e} +/- {:.3e}'.format(eng_err.mean(), eng_err.std()))
        print(result)

        # solutions forms ----------------------------------------------------------------
        check_index = 0
        ground_true = labels[check_index]
        net_pred = preds[check_index]
        true_q, true_p = ground_true.chunk(2, dim=-1)  # (T, states)
        pred_q, pred_p = net_pred.chunk(2, dim=-1)  # (T, states)

        true_eng = self.energy_fn(ground_true)
        true_kinetic_eng = self.kinetic_fn(ground_true)
        true_potential_eng = self.potential_fn(ground_true)
        pred_eng = self.energy_fn(net_pred)
        pred_kinetic_eng = self.kinetic_fn(net_pred)
        pred_potential_eng = self.potential_fn(net_pred)

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

        # save results ----------------------------------------------------------------
        save_path = osp.join('./outputs/', self.taskname)
        name = 'gt_' + self.data[0].__class__.__name__
        save_list = preds.detach().cpu().numpy()
        np.save(save_path + '/result_' + name + '.npy', save_list)
        save_list = labels.detach().cpu().numpy()
        name = self.net.__class__.__name__
        np.save(save_path + '/result_' + name + '.npy', save_list)

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
        dataset, _, _, test_loader = self.data

        self.data_name = dataset.__class__.__name__
        # dataloader
        self.test_loader = test_loader

        # energy function
        dataclass = getattr(gendata.dataset, self.data_name)(self.obj, self.dim)
        dataclass.device = self.device
        dataclass.dtype = self.dtype
        self.energy_fn = dataclass.energy_fn
        self.kinetic_fn = dataclass.kinetic
        self.potential_fn = dataclass.potential

    def __init_net(self):
        self.net.device = self.device
        self.net.dtype = self.dtype
