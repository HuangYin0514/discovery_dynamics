import os
import time
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from .nn.base_module import LossNN
from .utils import timing, deprecated
from .scheduler import lr_decay_scheduler, no_scheduler

from torch.optim import lr_scheduler


class Brain:
    '''Runner based on torch.
    '''
    brain = None

    @classmethod
    def Init(cls, taskname, data, net, criterion, optimizer, scheduler, lr, iterations, batch_size=None,
             print_every=1000, save=False, dtype='float', device='cpu'):
        cls.brain = cls(taskname, data, net, criterion, optimizer, scheduler, lr, iterations, batch_size,
                        print_every, save, dtype, device)

    @classmethod
    def Run(cls):
        cls.brain.run()

    @classmethod
    def Restore(cls):
        cls.brain.restore()

    @classmethod
    def Output(cls, data=True, best_model=True, loss_history=True, info=None, path=None, **kwargs):
        cls.brain.output(data, best_model, loss_history, info, path, **kwargs)

    @classmethod
    def Encounter_nan(cls):
        return cls.brain.encounter_nan

    @classmethod
    def Best_model(cls):
        return cls.brain.best_model

    def __init__(self, taskname, data, net, criterion, optimizer, scheduler, lr, iterations, batch_size,
                 print_every, save, dtype, device):
        self.taskname = taskname
        self.data = data
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.iterations = iterations
        self.batch_size = batch_size
        self.print_every = print_every
        self.save = save
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
        print('Training...', flush=True)
        loss_history = []
        if not os.path.isdir('./' + 'training_file/' + self.taskname + '/model'):
            os.makedirs('./' + 'training_file/' + self.taskname + '/model')

        pbar = tqdm(range(self.iterations + 1), desc='Processing')
        for i in pbar:
            '''training'''
            for data in self.train_loader:
                inputs, labels = data
                X, t = inputs
                X, t = X.to(self.device), t.to(self.device)
                labels = labels.to(self.device)

                loss = self.__criterion(self.net(t, X), labels)

                if torch.any(torch.isnan(loss)):
                    self.encounter_nan = True
                    print('Encountering nan, stop training', flush=True)
                    return None

                if i < self.iterations:
                    self.__optimizer.zero_grad()
                    loss.backward()
                    self.__optimizer.step()

            '''test'''
            if i % self.print_every == 0 or i == self.iterations:

                for data in self.test_loader:
                    inputs, labels = data
                    X, t = inputs
                    X, t = X.to(self.device), t.to(self.device)
                    labels = labels.to(self.device)

                    loss_test = self.__criterion(self.net.integrate(X, t), labels)

                loss_history.append([i, loss.item(), loss_test.item()])
                postfix = {
                    'Train_loss': '{:.3e}'.format(loss.item()),
                    'Test_loss': '{:.3e}'.format(loss_test.item()),
                    'lr': self.__optimizer.param_groups[0]['lr']
                }
                pbar.set_postfix(postfix)

            '''save'''
            if self.save:
                torch.save(self.net, 'training_file/' + self.taskname + '/model/model{}.pkl'.format(i))

            if self.__scheduler is not None:
                self.__scheduler.step()

        loss_record = np.array(loss_history)
        np.savetxt('training_file/' + self.taskname + '/loss.txt', loss_record)

        self.loss_history = np.array(loss_history)
        print('Done!', flush=True)
        return self.loss_history

    def restore(self):
        if self.loss_history is not None and self.save is True:
            best_loss_index = np.argmin(self.loss_history[:, 1])
            iteration = int(self.loss_history[best_loss_index, 0])
            loss_train = self.loss_history[best_loss_index, 1]
            loss_test = self.loss_history[best_loss_index, 2]
            print('Best model at iteration {}:'.format(iteration), flush=True)
            print('Train loss: {:.4e}, Test loss: {:.4e}'.format(loss_train, loss_test), flush=True)

            path = './outputs/' + self.taskname
            if not os.path.isdir('./outputs/' + self.taskname): os.makedirs('./outputs/' + self.taskname)
            contents = ('\n\n'
                        + 'Train completion time: '
                        + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                        + '\n'
                        + 'Task name: {}'.format(self.taskname)
                        + '\n'
                        + 'net name: {}'.format(self.net.__class__.__name__)
                        + '\n'
                        + 'Best model at iteration: {}'.format(iteration)
                        + '\n'
                        + 'Train loss: %s' % (loss_train)
                        + '\n'
                        + 'Test loss: %s' % (loss_test))
            f = open(path + '/output.txt', mode='a')
            f.write(contents)
            f.close()
            # self.best_model = torch.load('training_file/' + self.taskname + '/model/model{}.pkl'.format(iteration))
            self.best_model = torch.load(
                'training_file/' + self.taskname + '/model/model{}.pkl'.format(self.iterations))
        else:
            raise RuntimeError('restore before running or without saved models')
        return self.best_model

    def output(self, data, best_model, loss_history, info, path, **kwargs):
        if path is None:
            path = './outputs/' + self.taskname
        if not os.path.isdir(path): os.makedirs(path)
        '''
        Example: 
            model-pend_2_hnn.pkl
            loss-pend_2_hnn.txt
            fig-pend_2_hnn.txt        
            info-pend_2_hnn.txt        
        '''
        if best_model:
            filename = '/model-{}.pkl'.format(self.taskname)
            torch.save(self.best_model.state_dict(), path + filename)
        if loss_history:
            # save loss history to txt
            filename = '/loss-{}.txt'.format(self.taskname)
            np.savetxt(path + filename, self.loss_history)
            # save loss history to png
            filename = '/fig-{}.png'.format(self.taskname)
            plt.figure()
            plt.semilogy(self.loss_history[:, 0], self.loss_history[:, 1], 'b', label='train loss')
            plt.semilogy(self.loss_history[:, 0], self.loss_history[:, 2], 'r', label='test loss')
            plt.legend()
            plt.savefig(path + filename, format='png')
            plt.show()

        if info is not None:
            filename = '/info-{}.txt'.format(self.taskname)
            with open(path + filename, 'w') as f:
                for key, value in info.items():
                    f.write('{}: {}\n'.format(key, str(value)))
        for key, arg in kwargs.items():
            np.savetxt(path + '/' + key + '.txt', arg)

    def __init_brain(self):
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None

        self.__init_data()
        self.__init_net()
        self.__init_optimizer()
        self.__init_scheduler()
        self.__init_criterion()

    def __init_data(self):
        train_loader, test_loader = self.data
        self.train_loader = train_loader
        self.test_loader = test_loader

    def __init_net(self):
        self.net.device = self.device
        self.net.dtype = self.dtype

    def __init_optimizer(self):
        if self.optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.net.parameters(),
                                                lr=self.lr,
                                                weight_decay=1e-4,  # 1e-8
                                                betas=(0.9, 0.999))
        else:
            raise NotImplementedError

    def __init_scheduler(self):
        if self.scheduler == 'no_scheduler':
            self.__scheduler = no_scheduler(self.__optimizer)
        elif self.scheduler == 'lr_decay':
            self.__scheduler = lr_decay_scheduler(self.__optimizer, lr_decay=1000, iterations=self.iterations)
        elif self.scheduler == 'MultiStepLR':
            self.__scheduler = lr_scheduler.MultiStepLR(self.__optimizer,
                                                        milestones=[3000, 9000, 15000],
                                                        gamma=0.5)
        elif self.scheduler == 'StepLR':
            self.__scheduler = lr_scheduler.StepLR(self.__optimizer, step_size=700, gamma=0.7)
        else:
            raise NotImplementedError

    def __init_criterion(self):
        if isinstance(self.net, LossNN):
            criterion = partial(self.net.criterion, criterion_method=self.criterion)
            self.__criterion = criterion
        else:
            raise NotImplementedError(' net must be LossNN instance')
