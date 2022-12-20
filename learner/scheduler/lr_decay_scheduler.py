import torch


# Schedule learning rate--------------------------------------------
class lr_decay_scheduler():
    def __init__(self, optimizer, lr_decay, iterations):
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.iterations = iterations

    def step(self):
        lr = self.optimizer.param_groups[0]['lr']
        lr_new = lr / self.lr_decay ** (1 / self.iterations)
        self.optimizer.param_groups[0]['lr'] = lr_new
