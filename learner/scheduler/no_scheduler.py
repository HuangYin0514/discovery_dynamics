import torch


# Schedule learning rate--------------------------------------------
class no_scheduler():
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass
