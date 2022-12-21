import random

import numpy as np
import scipy.integrate
import scipy.misc
import torch

solve_ivp = scipy.integrate.solve_ivp


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def init_random_state(random_seed=3407):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # speed up computation
    torch.backends.cudnn.benchmark = True


# 求导
def dfx(f, x):
    """f is output, x is input
    """
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]


def L2_loss(x, y):
    num_examples = x.size()[0]
    diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
    y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
    return torch.mean(diff_norms / y_norms)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###########################################################################################################
