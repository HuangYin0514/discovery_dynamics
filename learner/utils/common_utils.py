import random
import time
from functools import wraps

import numpy as np
import torch


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
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class lazy_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print('\'' + func.__name__ + '\'' + ' took {:.2f} minute '.format((time.time() - t) / 60))
        print('-' * 10)
        return result

    return wrapper


def enable_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.enable_grad():
            result = func(*args, **kwargs)
        return result

    return wrapper


def deprecated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # print('this function was deprecated!')
        raise Exception(f' {func.__name__} function was deprecated!')

    return wrapper
