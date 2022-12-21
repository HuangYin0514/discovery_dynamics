import torch


# def L2_norm_loss(u, v):
#     return (u - v).pow(2).mean()


def L2_norm_loss(x, y):
    print("l2")
    num_examples = x.size()[0]
    diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
    y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
    return torch.mean(diff_norms / y_norms)
