import numpy as np
import torch

import learner as ln

if __name__ == '__main__':
    obj = 2
    dim = 1

    dof = obj * dim * 2

    test_input = torch.randn(20, dof)
    test_target = torch.randn(20, dof)
    test_input.requires_grad = True

    '''
        model = ln.nn.FNN(4, 1, width=200)
        model = ln.nn.HNN(4, layers=1, width=200)
        model = ln.nn.MechanicsNN(4, layers=1, width=200)
        model = ln.nn.LNN(4, layers=3, width=200)

    '''
    model = ln.nn.ModLaNet(4, layers=3, width=200)
    model.device = "cpu"
    model.dtype = "float"
    print(model)

    test_output = model(None, test_input)
    print(test_output.shape)
    #
    # loss = model.criterion(test_input, test_target, criterion_method='MSELoss')
    # print(loss)
    #
