import numpy as np
import torch

import learner as ln

if __name__ == '__main__':
    test_input = torch.randn(20, 4)
    test_target = torch.randn(20, 4)
    test_input.requires_grad = True

    '''
        model = ln.nn.FNN(4, 1, width=200)
        model = ln.nn.HNN(4, layers=1, width=200)

    '''
    model = ln.nn.Baseline(4, layers=1, width=200)
    model.device = "cpu"
    model.dtype = "float"
    print(model)

    test_output = model(test_input)
    print(test_output.shape)

    loss = model.criterion(test_input, test_target, criterion_method='MSELoss')
    print(loss)

    h = 0.05
    test_traj = np.asarray([11, 2, 0, 0])
    pred_res = model.predict(test_traj, h, 0, 3, circular_motion=True)
    print(pred_res.shape)
    print("done")
