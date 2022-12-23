import os

import numpy as np

from .data import PendulumData
from .utils import download_file_from_google_drive, timing


def choose_data(args):
    data_name = args.data_name
    if data_name == 'PendulumData':
        data = PendulumData(obj=args.obj, dim=args.dim,
                            train_num=args.train_num,
                            test_num=args.test_num,
                            m=[1 for i in range(args.obj)],
                            l=[1 for i in range(args.obj)])
    else:
        raise NotImplementedError

    return data


@timing
def get_dataset(args):
    data_path = './outputs/' + args.taskname
    filename = data_path + '/dataset_{}_{}_hamilton.npy'.format(args.tasktype, args.obj)

    if len(args.dataset_url) != 0:
        print('Start downloading dataset.')
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        download_file_from_google_drive(args.dataset_url, filename)

    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        data = choose_data(args)
        data.Init_data()
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        np.save(filename, data)

    print('Number of samples in train dataset : ', len(data.y_train))
    print('Number of samples in test dataset : ', len(data.y_test))

    return data
