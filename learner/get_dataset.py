import os

import numpy as np

from .data import PendulumData, BodyData
from .utils import download_file_from_google_drive, timing


def choose_data(data_name, train_num, test_num, obj, dim):
    if data_name == 'PendulumData':
        data = PendulumData(obj=obj, dim=dim,
                            train_num=train_num,
                            test_num=test_num,
                            m=[1 for i in range(obj)],
                            l=[1 for i in range(obj)])
    elif data_name == 'BodyData':
        data = BodyData(obj=obj, dim=dim,
                            train_num=train_num,
                            test_num=test_num,
                            m=[1 for i in range(obj)],
                            l=[0 for i in range(obj)])
    else:
        raise NotImplementedError

    return data


#  taskname data_name tasktype obj dim dataset_url
@timing
def get_dataset(data_name, taskname, tasktype, train_num, test_num, obj, dim, dataset_url):
    data_path = './outputs/' + taskname

    # example: dataset_pend_2_1_hamilton.npy
    filename = data_path + '/dataset_{}_{}_{}_hamilton.npy'.format(tasktype, obj, dim)

    if len(dataset_url) != 0:
        print('Start downloading dataset.')
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        download_file_from_google_drive(dataset_url, filename)

    if os.path.exists(filename):
        print('Start loading dataset.')
        data = np.load(filename, allow_pickle=True).item()
    else:
        print('Start generating dataset.')
        data = choose_data(data_name, train_num, test_num, obj, dim)
        data.Init_data()
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        # np.save(filename, data)

    print('Number of samples in train dataset : ', len(data.y_train))
    print('Number of samples in test dataset : ', len(data.y_test))

    return data
