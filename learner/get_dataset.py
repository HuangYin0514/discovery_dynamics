import os

import numpy as np

from .utils import download_file_from_google_drive, timing


@timing
def get_dataset(args, data):
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
        data.Init_data()
        os.makedirs(data_path) if not os.path.exists(data_path) else None
        np.save(filename, data)

    print('Number of samples in train dataset : ', len(data.y_train))
    print('Number of samples in test dataset : ', len(data.y_test))

    return data