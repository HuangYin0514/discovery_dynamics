# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/3 3:50 PM
@desc:
"""
import glob
import os
import os.path as osp

from learner.data.datasets._bases import BaseDynamicsDataset
from learner.utils import download_file_from_google_drive


class Pendulum2(BaseDynamicsDataset):
    """
    Pendulum with 2 bodies
    Reference:
    # ref: Simplifying Hamiltonian and Lagrangian Neural Networks via Explicit Constraints
    # URL: https://proceedings.neurips.cc/paper/2020/file/9f655cc8884fda7ad6d8a6fb15cc001e-Paper.pdf
    Dataset statistics:
    # type: hamilton
    # obj: 2
    # dim: 1
    """

    dataset_dir = ''

    train_url = '1'
    val_url = ''
    test_url = ''

    def __init__(self, root='',download_data=False, **kwargs):
        super(Pendulum2, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        if download_data == 'True':
            print('Start downloading dataset.')
            os.makedirs(self.train_dir) if not os.path.exists(self.train_dir) else None
            os.makedirs(self.val_dir) if not os.path.exists(self.val_dir) else None
            os.makedirs(self.test_dir) if not os.path.exists(self.test_dir) else None
            download_file_from_google_drive(self.train_url, self.train_dir)
            download_file_from_google_drive(self.val_url, self.train_dir)
            download_file_from_google_drive(self.test_url, self.train_dir)

        self._check_before_run()

        train = self._process_dir(self.train_dir)
        val = self._process_dir(self.val_dir)
        test = self._process_dir(self.test_dir)

        print("=> Pendulum2 loaded")
        self.print_dataset_statistics('train', train)
        self.print_dataset_statistics('val', val)
        self.print_dataset_statistics('test', test)

        self.train = train
        self.val = val
        self.test = test

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, ):
        data_paths = glob.glob(osp.join(dir_path, '*.npy'))
        pattern = r'dataset_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+)_(\d+)\.npy'

        dataset = []
        for data_path in data_paths:
            parts = data_path.split('/')
            parts = parts[-1].split('.npy')[:-1]
            dataset_info = parts[-1].split('_')
            num_train_traj = int(dataset_info[1])
            states = int(dataset_info[2])
            min_t = float(dataset_info[3])
            max_t = float(dataset_info[4])
            len_t = int(dataset_info[5])

            dataset.append((data_path, num_train_traj,states, min_t, max_t, len_t))

        return dataset
