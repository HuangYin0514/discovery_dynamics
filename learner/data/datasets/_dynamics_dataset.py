# encoding: utf-8
"""
@author: Yin Huang
@contact: hy1071324110@gmail.com
@time: 2023/1/4 3:42 PM
@desc:
"""

from torch.utils.data import Dataset


class DynamicsDataset(Dataset):
    """learning dynamics Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x0, t, h, X, y, E = self.dataset[index]

        if self.transform is not None:
            X = self.transform(X)

        return x0, t, h, X, y, E
