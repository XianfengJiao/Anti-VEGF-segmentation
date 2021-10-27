import torch
import os.path as op
import numpy as np
import torch.utils.data as data
import scipy.io as scio

class OctData(data.Dataset):
    def __init__(self, dataset, train=True):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # anno wait to edit
        data = scio.loadmat(self.dataset[index])
        img = data['img']
        annotation = data['annotation']
        # anno = np.zeros((4, annotation.shape[0], annotation.shape[1]))
        # for i in range(1, 5):
        #     anno[i - 1][annotation == i] = 1
        return img, annotation