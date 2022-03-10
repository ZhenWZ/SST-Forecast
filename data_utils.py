import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
from torch.utils.data import *

class MyDataset(Dataset):
    def __init__(self, root, mode, in_channels=8, out_channels=32, steps=1, transform=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.dataset = h5py.File(self.root, 'r')['samples']
        self.steps = steps
        self.channel_length = in_channels + out_channels
        self.transform = transform
        self.split_idx = int(self.dataset.shape[0]*0.7)
        self.test_split_idx = int(self.dataset.shape[0]*0.9)
        self.data_min_ = 17.
        self.data_max_ = 32.

        # Generate index of the dataset
        if mode == "train":
            self.length = np.floor((self.split_idx-self.channel_length)/steps + 1).astype(int)
            self.data_idx = np.zeros((self.length, self.channel_length), dtype=int)

            for i in range(self.length):
                self.data_idx[i] = np.arange(i*steps, i*steps+self.channel_length, 1, dtype=int)

            self.x_idx = self.data_idx[:, :in_channels]
            self.y_idx = self.data_idx[:, in_channels:]

        elif mode =="valid":
            self.length = np.floor((self.test_split_idx-self.split_idx-self.channel_length)/steps + 1).astype(int)
            self.data_idx = np.zeros((self.length, self.channel_length), dtype=int)

            for i in range(self.length):
                self.data_idx[i] = np.arange(self.split_idx+i*steps, self.split_idx+i*steps+self.channel_length, 1, dtype=int)

            self.x_idx = self.data_idx[:, :in_channels]
            self.y_idx = self.data_idx[:, in_channels:]

        else:
            self.length = np.floor((self.dataset.shape[0]-self.test_split_idx-self.channel_length)/steps + 1).astype(int)
            self.data_idx = np.zeros((self.length, self.channel_length), dtype=int)

            for i in range(self.length):
                self.data_idx[i] = np.arange(self.test_split_idx+i*steps, self.test_split_idx+i*steps+self.channel_length, 1, dtype=int)

            self.x_idx = self.data_idx[:, :in_channels]
            self.y_idx = self.data_idx[:, in_channels:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Use size of 256*256
        x = self.dataset[self.x_idx[idx]][:,:256,:256].astype(np.float32)
        y = self.dataset[self.y_idx[idx]][:,:256,:256].astype(np.float32)
        x = self.rescale(x)
        y = self.rescale(y)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def rescale(self, x):
        # scale to (0,1)
        x = (x - self.data_min_)/(self.data_max_ - self.data_min_)
        return x

    
def rescale(x, min, max):
    return (x-min)/(max-min)

def recover(x, min, max):
    return x*(max-min)+min
