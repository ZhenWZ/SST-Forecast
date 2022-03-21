import glob
import h5py
import numpy as np
from fastai.basics import *
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.hook import summary
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.fp16 import *
from fastai.data.load import DataLoader as FastDataLoader
from fastai.callback.progress import *
from fastai.callback.tracker import *
from fastai.callback.tensorboard import *
from torch.utils.data import *
from collections.abc import Iterable
from data_utils import *

from vunet_model import Net1, VUNetLoss2, valid_leaderboard, valid_leaderboard2

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('vunet/log')                  # for tensorboard


# Datasets
root = '/home/featurize/data/Generate_Data_Step_0_496_264_20020601_20190409.mat'
train_dataset = MyDataset(root, mode='train', in_channels=32, out_channels=8, steps=4)
valid_dataset = MyDataset(root, mode='valid', in_channels=32, out_channels=8, steps=4)
test_dataset = MyDataset(root, mode='test', in_channels=32, out_channels=8, steps=4)

batch_size = 8
device = 'cuda'
folder_to_save_models = 'weights_32-32_epoch100'

# DataLoader
train_dl = FastDataLoader(dataset=train_dataset,
                          bs=batch_size,
                          pin_memory=True,
                          shuffle=True,
                          device=torch.device(device))
valid_dl = FastDataLoader(dataset=valid_dataset,
                          bs=batch_size,
                          pin_memory=True,
                          shuffle=True,
                          device=torch.device(device))
test_dl = FastDataLoader(dataset=test_dataset,
                          bs=batch_size,
                          pin_memory=True,
                          shuffle=False,
                          device=torch.device(device))
data = DataLoaders(train_dl, valid_dl, device=torch.device(device))
test_data = DataLoaders(test_dl, device=torch.device(device))

# Model
Model = Net1(in_channels=32,out_channels=8)

# Learner
learn = Learner(data, Model.to(device), loss_func=VUNetLoss2, metrics=[valid_leaderboard], model_dir=folder_to_save_models, cbs=CSVLogger)

# Training
learn.fit(100, lr=2e-4, cbs=[CSVLogger, EarlyStoppingCallback(monitor='valid_loss', patience=15), SaveModelCallback(every_epoch=2)])

