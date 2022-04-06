import glob
import h5py
import argparse
import numpy as np
from torch.utils.data import *
from collections.abc import Iterable
from data_utils import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from vq_unet_model import VQ_Unet

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_channels', type=int, default=16)
    parser.add_argument('--out_channels', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--skip', type=int, default=5)
    parser.add_argument('--root', type=str)
    parser.add_argument('--log_name', type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    dataset = MyDataset(args.root, mode='train', in_channels=16, out_channels=4, steps=1, skip=5)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8)+1, int(len(dataset) * 0.2)])
    test_dataset = MyDataset(args.root, mode='test', in_channels=16, out_channels=4, steps=1, skip=5)

    device = 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

    model = VQ_Unet(in_channels=args.in_channels,out_channels=args.out_channels)
    logger = TensorBoardLogger("log", name=args.log_name)
    callbacks = [
        EarlyStopping(monitor='mse_loss', patience=20, mode='min', check_on_train_epoch_end=False), 
        ModelCheckpoint(dirpath='./checkpoints/', filename='vqunet-16-4-5-{epoch:02d}-{val_loss:.2f}', monitor='mse_loss', mode='min', save_top_k=3, verbose=True)
        ]
    trainer = pl.Trainer(gpus=1, callbacks=callbacks, logger=logger)
    trainer.fit(model, train_loader, val_loader)