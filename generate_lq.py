import paddle
from paddle.io import DataLoader
from dataloaders.LQGTRN_dataset import LQGTRNDataset
from config import Config
import os
import math
from networks.InvDN_model import InvNet, constructor, gaussian_batch
from utils import dir_utils,img_utils
import scipy
import numpy as np

def run_one_epoch():
    opt = Config('training_4cards.yml') 

    train_dataset = LQGTRNDataset(opt, is_train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    val_dataset = LQGTRNDataset(opt, is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    with paddle.no_grad():
        for iter,data_val in enumerate(train_loader):
            print(iter)

if __name__ == '__main__':
    run_one_epoch()
