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
import argparse

parser = argparse.ArgumentParser(description="generate lq")
parser.add_argument("--data_dir", type=str, default="./SIDD_Medium_Srgb_Patches_512/train/", help="path of train dataset")
parser.add_argument("--val_dir", type=str, default="./SIDD_Valid_Srgb_Patches_256/valid/", help="path of val dataset")
parser.add_argument("--scale", type=int, default=4, help="scale")
parser.add_argument("--gt_size", type=int, default=144, help="crop size for training")
opt = parser.parse_args()

def run_one_epoch():
    alternative_opt = {
        "train_dir":opt.data_dir,
        "val_dir":opt.val_dir,
        "scale":opt.scale,
        "gt_size":opt.gt_size
    }

    train_dataset = LQGTRNDataset(opt=None, is_train=True, alternative_opt=alternative_opt)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    with paddle.no_grad():
        for iter,data_val in enumerate(train_loader):
            print("lq image ", iter, " done.")
    
    print("All LQ images generated.")

if __name__ == '__main__':
    run_one_epoch()
