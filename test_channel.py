import paddle
from paddle.io import DataLoader
from dataloaders.LQGTRN_dataset import LQGTRNDataset
from config import Config
import os
import math
from networks.InvDN_model import InvNet, constructor, gaussian_batch
from utils import dir_utils,img_utils


#有切割39.16
#无切割39.10

if __name__ == '__main__':
    opt = Config('training_4cards.yml') 
    val_dataset = LQGTRNDataset(opt,  is_train=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=10, drop_last=False) #1280/64=20

    scale = opt.TRAINING.SCALE
    down_num = int(math.log(scale, 2))
    model = InvNet(channel_in=3, channel_out=3, subnet_constructor=constructor, block_num=[8, 8], down_num=down_num)
    print("最新")
    ckpt = paddle.load("/home/zhangyichen/dct/cvpr/InvDN_paddlepaddle/experiments/Denoising/models-channel/InvDN/model_latest.pdparams")
    model.set_state_dict(ckpt['state_dict'])

    psnr_val_rgb = []
    model.eval()
    with paddle.no_grad():
        for data_val in val_loader:
            gt,noisy = data_val['GT'], data_val['Noisy']
            output = model(x=noisy)
            y_forw = paddle.concat((output[:, :3, :, :], 1 * gaussian_batch(output[:, 3:, :, :].shape)), axis=1)
            fake_H = model(x=y_forw, rev=True)

            crop_size = opt.TRAINING.SCALE
            gt = gt[:,:,crop_size:-crop_size, crop_size:-crop_size] #TODO ???为啥要crop那么奇怪
            fake_H = fake_H[:,:,crop_size:-crop_size, crop_size:-crop_size]
            psnr = img_utils.batch_PSNR(fake_H, gt, 1.)
            print(psnr)
            psnr_val_rgb.append(psnr)
    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
    print("PSNR: ",psnr_val_rgb)
    