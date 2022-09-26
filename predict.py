import paddle
from paddle.io import DataLoader
from dataloaders.LQGTRN_dataset import LQGTRNDataset
from config import Config
import os
import math
from networks.InvDN_model import InvNet, constructor
from utils import dir_utils,img_utils
import scipy,cv2
import numpy as np
from utils.util import forward_x8, Multi_forward_x8
import argparse

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')

parser.add_argument('--result_dir', default='./predict_results',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained/model_best.pdparams',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument("--data_path", type=str, default="./SIDD_mini/val_mini/", help='path to training data')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
paddle.set_device('gpu:0')

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

def main():
    alternative_opt = {
        "train_dir":args.data_path,
        "val_dir":args.data_path,
        "scale":4,
        "gt_size":144
    }
    val_dataset = LQGTRNDataset(opt=None,  is_train=False, alternative_opt=alternative_opt)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False, num_workers=1, drop_last=False)

    scale = 4
    down_num = int(math.log(scale, 2))
    model = InvNet(channel_in=3, channel_out=3, subnet_constructor=constructor, block_num=[8, 8], down_num=down_num)

    ckpt = paddle.load(args.weights)
    model.set_state_dict(ckpt['state_dict'])

    gaussian_scale = 1
    psnr_val_rgb = []
    ssim_val_rgb = []
    model.eval()
    with paddle.no_grad():
        for idx, data_val in enumerate(val_loader):
            gt,noisy = data_val['GT'], data_val['Noisy']
            fake_H = forward_x8(noisy, model.forward, gaussian_scale) # faster
            # fake_H = Multi_forward_x8(noisy, model.forward, gaussian_scale) #slower
            psnr = img_utils.batch_PSNR(fake_H, gt, 1.)
            ssim = img_utils.batch_SSIM(fake_H, gt)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

            res_path = os.path.join(args.result_dir, "denoised_"+str(idx)+".png")
            cv2.imwrite(res_path, fake_H[0].detach().cpu().numpy().transpose([1,2,0])*255.0)

    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
    ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)
    print("PSNR: {:.2f}".format(psnr_val_rgb))
    print("SSIM: {:.3f}".format(ssim_val_rgb))

if __name__ == '__main__':
    main()