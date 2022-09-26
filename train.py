import os
import argparse

import paddle

import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle import nn
import random
import time
import numpy as np
import math
import logging

from utils import dir_utils,img_utils
from dataloaders.LQGTRN_dataset import LQGTRNDataset

from networks.InvDN_model import InvNet, constructor, gaussian_batch

from losses import ReconstructionLoss, Gradient_Loss, SSIM_Loss
import paddle.distributed as dist

from visualdl import LogWriter

parser = argparse.ArgumentParser(description="InvDN_TIPC_train")
parser.add_argument("--batchSize", type=int, default=14, help="Training batch size")
parser.add_argument("--iter", type=int, default=600000, help="Number of training iterations")
parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
parser.add_argument("--data_dir", type=str, default="SIDD_mini/train_mini/", help="path of train dataset")
parser.add_argument("--val_dir", type=str, default="SIDD_mini/val_mini/", help="path of val dataset")
parser.add_argument("--log_dir", type=str, default="output", help="path of save results")
parser.add_argument("--print_freq", type=int, default=2000, help="Training print frequency")
parser.add_argument("--scale", type=int, default=4, help="scale")
parser.add_argument("--gt_size", type=int, default=144, help="crop size for training")

opt = parser.parse_args()

def main():
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    print("nranks: ", nranks)

    ######### Set Seeds ###########
    random.seed(10)
    np.random.seed(10)
    paddle.seed(10)


    model_dir = os.path.join(opt.log_dir, 'models')
    log_dir = os.path.join(opt.log_dir, 'log')

    if local_rank == 0:
        dir_utils.mkdir(model_dir)
        dir_utils.mkdir(log_dir)

    ######### Model ###########
    scale = opt.scale
    down_num = int(math.log(scale, 2))

    model = InvNet(channel_in=3, channel_out=3, subnet_constructor=constructor, block_num=[8, 8], down_num=down_num)

    model.train()

    ######### Scheduler ###########
    new_lr = opt.lr

    scheduler = optim.lr.MultiStepDecay(learning_rate=new_lr, milestones=[100000, 200000, 300000, 400000, 500000], gamma=0.5)
    clip_grad_norm = nn.ClipGradByNorm(10)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-8, beta1=0.9, beta2=0.99, grad_clip=clip_grad_norm)

    ######### Loss ###########
    Reconstruction_forw = ReconstructionLoss(losstype="l2")
    Reconstruction_back = ReconstructionLoss(losstype="l1")
    # Rec_Forw_grad = Gradient_Loss()
    Rec_back_grad = Gradient_Loss()
    # Rec_forw_SSIM = SSIM_Loss()
    Rec_back_SSIM = SSIM_Loss()

    ######### DataLoaders ###########

    alternative_opt = {
        "train_dir":opt.data_dir,
        "val_dir":opt.val_dir,
        "scale":opt.scale,
        "gt_size":opt.gt_size
    }
    train_dataset = LQGTRNDataset(opt=None, is_train=True, alternative_opt=alternative_opt)
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=opt.batchSize, shuffle=True, drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=10)

    val_dataset = LQGTRNDataset(opt=None,  is_train=False, alternative_opt=alternative_opt)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=10, drop_last=False)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        ddp_model = paddle.distributed.fleet.distributed_model(model)
    
    print('------------------------------------------------------------------------------')
    print("==> Start Training ")
    print('------------------------------------------------------------------------------')

    with LogWriter(logdir=log_dir) as writer:
        step = 0
        best_psnr = 0
        best_iter = 0

        eval_now = 1000

        print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

        current_iter = 0
        total_iters = opt.iter
        
        while current_iter <= total_iters:
            epoch_start_time = time.time()
            for data in train_loader:
                current_iter += 1
                if current_iter > total_iters:
                    break
                gt,noisy,lq = data['GT'], data['Noisy'], data['LQ']
                #forward
                if nranks > 1:
                    output = ddp_model(x=noisy)
                else:
                    output = model(x=noisy)
                lq = lq.detach()
                l_forw_fit = 16.0 * Reconstruction_forw(output[:, :3, :, :], lq)
                #backward
                y_ = paddle.concat((output[:, :3, :, :], 1 * gaussian_batch(output[:, 3:, :, :].shape)), axis=1)
                if nranks > 1:
                    x_samples = ddp_model(x=y_, rev=True)
                else:
                    x_samples = model(x=y_, rev=True)
                x_samples_image = x_samples[:, :3, :, :]
                l_back_rec = Reconstruction_back(gt, x_samples_image)
                l_grad_back_rec = 0.1*Rec_back_grad(gt, x_samples_image)
                l_back_SSIM = Rec_back_SSIM(gt, x_samples_image).mean()

                # l_back_rec = l_back_rec + l_grad_back_rec + l_back_SSIM

                l_total = l_forw_fit + l_back_rec
                optimizer.clear_grad()
                l_total.backward()
                optimizer.step()

                if current_iter % opt.print_freq == 0 and local_rank == 0:
                    step += 1
                    writer.add_scalar(tag='loss', value=l_total.item(), step=step)
                    writer.add_scalar(tag='lr', value=optimizer.get_lr(), step=step)
                    print("Iter: {}\tTime: {:.4f}s\tLoss: {:.4f}\tLR: {:.6f}".format(current_iter, time.time() - epoch_start_time, l_total.item(), optimizer.get_lr()))
                    
                # validation
                if current_iter % eval_now == 0 and local_rank == 0:
                    valid_start_time = time.time()
                    model.eval()
                    with paddle.no_grad():
                        psnr_val_rgb = []
                        ssim_val_rgb = []
                        for data_val in val_loader:
                            gt,noisy = data_val['GT'], data_val['Noisy']
                            #forward
                            if nranks > 1:
                                output = ddp_model(x=noisy)
                            else:
                                output = model(x=noisy)
                            #backward
                            y_forw = paddle.concat((output[:, :3, :, :], 1 * gaussian_batch(output[:, 3:, :, :].shape)), axis=1)
                            if nranks > 1:
                                fake_H = ddp_model(x=y_forw, rev=True)
                            else:
                                fake_H = model(x=y_forw, rev=True)

                            crop_size = opt.scale
                            gt = gt[:,:,crop_size:-crop_size, crop_size:-crop_size]
                            fake_H = fake_H[:,:,crop_size:-crop_size, crop_size:-crop_size]
                            psnr_val_rgb.append(img_utils.batch_PSNR(fake_H, gt, 1.))
                            ssim_val_rgb.append(img_utils.batch_SSIM(fake_H, gt))

                        psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
                        ssim_val_rgb = sum(ssim_val_rgb) / len(ssim_val_rgb)

                        if psnr_val_rgb > best_psnr:
                            best_psnr = psnr_val_rgb
                            best_iter = current_iter
                            paddle.save({'iter': current_iter,
                                        'state_dict': model.state_dict(),
                                        'optimizer': optimizer.state_dict()
                                        }, os.path.join(model_dir, "model_best.pdparams"))

                        print(
                            "[iter %d\t TIME: %.4fs\t PSNR SIDD: %.4f\t SSIM SIDD: %.4f\t] ----  [best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                                current_iter, time.time() - valid_start_time, psnr_val_rgb, ssim_val_rgb, best_iter, best_psnr))
                    

                    writer.add_scalar(tag='PSNR_val', value=psnr_val_rgb, step=step)
                    writer.add_scalar(tag='SSIM_val', value=ssim_val_rgb, step=step)

                    model.train()
                
                # update lr
                if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                    lr_sche = optimizer.user_defined_optimizer._learning_rate
                else:
                    lr_sche = optimizer._learning_rate
                if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
                    lr_sche.step()

            if local_rank == 0:
                paddle.save({'iter': current_iter,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_latest.pdparams"))

if __name__ == '__main__':
    main()