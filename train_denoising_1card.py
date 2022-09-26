import os
from config import Config

opt = Config('training_1card.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

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

def main():
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    print("nranks: ", nranks)

    ######### Set Seeds ###########
    random.seed(10)
    np.random.seed(10)
    paddle.seed(10)

    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models-final2', session)
    log_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'logs-final2', session)

    if local_rank == 0:
        dir_utils.mkdir(model_dir)
        dir_utils.mkdir(log_dir)

    if local_rank == 0:
        dir_utils.setup_logger(logger_name = 'base', root = log_dir, phase = 'train_', level=logging.INFO,screen=True, tofile=True)
        dir_utils.setup_logger(logger_name = 'val', root = log_dir, phase = 'val_', level=logging.INFO,screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger_val = logging.getLogger('val')
    else:
        dir_utils.setup_logger(logger_name='base', root=log_dir, phase='train', level=logging.INFO, screen=True, tofile=False)
        logger = logging.getLogger('base')
    ######### Model ###########
    scale = opt.TRAINING.SCALE
    down_num = int(math.log(scale, 2))

    model = InvNet(channel_in=3, channel_out=3, subnet_constructor=constructor, block_num=[8, 8], down_num=down_num)

    model.train()

    ######### Scheduler ###########
    new_lr = opt.OPTIM.LR_INITIAL

    scheduler = optim.lr.MultiStepDecay(learning_rate=new_lr, milestones=opt.OPTIM.MILESTONES, gamma=opt.OPTIM.GAMMA)
    wd_G = opt.OPTIM.WEIGHT_DECAY_G if opt.OPTIM.WEIGHT_DECAY_G else 0
    clip_grad_norm = nn.ClipGradByNorm(opt.OPTIM.GRADIENT_CLIPPING)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=wd_G, beta1=opt.OPTIM.BETA1, beta2=opt.OPTIM.BETA2, grad_clip=clip_grad_norm)

    ######### Resume ###########
    #使用torch的初始化
    ckpt = paddle.load("./pretrained/default_invdn.pdparams")
    model.set_dict(ckpt)

    if opt.TRAINING.RESUME:
        ckpt = paddle.load(opt.TRAINING.RESUME_PATH)
        model.set_state_dict(ckpt['state_dict'])
        optimizer.set_state_dict(ckpt['optimizer'])

        resume_iter = ckpt['iter']
        resume_step = resume_iter // opt.TRAINING.PRINT_FREQ

    ######### Loss ###########
    Reconstruction_forw = ReconstructionLoss(losstype=opt.OPTIM.PIXEL_CRITERION_FORW)
    Reconstruction_back = ReconstructionLoss(losstype=opt.OPTIM.PIXEL_CRITERION_BACK)
    # Rec_Forw_grad = Gradient_Loss()
    Rec_back_grad = Gradient_Loss()
    # Rec_forw_SSIM = SSIM_Loss()
    Rec_back_SSIM = SSIM_Loss()

    ######### DataLoaders ###########

    train_dataset = LQGTRNDataset(opt, is_train=True)
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=10)

    val_dataset = LQGTRNDataset(opt,  is_train=False)
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
        step = resume_step if opt.TRAINING.RESUME else 0
        best_psnr = 0
        best_iter = 0

        eval_now = 1000
        # eval_now = len(train_loader)
        # eval_now = 800
        print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

        current_iter = resume_iter if opt.TRAINING.RESUME else 0
        total_iters = opt.OPTIM.NUM_ITERS
        
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
                l_forw_fit = opt.OPTIM.LAMBDA_FIT_FORW * Reconstruction_forw(output[:, :3, :, :], lq)
                #backward
                gaussian_scale = 1 
                y_ = paddle.concat((output[:, :3, :, :], gaussian_scale * gaussian_batch(output[:, 3:, :, :].shape)), axis=1)
                if nranks > 1:
                    x_samples = ddp_model(x=y_, rev=True)
                else:
                    x_samples = model(x=y_, rev=True)
                x_samples_image = x_samples[:, :3, :, :] 
                l_back_rec = opt.OPTIM.LAMBDA_REC_BACK * Reconstruction_back(gt, x_samples_image)
                l_grad_back_rec = 0.1*opt.OPTIM.LAMBDA_REC_BACK * Rec_back_grad(gt, x_samples_image)
                l_back_SSIM = opt.OPTIM.LAMBDA_REC_BACK * Rec_back_SSIM(gt, x_samples_image).mean()

                # l_back_rec = l_back_rec + l_grad_back_rec + l_back_SSIM

                l_total = l_forw_fit + l_back_rec
                optimizer.clear_grad()
                l_total.backward()
                optimizer.step()

                if current_iter % opt.TRAINING.PRINT_FREQ == 0 and local_rank == 0:
                    step += 1
                    writer.add_scalar(tag='loss', value=l_total.item(), step=step)
                    writer.add_scalar(tag='lr', value=optimizer.get_lr(), step=step)
                    # print("Iter: {}\tTime: {:.4f}s\tLoss: {:.4f}\tLR: {:.6f}".format(current_iter, time.time() - epoch_start_time, l_total.item(), optimizer.get_lr()))
                    logger.info("Iter: {}\tTime: {:.4f}s\tLoss: {:.4f}\tLR: {:.6f}".format(current_iter, time.time() - epoch_start_time, l_total.item(), optimizer.get_lr()))
                    
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
                            gaussian_scale = 1
                            y_forw = paddle.concat((output[:, :3, :, :], gaussian_scale * gaussian_batch(output[:, 3:, :, :].shape)), axis=1)
                            if nranks > 1:
                                fake_H = ddp_model(x=y_forw, rev=True)
                            else:
                                fake_H = model(x=y_forw, rev=True)

                            crop_size = opt.TRAINING.SCALE
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

                        # print(
                        #     "[iter %d\t TIME: %.4fs\t PSNR SIDD: %.4f\t SSIM SIDD: %.4f\t] ----  [best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                        #         current_iter, time.time() - valid_start_time, psnr_val_rgb, ssim_val_rgb, best_iter, best_psnr))
                        logger_val.info(
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