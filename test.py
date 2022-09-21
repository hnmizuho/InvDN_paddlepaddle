import paddle
from paddle.io import DataLoader

if __name__ == '__main__':
    # #数据加载对齐  评估指标对齐
    # from config import Config

    # opt = Config('training_1card.yml')
    # print(opt)
    # from dataloaders.LQGTRN_dataset import LQGTRNDataset

    # train_dataset = LQGTRNDataset(opt)
    # # batch_sampler = paddle.io.DistributedBatchSampler(
    # #     train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False)
    # batch_sampler = paddle.io.DistributedBatchSampler(
    #     train_dataset, batch_size=1, shuffle=True, drop_last=False)
    # train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=8)

    # val_dataset = LQGTRNDataset(opt)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

    # for data in train_loader:
    #     import cv2
    #     # img = data['GT'][0].transpose([1,2,0]).numpy()*255.
    #     # cv2.imwrite("gt.png",img)
    #     n = data['Noisy'][0].transpose([1,2,0]).numpy()*255.
    #     cv2.imwrite("noisy.png",n)
    #     n = data['LQ'][0].transpose([1,2,0]).numpy()*255.
    #     cv2.imwrite("lq.png",n)

    #     # from utils.img_utils import batch_PSNR, batch_SSIM
    #     # psnr = batch_PSNR(data['Noisy'], data['GT'], 1.)
    #     # print(psnr)
    #     # ssim = batch_SSIM(data['Noisy'], data['GT'])
    #     # print(ssim)

    #     break

    import paddle.distributed as dist
    print("S")
    print("world ",paddle.distributed.get_world_size()) #4
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    print("nranks: ", nranks)
    print("local",local_rank)
    from dataloaders.LQGTRN_dataset import LQGTRNDataset
    from config import Config
    import os
    opt = Config('training_4cards.yml') 
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    train_dataset = LQGTRNDataset(opt, is_train=True)
    batch_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler, num_workers=8)
    print("len ",len(train_loader)) # 当yml里bt=14时，这儿为572 for 4cards，2286 for 1card，
    current_iter = 0
    for data in train_loader:
        print(os.system("nvidia-smi"))
        print(data["LQ"].shape) # 当yml里bt=14时，这儿为 14 3 36 36 for 4cards，14 3 36 36 for 1card，
        break
        # current_iter += 1 
    print("current_iter ",current_iter) # == len(train_loader)，即当yml里bt=14时，这儿为572 for 4cards，2286 for 1card