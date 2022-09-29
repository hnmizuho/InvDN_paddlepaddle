import random
import numpy as np
import cv2
import paddle
from paddle.io import Dataset
from utils import util
import os.path as osp
import os


# 这一版本在val时不生成LQ。训练时即时保存LQ
class LQGTRNDataset(Dataset):
    '''
    Read LQ (Low Quality, here is LR), GT and noisy image pairs.
    If only GT and noisy images are provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    # opt支持读取yml，alternative_opt支持读取字典
    def __init__(self, opt, is_train, alternative_opt=None):
        super(LQGTRNDataset, self).__init__()
        self.opt = opt
        self.alternative_opt = alternative_opt
        self.is_train = is_train
        self.paths_LQ, self.paths_GT, self.paths_Noisy = None, None, None
        self.sizes_LQ, self.sizes_GT, self.sizes_Noisy = None, None, None
        self.LQ_env, self.GT_env, self.Noisy_env = None, None, None

        self.data_type = "img"

        if self.is_train:
            if opt:
                dataroot_gt = osp.join(opt.TRAINING.TRAIN_DIR, "GT")
                dataroot_noisy = osp.join(opt.TRAINING.TRAIN_DIR, "Noisy")
                dataroot_lq = osp.join(opt.TRAINING.TRAIN_DIR, "LQ")
            elif alternative_opt:
                dataroot_gt = osp.join(alternative_opt["train_dir"], "GT")
                dataroot_noisy = osp.join(alternative_opt["train_dir"], "Noisy")
                dataroot_lq = osp.join(alternative_opt["train_dir"], "LQ")          
        else:
            if opt:
                dataroot_gt = osp.join(opt.TRAINING.VAL_DIR, "GT")
                dataroot_noisy = osp.join(opt.TRAINING.VAL_DIR, "Noisy")
                dataroot_lq = None
            elif alternative_opt:
                dataroot_gt = osp.join(alternative_opt["val_dir"], "GT")
                dataroot_noisy = osp.join(alternative_opt["val_dir"], "Noisy")
                dataroot_lq = None

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, dataroot_gt)
        self.paths_Noisy, self.sizes_Noisy = util.get_image_paths(self.data_type, dataroot_noisy)
        if self.is_train and not os.path.exists(dataroot_lq):
            os.makedirs(dataroot_lq)     
        else:
            self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, dataroot_lq)
 
        assert self.paths_GT, 'Error: GT path is empty.'
        assert self.paths_Noisy, 'Error: Noisy path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

        if opt:
            self.scale = opt.TRAINING.SCALE
            self.GT_size = opt.TRAINING.GT_SIZE
        elif alternative_opt:
            self.scale = alternative_opt["scale"]
            self.GT_size = alternative_opt["gt_size"]          

    def __getitem__(self, index):
        GT_path, Noisy_path, LQ_path = None, None, None

        if self.opt:
            scale = self.opt.TRAINING.SCALE
            GT_size = self.opt.TRAINING.GT_SIZE
        else:
            scale = self.alternative_opt["scale"]
            GT_size = self.alternative_opt["gt_size"]         

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)

        # modcrop in the validation / test phase
        if not self.is_train:
            img_GT = util.modcrop(img_GT, scale)

        # change color space if necessary
        img_GT = util.channel_convert(img_GT.shape[2], "RGB", [img_GT])[0]

        # get Noisy image
        Noisy_path = self.paths_Noisy[index]
        resolution = None
        img_Noisy = util.read_img(self.Noisy_env, Noisy_path, resolution)

        # modcrop in the validation / test phase
        if not self.is_train:
            img_Noisy = util.modcrop(img_Noisy, scale)

        # change color space if necessary
        img_Noisy = util.channel_convert(img_Noisy.shape[2], "RGB", [img_Noisy])[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.is_train:
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                img_Noisy = cv2.resize(np.copy(img_Noisy), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_Noisy.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)
                    img_Noisy = cv2.cvtColor(img_Noisy, cv2.COLOR_GRAY2BGR)

                H, W, _ = img_GT.shape
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True) # (512, 512, 3) --> (128, 128, 3)
                # img_LQ = cv2.resize(img_GT, (H//scale, W//scale)) # (512, 512, 3) --> (128, 128, 3)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)
                
                #save lq
                LQ_path = os.path.join('/'.join(GT_path.split('/')[:-2]+['LQ']), GT_path.split('/')[-1])
                cv2.imwrite(LQ_path, img_LQ*255.)

        if self.is_train:
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                img_Noisy = cv2.resize(np.copy(img_Noisy), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] # (128, 128, 3) --> (36, 36, 3) 比例3.555
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :] # (512, 512, 3) --> (144, 144, 3) 比例3.555
            img_Noisy = img_Noisy[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            # augmentation - flip, rotate
            img_LQ, img_GT, img_Noisy = util.augment([img_LQ, img_GT, img_Noisy], True, True)

            # change color space if necessary
            C = img_LQ.shape[0]
            img_LQ = util.channel_convert(C, "RGB", [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_Noisy = img_Noisy[:, :, [2, 1, 0]]
            if self.is_train:
                img_LQ = img_LQ[:, :, [2, 1, 0]]

        img_GT = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))), dtype="float32")
        img_Noisy = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_Noisy, (2, 0, 1))), dtype="float32")
        if self.is_train:
            img_LQ = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1))), dtype="float32")

        if LQ_path is None:
            LQ_path = GT_path
            
        if self.is_train:
            return {'LQ': img_LQ, 'Noisy':img_Noisy, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

        return {'Noisy':img_Noisy, 'GT': img_GT, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT) #32000 for train, 1280 for valid
