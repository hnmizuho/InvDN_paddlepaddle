import os
import scipy
import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

tar = './SIDD_Valid_Srgb_Patches_256/valid'
noisy_patchDir = os.path.join(tar, 'Noisy')
clean_patchDir = os.path.join(tar, 'GT')
if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

files_gt = scipy.io.loadmat('ValidationGtBlocksSrgb.mat')
imgArray_gt = files_gt['ValidationGtBlocksSrgb']
files_noisy = scipy.io.loadmat('ValidationNoisyBlocksSrgb.mat')
imgArray_noisy = files_noisy['ValidationNoisyBlocksSrgb']

nImages = 40
nBlocks = imgArray_gt.shape[1]

def save_files(i):
    Inoisy = imgArray_noisy[i].astype(np.float32)
    Iclean = imgArray_gt[i].astype(np.float32)

    for j in range(nBlocks):

        noisy_patch = Inoisy[j]
        clean_patch = Iclean[j]
        noisy_patch = cv2.cvtColor(noisy_patch, cv2.COLOR_BGR2RGB)
        clean_patch = cv2.cvtColor(clean_patch, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(clean_patchDir, '%d_%d.PNG'%(i, j)), clean_patch)
        cv2.imwrite(os.path.join(noisy_patchDir, '%d_%d.PNG'%(i, j)), noisy_patch)
    print('[%d/%d] is done\n' % (i+1, 40))

NUM_CORES = 10
Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(nImages)))