import os
import torch
import numpy as np
from models.network_unet import UNetAuthor, UNet, UNetRes, UNetRes2, UNetRes3, UNetRes4
from scipy.ndimage import gaussian_filter
from skimage import io
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--sigma", type=float, default=0.25)
opt = parser.parse_args()

def blurring(start, sigma):

    path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff/'
    files = glob.glob(path + '*.tif')
    end = start+250 if start+250 <= len(files) else len(files)
    files = files[start:end]

    taskname = 'blurred_{}'.format(str(sigma))

    for i, file in enumerate(files):
        img = io.imread(file) 
        img_blurred = gaussian_filter(img, sigma=sigma)
        # img_blurred[img_blurred > 1] = 1
        img_blurred /= np.max(img_blurred)
        img_blurred[img_blurred < 0] = 0
        outpth = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff_' + taskname
        if i == 0:
            if not os.path.exists(outpth):
                os.makedirs(outpth, exist_ok=True)
        io.imsave(outpth + '/' + os.path.basename(file), img_blurred)
            
if __name__ == '__main__':
    blurring(opt.start, opt.sigma)