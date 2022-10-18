import os
import torch
# from astropy.io import fits
import numpy as np
from models.network_unet import UNetAuthor, UNet, UNetRes, UNetRes2, UNetRes3, UNetRes4
from models.basic_models import simple_CNN as dncnn
# from models.models_helper import get_model
# from lightning_denoiser import GradMatch
import scipy
from skimage import io
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
# parser.add_argument("--taskname", type=str, default=0)
opt = parser.parse_args()

def denoise(start):
    print('Loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device = {}'.format(device))
    # model = UNetRes4(nc=[32, 64, 128, 256], nb=2)
    model = dncnn(n_ch_in=1, n_ch_out=1, n_ch=64, nl_type='relu', depth=20, bn=False, clip='relu')
    network_pth = '/work/sc004/sc004/tc1213/Prox-Pnp2/GS_denoising/tmp_ckpts/dncnn_nobn_nch_1_relu_bio_pre_l1_lr_1e-05_800_0.5_epsilon_-0.05_ljr_1e-05_noise_0.00032_seed_0_epoch_1000_dr_205.0.pth'
    checkpoint = torch.load(network_pth, map_location=device)      
    new_state_dict = {}
    # for key, val in checkpoint['state_dict'].items():
    #     new_key = key.split('student_grad.model.')[-1]
    #     new_state_dict[new_key] = val
    model.load_state_dict(checkpoint, strict=True)
    print('Network loaded.')

    path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff/'
    files = glob.glob(path + '*.tif')
    end = start+250 if start+250 <= len(files) else len(files)
    files = files[start:end]

    # taskname = '3.2e-4_blurred_0.25'
    taskname = '3.2e-e-4_denoised_dncnn'
    
    progress_check = [0.25, 0.5, 0.75, 1]

    for i, file in enumerate(files):
        img = io.imread(file) 
        img_tmp = img + np.random.normal(size=img.shape) * 0.00032
        for _ in range(50):
            img_denoised = model.forward(torch.Tensor(img_tmp).unsqueeze(0))
        outpth = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff_' + taskname
        if i == 0:
            if not os.path.exists(outpth):
                os.makedirs(outpth, exist_ok=True)
        # io.imsave(outpth + '/' + os.path.basename(file), img_blurred)
        img_denoised = img_denoised.squeeze(0).detach().numpy()
        img_combined = np.column_stack((img, img_denoised))
        io.imsave(outpth + '/' + os.path.basename(file), img_combined)
        if np.round(i/len(files),2) in progress_check:
            print("{}% done;".format(str(np.round(i/len(files),2)*100)))
            
if __name__ == '__main__':
    denoise(opt.start)