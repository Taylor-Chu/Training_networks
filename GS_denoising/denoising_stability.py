from operator import truediv
import os
import torch
from astropy.io import fits
import numpy as np
from models.network_unet import UNetRes3
# from lightning_denoiser import GradMatch
from argparse import ArgumentParser

'''
Test stability of a denoiser by iteratively denoising a given noisy image
'''

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--network_pth', type=str)
    parser.add_argument('--noise', type=float)
    parser.add_argument('--iter', type=int, default=4000)
    hparams = parser.parse_args()

    with torch.no_grad():   
        gt_pth = '/work/sc004/sc004/tc1213/data/data_groundtruth/3c353.fits'
        img_gt = fits.getdata(gt_pth)
        print('Loaded groundtruth image.')

        print('Loading model...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device = {}'.format(device))
        model = UNetRes3(nc=[32, 64, 128, 256], nb=2)
        checkpoint = torch.load(hparams.network_pth, map_location=device)            
        new_state_dict = {}
        for key, val in checkpoint['state_dict'].items():
            new_key = key.split('student_grad.model.')[-1]
            new_state_dict[new_key] = val
        model.load_state_dict(new_state_dict, strict=True)
        # model.load_state_dict(checkpoint['state_dict'])
        print('Network loaded.')

        img_noisy = np.random.normal(0, hparams.noise, img_gt.shape) + img_gt
        x = torch.from_numpy(img_noisy).unsqueeze(0).unsqueeze(0).float()
        result_pth = '/work/sc004/sc004/tc1213/results/bio/denoiser/dncnn/bio2/' + str(hparams.noise)
        os.makedirs(result_pth, exist_ok = True)
        fits.writeto(result_pth + '/noisy.fits', x.detach().numpy(), overwrite=True)
        print('Created noisy image and saved at {}.'.format(result_pth))

        for i in range(hparams.iter):
            x = model.forward(x)
            if (i+1)%100 == 0:
                print(i)
                fits.writeto(result_pth + '/result_{}.fits'.format(str(i+1)), x.detach().numpy(), overwrite = truediv)
            

        print('Finished denoising and results saved at {}.'.format(result_pth))

