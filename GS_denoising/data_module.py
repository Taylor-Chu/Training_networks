import pytorch_lightning as pl
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
import glob
from skimage import io
from astropy.io import fits
import numpy as np
from torch.utils import data as D
from torch.utils.data import ConcatDataset
import torch
import torchvision
from scipy.optimize import fsolve
import gryds
import random

def random_rot_resize_crop(im,
                            patch_size=(64, 64),
                            zoom_fact=None,
                            z_max=1.33,
                            z_min=0.33,
                            seed=None):
    im_mod = np.copy(im)

    if seed is not None:
        random.seed(seed)

    if zoom_fact is None:
        zoom_fact = (z_max - z_min) * np.random.random() + z_min

    affine = gryds.AffineTransformation(
        ndim=2,
        angles=[np.pi * random.uniform(-1, 1)],
        center=[random.uniform(0.2, 0.8),
                random.uniform(0.2, 0.8)],
        scaling=[1 / zoom_fact, 1 / zoom_fact])

    m, M = im_mod.min(), im_mod.max()
    # Necessary because of interpolation errors

    interpolator_im = gryds.Interpolator(im_mod, order=3, mode='mirror')
    transformed_im = interpolator_im.transform(affine)
    transformed_im = np.clip(transformed_im, m, M)

    im_size = transformed_im.shape
    rx, ry = np.random.randint(0, im_size[0] - patch_size[0]), np.random.randint(0, im_size[1] - patch_size[1])

    cropped_im = transformed_im[rx:rx + patch_size[0], ry:ry + patch_size[1]]

    return cropped_im

class CustomDataset(D.Dataset):
    """Loads the fits image dataset
    To avoid problematic samples return None as in
    https://github.com/msamogh/nonechucks/blob/master/nonechucks/dataset.py
    """
    def __init__(self, hparams, path_data,
                transform=None,
                reduction_size=None,                 
                patch_size=(64, 64),
                blurred=False):
        """
        Intialize the dataset
        path_data = list of paths
        """
        self.hparams = hparams
        self.path_data = path_data
        self.transform = transform
        self.patch_size = patch_size
        self.filenames_c = []
        clean_components_list = []

        clean_components_list = sorted(glob.glob(os.path.join(self.path_data, '*.*')))
        for fn in clean_components_list:
            self.filenames_c.append(fn)

        if reduction_size is not None:
            self.filenames_c = self.filenames_c[:reduction_size]

        self.len = len(self.filenames_c)
        self.blurred = blurred
        if self.blurred:
            self.transf_blur = torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0))
            
        self.len = len(self.filenames_c)

    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        """
        im_path = self.filenames_c[index]
        _, file_extension = os.path.splitext(im_path)
        self.file_format = '*' + file_extension
        image_true = fits.getdata(self.filenames_c[index]) if 'fits' in self.file_format else io.imread(self.filenames_c[index])
        # if self.hparams.dataset_name == 'bio':
        #     image_true = io.imread(self.filenames_c[index])
        # elif self.hparams.dataset_name == 'Astro':
        #     image_true = fits.getdata(self.filenames_c[index])
        image_name = os.path.basename(im_path)[:-(len(self.file_format)-1)]

        sel_seed = None if not self.hparams.seed else index

        if self.transform is not None:
            image_true = random_rot_resize_crop(image_true,
                                                patch_size=self.patch_size,
                                                seed=sel_seed)

        if 'bio' in self.hparams.dataset_name:
            image_true = (self.hparams.dr ** image_true - 1.) / self.hparams.dr
        elif self.hparams.dataset_name == 'Astro':
            image_true = np.log10(1000. * image_true + 1) / np.log10(1000.)
            image_true = (self.hparams.dr ** image_true - 1.) / self.hparams.dr
        elif self.hparams.dataset_name == 'bio+Astro':
            if 'bio' in im_path:
                image_true = (self.hparams.dr_bio ** image_true - 1.) / self.hparams.dr_bio
            elif 'astro' in im_path:
                image_true = np.log10(1000. * image_true + 1) / np.log10(1000.)
                image_true = (self.hparams.dr_astro ** image_true - 1.) / self.hparams.dr_astro

        image_true = torch.from_numpy(image_true).unsqueeze(0)
        # print(image_true.size())

        return [
            image_true,
            image_name
        ]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len



class DataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        # self.hparams.update(dict(hparams))
        self.save_hyperparameters(hparams)

        if self.hparams.dataset_name == 'bio':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff'
            # self.dataset_path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff'
            # self.valset_path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff'
        elif self.hparams.dataset_name == 'bio_blurred_0.25':
            self.dataset_path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff_blurred_0.25'
            self.valset_path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff_blurred_0.25'
        elif self.hparams.dataset_name == 'bio_blurred_1.0':
            self.dataset_path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff_blurred_1.0'
            self.valset_path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff_blurred_1.0'
        elif self.hparams.dataset_name == 'bio_blurred_0.5':
            self.dataset_path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff_blurred_0.5'
            self.valset_path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff_blurred_0.5'
        elif self.hparams.dataset_name == 'bio_blurred_0.75':
            self.dataset_path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff_blurred_0.75'
            self.valset_path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff_blurred_0.75'
        elif self.hparams.dataset_name == 'bio_blurred_1.25':
            self.dataset_path = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff_blurred_1.25'
            self.valset_path = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff_blurred_1.25'
        elif self.hparams.dataset_name == 'bio_m87':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff_m87'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff_m87'
        elif self.hparams.dataset_name == 'bio_b35':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff_dr_35'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff_dr_35'
        elif self.hparams.dataset_name == 'bio_b53':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff_dr_53'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff_dr_53'
        elif self.hparams.dataset_name == 'bio_b90':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff_dr_90'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff_dr_90'
        elif self.hparams.dataset_name == 'astro_b35':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/astro/trainingset_fullnumpy_dr_35'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/astro/validationset_fullnumpy_dr_35'
        elif self.hparams.dataset_name == 'astro_b53':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/astro/trainingset_fullnumpy_dr_53'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/astro/validationset_fullnumpy_dr_53'
        elif self.hparams.dataset_name == 'astro_b90':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/astro/trainingset_fullnumpy_dr_90'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/astro/validationset_fullnumpy_dr_90'
            # self.dataset_path = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff_m87'
            # self.valset_path = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff_m87'
        elif self.hparams.dataset_name == 'Astro':
            self.dataset_path = '/scratch/space1/sc004/tc1213/data/astro/trainingset_fullnumpy'
            self.valset_path = '/scratch/space1/sc004/tc1213/data/astro/validationset_fullnumpy'
        elif self.hparams.dataset_name == 'bio+Astro':
            # self.dataset_path = '/mnt/shared/home/ps2phd2/database/combined/trainingset_1'
            # self.valset_path = '/mnt/shared/home/ps2phd2/database/combined/validationset_1'
            # self.dataset_path1 = '/scratch/space1/sc004/tc1213/data/bio_images/trainingset_true_tiff_scaled'
            # self.valset_path1 = '/scratch/space1/sc004/tc1213/data/bio_images/testingset_true_tiff_scaled'
            self.dataset_path1 = '/work/sc004/sc004/tc1213/data/bio_images/trainingset_true_tiff'
            self.valset_path1 = '/work/sc004/sc004/tc1213/data/bio_images/testingset_true_tiff'
            self.dataset_path2 = '/work/sc004/sc004/tc1213/data/astro/trainingset_fullnumpy'
            self.valset_path2 = '/work/sc004/sc004/tc1213/data/astro/validationset_fullnumpy'
        # self.testset_pth = '/mnt/shared/home/ps2phd2/database/data_groundtruth'

    def setup(self, stage=None):
        # Unet 2 - 10, cannot learn with > 20
        if 'UNet' in self.hparams.model_name:
            if self.hparams.jacobian_loss_weight > 0:
                self.hparams.batch_size_train = 10
            else:
                self.hparams.batch_size_train = 10
        else:
            if self.hparams.jacobian_loss_weight > 0:
                self.hparams.batch_size_train = 10
            else:
                self.hparams.batch_size_train = 50
        print('batch size = {}'.format(self.hparams.batch_size_train))

        # if self.hparams.jacobian_loss_weight > 0:
        if 'unet' in self.hparams.model_name.lower():
            patch_size = 192
        else:
            # if self.hparams.jacobian_loss_weight > 0:
            patch_size = 46
        # patch_size = 46 if 'unet' not in self.hparams.model_name.lower() else 192
        # else:
        #     patch_size = 46 if 'unet' not in self.hparams.model_name.lower() else 144
        # increase to cover receptive field
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            if self.hparams.dataset_name == 'bio+Astro':
                dataset_train1 = CustomDataset(self.hparams, 
                                                    path_data = self.dataset_path1,
                                                    transform=True,
                                                    blurred=False,
                                                    patch_size = (patch_size, patch_size))
                dataset_train2 = CustomDataset(self.hparams, 
                                                    path_data = self.dataset_path2,
                                                    transform=True,
                                                    blurred=False,
                                                    patch_size = (patch_size, patch_size))
                self.dataset_train = ConcatDataset([dataset_train1, dataset_train2])
                # self.dataset_train1 = dataset_train1
                # self.dataset_train2 = dataset_train2

                dataset_val1 = CustomDataset(self.hparams, 
                                                path_data = self.valset_path1,
                                                transform=True,
                                                blurred=False,
                                                patch_size = (patch_size, patch_size))
                dataset_val2 = CustomDataset(self.hparams, 
                                                path_data = self.valset_path2,
                                                transform=True,
                                                blurred=False,
                                                patch_size = (patch_size, patch_size))
                self.dataset_val = ConcatDataset([dataset_val1, dataset_val2])
                # self.dataset_val1 = dataset_val1
                # self.dataset_val2 = dataset_val2
            else:
                self.dataset_train = CustomDataset(self.hparams, 
                                                    path_data = self.dataset_path,
                                                    transform=True,
                                                    blurred=False,
                                                    patch_size = (patch_size, patch_size))

                self.dataset_val = CustomDataset(self.hparams, 
                                                path_data = self.valset_path,
                                                transform=True,
                                                blurred=False,
                                                patch_size = (patch_size, patch_size))
            

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            if self.hparams.dataset_name == 'bio+Astro':
                dataset_test1 = CustomDataset(self.hparams, 
                                path_data = self.valset_path1,
                                transform=True,
                                blurred=False,
                                patch_size = (patch_size, patch_size))
                dataset_test2 = CustomDataset(self.hparams, 
                                path_data = self.valset_path2,
                                transform=True,
                                blurred=False,
                                patch_size = (patch_size, patch_size))
                # self.dataset_test1 = dataset_test1
                # self.dataset_test2 = dataset_test2
                self.dataset_test = ConcatDataset([dataset_test1, dataset_test2])
            else:
                self.dataset_test = CustomDataset(self.hparams, 
                                    path_data = self.valset_path,
                                    transform=True,
                                    blurred=False,
                                    patch_size = (patch_size, patch_size))

            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.hparams.batch_size_train,
                          shuffle=self.hparams.train_shuffle,
                          num_workers=self.hparams.num_workers_train,
                          drop_last=True,
                          pin_memory=True)
        # return [DataLoader(self.dataset_train1,
        #                   batch_size=self.hparams.batch_size_train,
        #                   shuffle=self.hparams.train_shuffle,
        #                   num_workers=self.hparams.num_workers_train,
        #                   drop_last=True,
        #                   pin_memory=True),
        #         DataLoader(self.dataset_train2,
        #                   batch_size=self.hparams.batch_size_train,
        #                   shuffle=self.hparams.train_shuffle,
        #                   num_workers=self.hparams.num_workers_train,
        #                   drop_last=True,
        #                   pin_memory=True)]

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.hparams.batch_size_test,
                          shuffle=False,
                          num_workers=self.hparams.num_workers_test,
                          drop_last=True,
                          pin_memory=False)
        # return [DataLoader(self.dataset_val1,
        #                   batch_size=self.hparams.batch_size_test,
        #                   shuffle=False,
        #                   num_workers=self.hparams.num_workers_test,
        #                   drop_last=True,
        #                   pin_memory=True),
        #         DataLoader(self.dataset_val2,
        #                   batch_size=self.hparams.batch_size_test,
        #                   shuffle=False,
        #                   num_workers=self.hparams.num_workers_test,
        #                   drop_last=True,
        #                   pin_memory=True)]

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.hparams.batch_size_train,
                          shuffle=False,
                          num_workers=self.hparams.num_workers_train,
                          drop_last=False,
                          pin_memory=False)
        # return [DataLoader(self.dataset_test1,
        #                   batch_size=self.hparams.batch_size_test,
        #                   shuffle=False,
        #                   num_workers=self.hparams.num_workers_test,
        #                   drop_last=True,
        #                   pin_memory=True),
        #         DataLoader(self.dataset_test2,
        #                   batch_size=self.hparams.batch_size_test,
        #                   shuffle=False,
        #                   num_workers=self.hparams.num_workers_test,
        #                   drop_last=True,
        #                   pin_memory=True)]

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_name', type=str, default='bio')
        # parser.add_argument('--train_patch_size', type=int, default=50)
        # parser.add_argument('--test_patch_size', type=int, default=50)
        parser.add_argument('--train_shuffle', dest='train_shuffle', action='store_true')
        parser.add_argument('--no-train_shuffle', dest='train_shuffle', action='store_false')
        parser.set_defaults(train_shuffle=True)
        # parser.add_argument('--no_test_resize', dest='test_resize', action='store_false')
        # parser.set_defaults(test_resize=True)
        parser.add_argument('--num_workers_train',type=int, default=6)
        parser.add_argument('--num_workers_test', type=int, default=6)
        parser.add_argument('--batch_size_train', type=int, default=100)
        parser.add_argument('--batch_size_test', type=int, default=10)
        # parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--dr', type=int, default=1e3)
        parser.add_argument('--dr_bio', type=int, default=1e3)
        parser.add_argument('--dr_astro', type=int, default=1e3)
        parser.add_argument('--log', type=int, default=1)
        parser.add_argument('--exp', type=int, default=1)
        return parser