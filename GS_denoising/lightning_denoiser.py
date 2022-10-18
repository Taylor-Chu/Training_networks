import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim import lr_scheduler
import random
import torchmetrics
from argparse import ArgumentParser
import cv2
import torchvision
import numpy as np
from test_utils import test_mode
import matplotlib.pyplot as plt
from GS_utils import normalize_min_max
from models.network_unet import UNetAuthor, UNet, UNetRes, UNetRes2, UNetRes3, UNetRes4
from models import DNCNN
from models import wdsr
from models.FFDNET import FFDNet
from models.basic_models import simple_CNN as dncnn
import math

# from bot import send_update

from lossfunc import LogLoss, LogCoshLoss, LogL1Loss, LogLoss1, LogLoss05, LogLoss01, LogLoss001, LogLoss005
from jacobian import JacobianReg_l2
from utils.metrics_helper import snr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(
            -0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)
        
def log_img(img, log_exp=torch.tensor(1000.)):
    img_cur = torch.clone(img)
    img_cur[img_cur<0] = 0
    return torch.log10(log_exp * img_cur + torch.tensor(1.)) / torch.log10(log_exp)


class StudentGrad(pl.LightningModule):
    '''
    Standard Denoiser model
    '''
    def __init__(self, model_name, pretrained_checkpoint, act_mode, DRUNET_nb, residual_learning, check):
        super().__init__()
        self.model_name = model_name
        self.check = check
        self.residual_learning = residual_learning
        if self.model_name == 'UNet':
            self.model = UNet(in_nc = 1, out_nc = 1, nb = 2, nc=[32, 64, 128, 256])
        elif self.model_name == 'UNetAuthor':
            self.model = UNetAuthor(nc=[32, 64, 128, 256])
        elif self.model_name == 'UNetRes':
            self.model = UNetRes(nc=[32, 64, 128, 256], nb=2)
        elif self.model_name == 'UNetRes2':
            self.model = UNetRes2(nc=[32, 64, 128, 256], nb=2)
        elif self.model_name == 'UNetRes3':
            self.model = UNetRes3(nc=[32, 64, 128, 256], nb=2)
        elif self.model_name == 'UNetRes4':
            self.model = UNetRes4(nc=[32, 64, 128, 256], nb=2)
        elif self.model_name == 'DNCNN':
            # self.model = DNCNN.dncnn(nc_in=1, nc_out=1, nb=20, mode='C', act_mode='l', bias=True)
            self.model = dncnn(n_ch_in=1, n_ch_out=1, n_ch=64, nl_type='relu', depth=20, bn=False, clip='relu')
        elif self.model_name =='wdsr':
            self.model = wdsr.MODEL(num_residual_units = 64, num_blocks = 16, width_multiplier = 4, scale=1)
        # elif self.model_name == 'FFDNET':
        #     self.model = FFDNet(3, 3, 64, 15, act_mode = act_mode)
        self.model.to(self.device)
        
        # Load model weights only or initialise weights
        if pretrained_checkpoint is not None and check != 1:
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            try:
                state_dict = checkpoint['state_dict']
            except:
                state_dict = checkpoint
            new_state_dict = {}
            for key, val in state_dict.items():
                new_key = key.split('student_grad.model.')[-1]
                new_state_dict[new_key] = val
            self.model.load_state_dict(new_state_dict, strict=True)
            print('Weights loaded from checkpoint.')
        else:
            print('Initialising weights.')
            self.model.apply(weights_init_kaiming)

    def forward(self, x):
        n = self.model(x)
        # if self.model_name=='DNCNN':
        #     clip = nn.ReLU()
        #     n = clip(n+x)
        # summary(self.model, (1, 192, 192))
        return n



class GradMatch(pl.LightningModule):
    '''
    Gradient Step Denoiser
    '''

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.student_grad = StudentGrad(self.hparams.model_name, 
                                        self.hparams.pretrained_checkpoint, self.hparams.act_mode,
                                        self.hparams.DRUNET_nb, self.hparams.residual_learning,
                                        self.hparams.resume_from_ckpt)
        self.train_ISNR = snr()
        self.train_log_ISNR = snr()
        self.train_RSNR = snr()
        self.train_log_RSNR = snr()
        self.val_ISNR = snr()
        self.val_log_ISNR = snr()
        self.val_RSNR = snr()
        self.val_log_RSNR = snr()

        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # if 'wdsr' in self.hparams.model_name.lower():
        #     j_max_iter = 40
        # else:
        j_max_iter = 20
        self.reg_fun, self.reg_fun_val = JacobianReg_l2(max_iter=j_max_iter), JacobianReg_l2(max_iter=j_max_iter,eval_mode=True)
        
        if self.hparams.starting_sigma > 0:
            self.current_sigma = self.hparams.starting_sigma
            self.drop_every_n_epoch = np.ceil(self.hparams.max_epochs/(np.log(hparams.sigma / hparams.starting_sigma) / np.log(hparams.sigma_drop)))
        else:
            self.current_sigma = self.hparams.sigma
            self.drop_every_n_epoch = 1



    def forward(self, x):
        x_hat = self.student_grad.forward(x)
        return x_hat

    def lossfn(self, x, y): # L2 loss
        curr_reduction = 'sum'
        if self.hparams.loss_name == 'l2':
            criterion = nn.MSELoss(reduction=curr_reduction)
        elif self.hparams.loss_name == 'l1':
            criterion = nn.L1Loss(reduction=curr_reduction)
        elif self.hparams.loss_name == 'sl1':
            criterion = nn.SmoothL1Loss(reduction=curr_reduction)
        elif self.hparams.loss_name == 'log':
            criterion = LogLoss(reduction=curr_reduction)
        elif self.hparams.loss_name == 'logl1':
            criterion = LogL1Loss(reduction=curr_reduction)
        elif self.hparams.loss_name == 'logcosh':
            criterion = LogCoshLoss(reduction=curr_reduction)
        elif self.hparams.loss_name == 'log1':
            criterion = LogLoss1(reduction=curr_reduction)
        elif self.hparams.loss_name == 'log05':
            criterion = LogLoss05(reduction=curr_reduction)
        elif self.hparams.loss_name == 'log01':
            criterion = LogLoss01(reduction=curr_reduction)
        elif self.hparams.loss_name == 'log001':
            criterion = LogLoss001(reduction=curr_reduction)
        elif self.hparams.loss_name == 'log005':
            criterion = LogLoss005(reduction=curr_reduction)
        size_tot = y.size()[0] * 2 * y.size()[-1]**2
        crit = criterion(x.view(x.size()[0], -1), y.view(y.size()[0], -1))/size_tot
        return crit

    def training_step(self, batch, batch_idx):
        # torch.manual_seed(self.hparams.seed)
        torch.set_grad_enabled(True)
        data_true, _ = batch
        # print(y[:,:1,...].size())
        # print(y[:,:1,...])
        # data_true = y[:,:1,...].type(self.Tensor)
        
        noise_in = torch.randn_like(data_true, device=self.device) * self.current_sigma
        x = data_true + noise_in
        x_hat = self.forward(x)
        loss = self.lossfn(x_hat, data_true)

        jacobian_norm = 0.

        if self.hparams.jacobian_loss_weight > 0:
            if 'unet' in self.hparams.model_name.lower():
                jacobian_norm = self.jacobian_spectral_norm(data_true[0:1,:,:128,:128], x_hat[0:1,:,:128,:128], self.forward, self.reg_fun)
            else:
                jacobian_norm = self.jacobian_spectral_norm(data_true, x_hat, self.forward, self.reg_fun)
            self.log('train/6_jacobian_norm_max', jacobian_norm.max(), on_epoch=True, sync_dist=True)
            jacobian_loss = jacobian_norm * self.hparams.jacobian_loss_weight
            self.log('train/7_jacobian_loss', jacobian_loss, sync_dist=True)
            
            # jacobian_loss = torch.clip(jacobian_loss, 0, 1e3)
            # self.log('train/5_jacobian_loss_max', jacobian_loss.max(), sync_dist=True)

            loss += jacobian_loss.item()

        loss = loss.mean()

        # psnr = self.train_PSNR.compute()
        with torch.no_grad():
            self.train_ISNR.update(x, data_true)
            x_log = log_img(x)
            data_true_log = log_img(data_true)
            x_hat_log = log_img(x_hat)
            self.train_RSNR.update(x_hat, data_true)
            self.train_log_RSNR.update(x_hat_log, data_true_log)
            self.train_log_ISNR.update(x_log, data_true_log)
            rsnr = self.train_RSNR.compute()
            isnr = self.train_ISNR.compute()
            if (self.global_step+1)%10000 == 0:
                print('Step {}: sigma = {:e}; SNR difference = {}'.format(int(self.global_step+1), self.current_sigma, (rsnr-isnr).detach()))
            log_isnr = self.train_log_ISNR.compute()
            log_rsnr = self.train_log_RSNR.compute()
            self.log('train/3_train_loss', loss.detach(), sync_dist=True)
            self.log('train/1_ISNR_step', isnr.detach(), sync_dist=True)
            self.log('train/2_RSNR_step', rsnr.detach(), sync_dist=True)
            self.log('train/4_SNR_diff', (rsnr-isnr).detach(), sync_dist=True)
            self.log('train/5_sigma', self.current_sigma, sync_dist=True)
            self.log('train/8_logISNR_step', log_isnr.detach(), sync_dist=True)
            self.log('train/9_logRSNR_step', log_rsnr.detach(), sync_dist=True)
            self.log('train/10_logSNR_diff', (log_rsnr-log_isnr).detach(), sync_dist=True)
            

            if batch_idx == 0:
                noisy_train = normalize_min_max(x.detach())[:1]
                denoised_train = normalize_min_max(x_hat.detach())[:1]
                clean_train = normalize_min_max(data_true.detach())[:1]
                noisy_train_log = log_img(noisy_train)
                denoised_train_log = log_img(denoised_train)
                clean_train_log = log_img(clean_train)
                noisy_grid = torchvision.utils.make_grid(noisy_train_log)
                denoised_grid = torchvision.utils.make_grid(denoised_train_log)
                clean_grid = torchvision.utils.make_grid(clean_train_log)
                # noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
                # denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
                # clean_grid = torchvision.utils.make_grid(normalize_min_max(data_true.detach())[:1])
                self.logger.experiment.add_image('train/1_clean', clean_grid, self.current_epoch)
                self.logger.experiment.add_image('train/2_noisy', noisy_grid, self.current_epoch)
                self.logger.experiment.add_image('train/3_denoised', denoised_grid, self.current_epoch)

            self.train_ISNR.reset()
            self.train_RSNR.reset()
            self.train_log_ISNR.reset()
            self.train_log_RSNR.reset()

        return loss

    def training_epoch_end(self, outputs):
        # rsnr = self.train_RSNR.compute()
        # isnr = self.train_ISNR.compute()
        # self.log('train/1_ISNR_epoch', isnr.detach(),sync_dist=True)
        # self.log('train/2_RSNR_epoch', rsnr.detach(),sync_dist=True)
        # training_update_text = 'Training for sigma={}, dr={}: test in training epoch {}'.format(self.hparams.sigma, self.hparams.dr, int(self.current_epoch+1))
        # print(training_update_text)
        # send_update(training_update_text)
        # torch.set_grad_enabled(False)
        with torch.no_grad():
            if (self.current_epoch+1)%50 == 0:
                print('Metrics updated for epoch {}-{} for training.'.format(int(self.current_epoch+1)-49,int(self.current_epoch+1)))
            # self.train_PSNR.reset()
            if (self.current_epoch + 2) % self.drop_every_n_epoch == 0 and self.hparams.starting_sigma > 0:
                self.current_sigma *= self.hparams.sigma_drop
                print('At epoch # {}, sigma dropped from {} to {}.'.format(str(self.current_epoch+1), str(self.current_sigma/ self.hparams.sigma_drop), str(self.current_sigma)))

    def validation_step(self, batch, batch_idx):
        # torch.manual_seed(self.hparams.seed)
        y, _ = batch
        data_true = y[:,:1,...].type(self.Tensor)
        batch_dict = {}

        x = data_true + torch.randn(data_true.size(), device=self.device) * self.current_sigma

        torch.set_grad_enabled(True)
        x_hat = self.forward(x)
        torch.set_grad_enabled(False)
        data_true_log = log_img(data_true)
        x_log = log_img(x)
        x_hat_log = log_img(x_hat)
        l = self.lossfn(x_hat, data_true)
        
        self.val_ISNR.update(x, data_true)
        self.val_RSNR.update(x_hat, data_true)
        self.val_log_ISNR.update(x_log, data_true_log)
        self.val_log_RSNR.update(x_hat_log, data_true_log)

        if self.hparams.jacobian_loss_weight > 0:
            torch.set_grad_enabled(True)
            if 'unet' in self.hparams.model_name.lower():
                val_jacobian_norm = self.jacobian_spectral_norm(data_true[0:1,:,:128,:128], x_hat[0:1,:,:128,:128], self.forward, self.reg_fun_val)
            else:
                val_jacobian_norm = self.jacobian_spectral_norm(data_true, x_hat, self.forward, self.reg_fun)
            # self.log('val/4_jacobian_norm_max', val_jacobian_norm.max(), on_epoch=True, sync_dist=True)
            val_jacobian_loss = val_jacobian_norm * self.hparams.jacobian_loss_weight
            # self.log('val/5_jacobian_loss_max', val_jacobian_loss.max(), sync_dist=True)

            l += val_jacobian_loss.item()
        
        l = l.mean()
        torch.set_grad_enabled(False)
        rsnr = self.val_RSNR.compute()
        isnr = self.val_ISNR.compute()
        log_rsnr = self.val_log_RSNR.compute()
        log_isnr = self.val_log_ISNR.compute()

        if batch_idx == 0: # logging for tensorboard
            clean_grid = torchvision.utils.make_grid(normalize_min_max(log_img(y).detach())[:1])
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(log_img(x).detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(log_img(x_hat).detach())[:1])
            self.logger.experiment.add_image('val/1_clean', clean_grid, self.current_epoch)
            self.logger.experiment.add_image('val/2_noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('val/3_denoised', denoised_grid, self.current_epoch)

        if self.hparams.save_images:
            save_dir = 'images/' + self.hparams.name

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                os.mkdir(save_dir + '/noisy')
                os.mkdir(save_dir + '/denoised')
                os.mkdir(save_dir + '/denoised_no_noise')
                os.mkdir(save_dir + '/clean')
            for i in range(len(x)):
                clean = y[i].detach().cpu().numpy().transpose(1, 2, 0)
                noisy = x[i].detach().cpu().numpy().transpose(1, 2, 0)
                denoised = x_hat[i].detach().cpu().numpy().transpose(1, 2, 0)

                cv2.imwrite(save_dir + '/denoised/' + str(batch_idx) + '.png', denoised)
                cv2.imwrite(save_dir + '/clean/' + str(batch_idx) + '.png', clean)
                cv2.imwrite(save_dir + '/noisy/' + str(batch_idx) + '.png', noisy)
        batch_dict["val_loss"] = l.detach()
        batch_dict["val_rsnr"] = rsnr.detach()
        batch_dict["val_isnr"] = isnr.detach()
        self.log('val/3_val_loss', l.detach(), sync_dist=True)
        self.log('val/1_ISNR', isnr.detach(), sync_dist=True)
        self.log('val/2_RSNR', rsnr.detach(), sync_dist=True)
        self.log('val/4_SNR_diff', (rsnr-isnr).detach(), sync_dist=True)
        self.log('val/6_logISNR', log_isnr.detach(), sync_dist=True)
        self.log('val/7_logRSNR', log_rsnr.detach(), sync_dist=True)
        self.log('val/8_logSNR_diff', (log_rsnr-log_isnr).detach(), sync_dist=True)
        if self.hparams.jacobian_loss_weight > 0:
            batch_dict["val_jacobian_norm"] = val_jacobian_norm.detach()
            batch_dict["val_jacobian_loss"] = val_jacobian_loss.detach()
            self.log('val/5_jac_norm', val_jacobian_norm.detach(), sync_dist=True)
            # self.log('val/6_jac_loss', val_jacobian_loss.detach(), sync_dist=True)

        return batch_dict

    def validation_epoch_end(self, outputs):
        torch.set_grad_enabled(False)
        print('Metrics updated at epoch {} for validation'.format(int(self.current_epoch+1)))
        # res_l = []
        # res_rsnr = []
        # res_isnr = []
        # if self.hparams.jacobian_loss_weight > 0:
        #     res_jac_norm = []
        #     res_jac_loss = []
        # for x in outputs:
        #     res_l.append(x["val_loss"])
        #     res_rsnr.append(x["val_rsnr"])
        #     res_isnr.append(x["val_isnr"])
        #     if self.hparams.jacobian_loss_weight > 0:
        #         res_jac_norm.append(x["val_jacobian_norm"])
        #         res_jac_loss.append(x["val_jacobian_loss"])
        # avg_loss = torch.stack(res_l).mean()
        # avg_rsnr = torch.stack(res_rsnr).mean()
        # avg_isnr = torch.stack(res_isnr).mean()
        # if self.hparams.jacobian_loss_weight > 0:
        #     avg_jac_norm = torch.stack(res_jac_norm).mean()
        #     avg_jac_loss = torch.stack(res_jac_loss).mean()
        # self.log('val/3_val_loss', avg_loss, sync_dist=True)
        # self.log('val/1_ISNR', avg_isnr, sync_dist=True)
        # self.log('val/2_RSNR', avg_rsnr, sync_dist=True)
        # self.log('val/4_SNR_diff', (avg_rsnr-avg_isnr), sync_dist=True)
        # self.val_RSNR.reset()
        # self.val_ISNR.reset()
        # if self.hparams.jacobian_loss_weight > 0:
        #     self.log('val/5_jac_norm', avg_jac_norm, sync_dist=True)
        #     self.log('val/6_jac_loss', avg_jac_loss, sync_dist=True)


    def test_step(self, batch, batch_idx):
        # return self.validation_step(batch, batch_idx)
        return None

    def test_epoch_end(self, outputs):
        # return self.validation_epoch_end(outputs)
        return None

    # if self.hparams.scheduler_milestones > 0:
    def configure_optimizers(self):
        optim_params = []
        for k, v in self.student_grad.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = Adam(optim_params, lr=self.hparams.optimizer_lr, weight_decay=0)
        print('Initial learning rate: {:.4e}'.format(self.hparams.optimizer_lr))
        scheduler = lr_scheduler.StepLR(optimizer,
                                            self.hparams.scheduler_milestones,
                                            self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def jacobian_spectral_norm(self, y, x_hat, model, reg_fun):
        '''
        computes the regularization reg_fun applied to the correct point
        '''

        x_detached = x_hat.detach().type(self.Tensor)
        y_detached = y.detach().type(self.Tensor)

        tau = torch.rand(y_detached.shape[0], 1, 1, 1).type(self.Tensor)
        x_detached = tau * x_detached + (1 - tau) * y_detached
        x_detached.requires_grad_()

        x_reg = model(x_detached)
        x_net_reg = 2. * x_reg - x_detached
        reg_loss = reg_fun(x_detached, x_net_reg)
        reg_loss = torch.nan_to_num(reg_loss)
        reg_loss_max = torch.max(reg_loss, torch.ones_like(reg_loss) + self.hparams.epsilon)
        reg_loss_comb = reg_loss_max.max()
        return reg_loss_comb


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='DRUNET')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--loss_name', type=str, default='l1')
        parser.add_argument('--pretrained_checkpoint', type=str,default=None)
        parser.add_argument('--nc', type=int, default=64)
        parser.add_argument('--act_mode', type=str, default='s')
        parser.add_argument('--no_bias', dest='no_bias', action='store_false')
        parser.set_defaults(use_bias=True)
        parser.add_argument('--DRUNET_nb', type=int, default=2)
        parser.add_argument('--residual_learning', type=int, default=0)
        # parser.add_argument('--start_from_checkpoint', dest='start_from_checkpoint', action='store_true')
        # parser.set_defaults(start_from_checkpoint=False)
        # parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        # parser.set_defaults(resume_from_checkpoint=False)
        # parser.add_argument('--pretrained_student', dest='pretrained_student', action='store_true')
        # parser.set_defaults(pretrained_student=False)
        # parser.add_argument('--n_channels', type=int, default=3)
        # parser.add_argument('--nc_in', type=int, default=3)
        # parser.add_argument('--nc_out', type=int, default=3)
        # parser.add_argument('--nb', type=int, default=20)
        # parser.add_argument('--power_method_nb_step', type=int, default=50)
        # parser.add_argument('--power_method_error_threshold', type=float, default=1e-2)
        # parser.add_argument('--power_method_error_momentum', type=float, default=0.)
        # parser.add_argument('--power_method_mean_correction', dest='power_method_mean_correction', action='store_true')
        # parser.set_defaults(power_method_mean_correction=False)
        # parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        # parser.set_defaults(grad_matching=True)
        # parser.add_argument('--weight_Ds', type=float, default=1.)
        # parser.add_argument('--residual_learning', dest='residual_learning', action='store_true')
        # parser.set_defaults(residual_learning=False)
        return parser

    @staticmethod
    def add_optim_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--optimizer_lr', type=float, default=1e-4)
        # parser.add_argument('--scheduler_milestones', type=int, default=700)
        parser.add_argument('--scheduler_gamma', type=float, default=0.5)
        parser.add_argument('--early_stopping_patiente', type=str, default=5)
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--val_check_interval', type=float, default=1.)
        parser.add_argument('--sigma', type=float, default=3.2e-4)
        parser.add_argument('--starting_sigma', type=float, default=0)
        parser.add_argument('--sigma_drop', type=float, default=0.975)
        parser.add_argument('--epsilon', type=float, default=-0.05)
        parser.add_argument('--jacobian_loss_weight', type=float, default=0)
        parser.add_argument('--resume_from_ckpt', type=int, default=0)
        # parser.add_argument('--optimizer_type', type=str, default='adam')
        # parser.add_argument('--scheduler_type', type=str, default='MultiStepLR')
        # parser.add_argument('--min_sigma_test', type=int, default=0)
        # parser.add_argument('--max_sigma_test', type=int, default=50)
        # parser.add_argument('--min_sigma_train', type=int, default=0)
        # parser.add_argument('--max_sigma_train', type=int, default=50)
        # parser.add_argument('--sigma_list_test', type=int, nargs='+', default=[0,15,25,50])
        # parser.add_argument('--sigma_step', dest='sigma_step', action='store_true')
        # parser.set_defaults(sigma_step=False)
        # parser.add_argument('--get_spectral_norm', dest='get_spectral_norm', action='store_true')
        # parser.set_defaults(get_spectral_norm=False)
        # parser.add_argument('--eps_jacobian_loss', type=float, default=0.1)
        # parser.add_argument('--jacobian_loss_type', type=str, default='max')
        # parser.add_argument('--n_step_eval', type=int, default=1)
        # parser.add_argument('--use_post_forward_clip', dest='use_post_forward_clip', action='store_true')
        # parser.set_defaults(use_post_forward_clip=False)
        # parser.add_argument('--use_sigma_model', dest='use_sigma_model', action='store_true')
        # parser.set_defaults(use_sigma_model=False)
        # parser.add_argument('--sigma_model', type=int, default=25)
        # parser.add_argument('--get_regularization', dest='get_regularization', action='store_true')
        # parser.set_defaults(get_regularization=False)
        return parser
