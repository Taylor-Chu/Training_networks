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
from models.network_unet import UNetRes, UNet
from models import DNCNN
from models.FFDNET import FFDNet

from lossfunc import LogLoss, LogCoshLoss, LogL1Loss, LogLoss1, LogLoss05, LogLoss01, LogLoss001, LogLoss005
from jacobian import JacobianReg_l2
from utils.metrics_helper import snr


class StudentGrad(pl.LightningModule):
    '''
    Standard Denoiser model
    '''
    def __init__(self, model_name, pretrained_checkpoint, act_mode, DRUNET_nb, residual_learning):
        super().__init__()
        self.model_name = model_name
        self.residual_learning = residual_learning
        if self.model_name == 'UNet':
            self.model = UNet(in_nc = 1, out_nc = 1)
        elif self.model_name == 'DRUNET':
            self.model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=DRUNET_nb, act_mode=act_mode,
                                 downsample_mode='strideconv', upsample_mode='convtranspose')
        elif self.model_name == 'DNCNN':
            self.model = DNCNN.dncnn(nc_in=1, nc_out=1, nb=20, mode='C', act_mode='l', bias=True)
        elif self.model_name == 'FFDNET':
            self.model = FFDNet(3, 3, 64, 15, act_mode = act_mode)
        self.model.to(self.device)
        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, val in state_dict.items():
                new_state_dict[key[6:]] = val
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x, sigma):
        if self.model_name == 'FFDNET':
            n = self.model(x,torch.full((x.shape[0], 1, 1, 1), sigma).type_as(x))
        else :
            if self.model_name == 'DRUNET':
                noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(self.device)
                x = torch.cat((x, noise_level_map), 1)
            n = self.model(x)
            if self.model_name == 'DNCNN':
                n = torch.maximum((x+n),torch.zeros_like(n))
        
        if self.residual_learning:
            print('residual learning happening.')
            return x - n
        else:
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
                                        self.hparams.DRUNET_nb, self.hparams.residual_learning)
        # self.train_PSNR = torchmetrics.PSNR(data_range=1.0)
        self.train_ISNR = snr()
        self.train_RSNR = snr()
        # self.val_PSNR = torchmetrics.PSNR(data_range=1.0)
        self.val_ISNR = snr()
        self.val_RSNR = snr()

        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.reg_fun, self.reg_fun_val = JacobianReg_l2(), JacobianReg_l2(eval_mode=True)


    # def calculate_grad(self, x, sigma):
    #     '''
    #     Calculate Dg(x) the gradient of the regularizer g at input x
    #     :param x: torch.tensor Input image
    #     :param sigma: Denoiser level (std)
    #     :return: Dg(x), DRUNet output N(x)
    #     '''
    #     x = x.float()
    #     x = x.requires_grad_()

    #     if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
    #         N = self.student_grad.forward(x, sigma)
    #     else:
    #         current_model = lambda v: self.student_grad.forward(v, sigma)
    #         N = test_mode(current_model, x, mode=5, refield=64, min_size=256)
    #     JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=True, only_inputs=True)[0]
    #     Dg = x - N - JN
    #     return Dg, N

    def forward(self, x, sigma):
        '''
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :param sigma: Denoiser level (std)
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        '''
        # if self.hparams.grad_matching: # If gradient step denoising
        #     Dg, _ = self.calculate_grad(x, sigma)
        #     x_hat = x - self.hparams.weight_Ds * Dg
        #     return x_hat, Dg
        # else: # If denoising with standard forward CNN
        x_hat = self.student_grad.forward(x, sigma)
        Dg = x - x_hat
        return x_hat, Dg

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
        torch.manual_seed(self.hparams.seed)
        y, _ = batch

        data_true = y[:,:1,...].type(self.Tensor)
        u = torch.randn(data_true.size(), device=self.device)
        noise_in = u * self.hparams.sigma
        x = data_true + noise_in
        x_hat, _ = self.forward(x, self.hparams.sigma)
        loss = self.lossfn(x_hat, data_true)
        # self.train_PSNR.update(x_hat, data_true)
        # train_RSNR = snr(x_hat, data_true)
        # train_ISNR = snr(x, data_true)
        self.train_RSNR.update(x_hat, data_true)
        self.train_ISNR.update(x, data_true)

        jacobian_norm = 0.

        if self.hparams.jacobian_loss_weight > 0:
            jacobian_norm = self.jacobian_spectral_norm(data_true, x_hat)
            self.log('train/jacobian_norm_max_step', jacobian_norm, sync_dist=True)
            jacobian_loss = jacobian_norm * self.hparams.jacobian_loss_weight
            jacobian_loss = torch.clip(jacobian_loss, 0, 1e3)
            self.log('train/7_jacobian_loss_max', jacobian_loss.max(), sync_dist=True)
            self.log('train/8_jacobian_norm_max_epoch', jacobian_norm.max(), sync_dist=True)
            self.log('train/9_jacobian_norm_min_epoch', jacobian_norm.min(), sync_dist=True)

            loss = loss + jacobian_loss

        loss = loss.mean()

        # psnr = self.train_PSNR.compute()
        rsnr = self.train_RSNR.compute()
        isnr = self.train_ISNR.compute()
        self.log('train/5_train_loss', loss.detach(), reduce_fx = 'mean', sync_dist=True)
        # self.log('train/6_train_psnr', psnr.detach())
        self.log('train/3_ISNR_step', isnr.detach(), sync_dist=True)
        self.log('train/4_RSNR_step', rsnr.detach(), sync_dist=True)

        if batch_idx == 0:
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
            clean_grid = torchvision.utils.make_grid(normalize_min_max(data_true.detach())[:1])
            self.logger.experiment.add_image('train/clean', clean_grid, self.current_epoch)
            self.logger.experiment.add_image('train/noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('train/denoised', denoised_grid, self.current_epoch)

        return loss

    def training_epoch_end(self, outputs):
        rsnr = self.train_RSNR.compute()
        isnr = self.train_ISNR.compute()
        self.log('train/1_ISNR_epoch', isnr.detach(),sync_dist=True)
        self.log('train/2_RSNR_epoch', rsnr.detach(),sync_dist=True)
        print('metrics updated')
        # self.train_PSNR.reset()
        self.train_ISNR.reset()
        self.train_RSNR.reset()

    def validation_step(self, batch, batch_idx):
        torch.manual_seed(self.hparams.seed)
        y, _ = batch
        data_true = y[:,:1,...].type(self.Tensor)
        batch_dict = {}

        x = data_true + torch.randn(data_true.size(), device=self.device) * self.hparams.sigma

        torch.set_grad_enabled(True)
        x_hat, _ = self.forward(x, self.hparams.sigma)
        l = self.lossfn(x_hat, data_true).mean()
        # self.val_PSNR.reset()
        self.val_RSNR.reset()
        self.val_ISNR.reset()
        # p = self.val_PSNR(x_hat, data_true)
        isnr_val = self.val_ISNR(x, data_true)
        rsnr_val = self.val_RSNR(x_hat, data_true)
        # val_RSNR = snr(x_hat, y)
        # val_ISNR = snr(x, y)
        # if self.hparams.grad_matching: # GS denoise
        #     x_hat = x
        #     for n in range(self.hparams.n_step_eval): # 1 step in practice
        #         # current_model = lambda v: self.forward(v, sigma_model)[0]
        #         x_hat, _ = self.forward(x_hat, self.hparams.sigma)
        #     if self.hparams.get_regularization: # Extract reguralizer value g(x)
        #         N = self.student_grad.forward(x, self.hparams.sigma)
        #         g = 0.5 * torch.sum((x - N).view(x.shape[0], -1) ** 2)
        #         batch_dict["g_" + str(self.hparams.sigma)] = g.detach()
        #     l = self.lossfn(x_hat, y).mean()
        #     self.val_PSNR.reset()
        #     p = self.val_PSNR(x_hat, y)
        #     Dg = (x - x_hat)
        #     Dg_norm = torch.norm(Dg, p=2)
        # else:
        #     for n in range(self.hparams.n_step_eval):
        #         # current_model = lambda v: self.forward(v, self.hparams.sigma / 255)[0]
        #         x_hat = x
        #         if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
        #             x_hat = self.forward(x_hat, self.hparams.sigma)
        #         elif x.size(2) % 8 != 0 or x.size(3) % 8 != 0:
        #             x_hat = test_mode(self.forward(x_hat, self.hparams.sigma), x_hat, refield=64, mode=5)
        #     Dg = (x - x_hat)
        #     Dg_norm = torch.norm(Dg, p=2)
        #     l = self.lossfn(x_hat, y)
        #     self.val_PSNR.reset()
        #     p = self.val_PSNR(x_hat, y)

        if self.hparams.jacobian_loss_weight > 0:
            jacobian_norm = self.jacobian_spectral_norm(data_true, x_hat)
            batch_dict["max_jacobian_norm"] = jacobian_norm.max().detach()
            batch_dict["mean_jacobian_norm"] = jacobian_norm.mean().detach()
            l += self.hparams.jacobian_loss_weight * jacobian_norm

        # batch_dict["psnr"] = p.detach()
        batch_dict["isnr"] = isnr_val.detach()
        batch_dict["rsnr"] = rsnr_val.detach()
        batch_dict["loss"] = l.detach()
        # batch_dict["Dg_norm_" + str(self.hparams.sigma)] = Dg_norm.detach()

        if batch_idx == 0: # logging for tensorboard
            clean_grid = torchvision.utils.make_grid(normalize_min_max(y.detach())[:1])
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(normalize_min_max(x_hat.detach())[:1])
            self.logger.experiment.add_image('val/clean', clean_grid, self.current_epoch)
            self.logger.experiment.add_image('val/noisy', noisy_grid, self.current_epoch)
            self.logger.experiment.add_image('val/denoised', denoised_grid, self.current_epoch)

        if self.hparams.save_images:
            save_dir = 'images/' + self.hparams.name

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                os.mkdir(save_dir + '/noisy')
                os.mkdir(save_dir + '/denoised')
                os.mkdir(save_dir + '/denoised_no_noise')
                os.mkdir(save_dir + '/clean')
            for i in range(len(x)):
                clean = y[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                noisy = x[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                denoised = x_hat[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
                clean = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
                noisy = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
                denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)

                if self.hparams.sigma < 1:
                    cv2.imwrite(save_dir + '/denoised_no_noise/' + str(batch_idx) + '.png', denoised)
                else:
                    cv2.imwrite(save_dir + '/denoised/' + str(batch_idx) + '.png', denoised)
                    cv2.imwrite(save_dir + '/clean/' + str(batch_idx) + '.png', clean)
                    cv2.imwrite(save_dir + '/noisy/' + str(batch_idx) + '.png', noisy)

        return batch_dict

    def validation_epoch_end(self, outputs):

        # self.val_PSNR.reset()

        # self.log('val/1_ISNR_epoch', outputs['isnr'].mean())
        # self.log('val/2_RSNR_epoch', outputs['rsnr'].mean())
        # self.log('val/3_loss_epoch', outputs['loss'].mean())
        # self.log('val/4_PSNR_epoch', outputs['psnr'].mean())

        res_mean_SN = []
        res_max_SN = []
        # res_psnr = []
        res_isnr = []
        res_rsnr = []
        res_loss = []
        # res_Dg = []
        # if self.hparams.get_regularization:
        #     res_g = []
        for x in outputs:
            # if x["psnr"] is not None:
            #     res_psnr.append(x["psnr"])
            if x["rsnr"] is not None:
                res_rsnr.append(x["rsnr"])
            if x["isnr"] is not None:
                res_isnr.append(x["isnr"])
            if x["loss"] is not None:
                res_loss.append(x["loss"])
            # res_Dg.append(x["Dg_norm"])
            # if self.hparams.get_regularization:
            #     res_g.append(x["g"])
            if self.hparams.jacobian_loss_weight > 0:
                res_max_SN.append(x["max_jacobian_norm"])
                res_mean_SN.append(x["mean_jacobian_norm"])
        # avg_psnr = torch.stack(res_psnr).mean()
        avg_isnr = torch.stack(res_isnr).mean()
        avg_rsnr = torch.stack(res_rsnr).mean()
        avg_loss = torch.stack(res_loss).mean()
        # avg_Dg_norm = torch.stack(res_Dg).mean()
        # if self.hparams.get_regularization:
        #     avg_s = torch.stack(res_g).mean()
        #     self.log('val/val_g_sigma', avg_s)
        if self.hparams.jacobian_loss_weight > 0:
            avg_mean_SN = torch.stack(res_mean_SN).mean()
            max_max_SN = torch.stack(res_max_SN).max()
            self.log('val/5_val_max_SN_sigma', max_max_SN, sync_dist=True)
            self.log('val/6_val_mean_SN_sigma', avg_mean_SN, sync_dist=True)
            # res_max_SN = np.array([el.item() for el in res_max_SN])
            # np.save('res_max_SN_sigma' + '.npy', res_max_SN)
            #plt.hist(res_max_SN, bins='auto', label=r'\sigma='+str(sigma), alpha=0.5)
        # self.log('val/4_PSNR_epoch', avg_psnr)
        self.log('val/1_ISNR_epoch', avg_isnr, sync_dist=True)
        self.log('val/2_RSNR_epoch', avg_rsnr, sync_dist=True)
        self.log('val/3_loss_epoch', avg_loss, sync_dist=True)
        # self.log('val/val_rsnr_sigma', avg_rsnr_sigma)
        # self.log('val/val_isnr_sigma', avg_isnr_sigma)
        # self.log('val/val_Dg_norm_sigma=' + str(self.hparams.sigma), avg_Dg_norm)

        # if self.hparams.get_spectral_norm:
        #     plt.grid(True)
        #     plt.legend()
        #     plt.savefig('histogram.png')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optim_params = []
        for k, v in self.student_grad.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        optimizer = Adam(optim_params, lr=self.hparams.optimizer_lr, weight_decay=0)
        scheduler = lr_scheduler.StepLR(optimizer,
                                             self.hparams.scheduler_milestones,
                                             self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def jacobian_spectral_norm(self, y, x_hat):
        '''
        computes the regularization reg_fun applied to the correct point
        '''

        x_detached = x_hat.detach().type(self.Tensor)
        y_detached = y.detach().type(self.Tensor)
        # print('1:x_detached:{}'.format(x_detached))
        # print('2:y_detached:{}'.format(y_detached))

        tau = torch.rand(y_detached.shape[0], 1, 1, 1).type(self.Tensor)
        # print('3:tau:{}'.format(tau))
        x_detached = tau * x_detached + (1 - tau) * y_detached
        # print('4:x_detached:{}'.format(x_detached))
        x_detached.requires_grad_()

        x_reg, _ = self.forward(x_detached, self.hparams.sigma)
        # print('5:x_reg:{}'.format(x_reg))
        x_net_reg = 2. * x_reg - x_detached
        # print('6:x_net_reg:{}'.format(x_net_reg))
        reg_loss = self.reg_fun(x_detached, x_net_reg)
        reg_loss = torch.nan_to_num(reg_loss)
        # print('7:reg_loss:{}'.format(reg_loss))
        reg_loss_max = torch.max(reg_loss, torch.ones_like(reg_loss) + self.hparams.epsilon)
        # print('8:reg_loss_max:{}'.format(reg_loss_max))
        reg_loss_comb = reg_loss_max.max()
        # print('9:reg_loss_comb:{}'.format(reg_loss_comb))
        return reg_loss_comb


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default='DRUNET')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--loss_name', type=str, default='l1')
        # parser.add_argument('--start_from_checkpoint', dest='start_from_checkpoint', action='store_true')
        # parser.set_defaults(start_from_checkpoint=False)
        # parser.add_argument('--resume_from_checkpoint', dest='resume_from_checkpoint', action='store_true')
        # parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument('--pretrained_checkpoint', type=str,default=None)
        # parser.add_argument('--pretrained_student', dest='pretrained_student', action='store_true')
        # parser.set_defaults(pretrained_student=False)
        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--nc_in', type=int, default=3)
        parser.add_argument('--nc_out', type=int, default=3)
        parser.add_argument('--nc', type=int, default=64)
        parser.add_argument('--nb', type=int, default=20)
        parser.add_argument('--act_mode', type=str, default='s')
        parser.add_argument('--no_bias', dest='no_bias', action='store_false')
        parser.set_defaults(use_bias=True)
        parser.add_argument('--power_method_nb_step', type=int, default=50)
        parser.add_argument('--power_method_error_threshold', type=float, default=1e-2)
        parser.add_argument('--power_method_error_momentum', type=float, default=0.)
        parser.add_argument('--power_method_mean_correction', dest='power_method_mean_correction', action='store_true')
        parser.add_argument('--DRUNET_nb', type=int, default=2)
        parser.set_defaults(power_method_mean_correction=False)
        parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        parser.set_defaults(grad_matching=True)
        parser.add_argument('--weight_Ds', type=float, default=1.)
        parser.add_argument('--residual_learning', type=int, default=0)
        # parser.add_argument('--residual_learning', dest='residual_learning', action='store_true')
        # parser.set_defaults(residual_learning=False)
        return parser

    @staticmethod
    def add_optim_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--optimizer_type', type=str, default='adam')
        parser.add_argument('--optimizer_lr', type=float, default=1e-4)
        # parser.add_argument('--scheduler_type', type=str, default='MultiStepLR')
        parser.add_argument('--scheduler_milestones', type=int, default=800)
        parser.add_argument('--scheduler_gamma', type=float, default=0.5)
        parser.add_argument('--early_stopping_patiente', type=str, default=5)
        parser.add_argument('--gradient_clip_val', type=float, default=1e-2)
        parser.add_argument('--val_check_interval', type=float, default=1.)
        parser.add_argument('--sigma', type=float, default=3.2e-4)
        parser.add_argument('--epsilon', type=float, default=-0.05)
        # parser.add_argument('--min_sigma_test', type=int, default=0)
        # parser.add_argument('--max_sigma_test', type=int, default=50)
        # parser.add_argument('--min_sigma_train', type=int, default=0)
        # parser.add_argument('--max_sigma_train', type=int, default=50)
        # parser.add_argument('--sigma_list_test', type=int, nargs='+', default=[0,15,25,50])
        # parser.add_argument('--sigma_step', dest='sigma_step', action='store_true')
        # parser.set_defaults(sigma_step=False)
        parser.add_argument('--get_spectral_norm', dest='get_spectral_norm', action='store_true')
        parser.set_defaults(get_spectral_norm=False)
        parser.add_argument('--jacobian_loss_weight', type=float, default=0)
        parser.add_argument('--eps_jacobian_loss', type=float, default=0.1)
        parser.add_argument('--jacobian_loss_type', type=str, default='max')
        parser.add_argument('--n_step_eval', type=int, default=1)
        parser.add_argument('--use_post_forward_clip', dest='use_post_forward_clip', action='store_true')
        parser.set_defaults(use_post_forward_clip=False)
        # parser.add_argument('--use_sigma_model', dest='use_sigma_model', action='store_true')
        # parser.set_defaults(use_sigma_model=False)
        # parser.add_argument('--sigma_model', type=int, default=25)
        parser.add_argument('--get_regularization', dest='get_regularization', action='store_true')
        parser.set_defaults(get_regularization=False)
        return parser
