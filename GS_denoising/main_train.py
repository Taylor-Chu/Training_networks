import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['MPLCONFIGDIR'] = '/work/sc004/sc004/tc1213/config/matplotlib'
import gc
gc.collect()

import pytorch_lightning as pl
from lightning_denoiser import GradMatch
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
# from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from models.model_to_onnx import convert_model
from argparse import ArgumentParser
import random
import torch
torch.cuda.empty_cache()
import numpy as np
# from bot import send_update

if __name__ == '__main__':
    # PROGRAM args
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--max_epochs', type=int, default=0)
    parser.add_argument('--scheduler_milestones', type=int, default=0)
    parser.add_argument('--log_folder', type=str, default='logs')
    parser.add_argument('--convert_to_onnx', type=int, default=0)
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.set_defaults(save_images=False)

    # MODEL args
    parser = GradMatch.add_model_specific_args(parser)
    # DATA args
    parser = DataModule.add_data_specific_args(parser)
    # OPTIM args
    parser = GradMatch.add_optim_specific_args(parser)

    hparams = parser.parse_args()
    print('Finished setting up hparams.')
    refresh_rate = 144
    pl.seed_everything(hparams.seed)

    # random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    
    if not os.path.exists(hparams.log_folder):
        os.makedirs(hparams.log_folder, exist_ok=True)
    log_path = hparams.log_folder + '/' + hparams.name
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(log_path,
                                            log_graph=True)
    # if hparams.pretrained_checkpoint is not None and hparams.resume_from_ckpt == 0:
    #     try:
    #         model = GradMatch(hparams).load_from_checkpoint(hparams.pretrained_checkpoint, strict=False)
    #         print('Loaded model from checkpoint.')
    #     except:
    #         print('cannot load by model.load_from_checkpoint, will load by torch.load later...')
    #         model = GradMatch(hparams)
    #         print('Loaded model.')
    # else:
    model = GradMatch(hparams)
    print('Loaded model.')
    dm = DataModule(hparams)
    print('Loaded dataset.')
    # if hparams.pretrained_checkpoint is not None:
    #     checkpoint = torch.load(hparams.pretrained_checkpoint)
    #     print('Loaded pretrained network from {}.'.format(hparams.pretrained_checkpoint))
    #     model.load_state_dict(checkpoint['state_dict'],strict=False)

    early_stop_callback = EarlyStopping(
        monitor='val/3_val_loss',
        min_delta=1e-8,
        patience=hparams.early_stopping_patiente,
        verbose=True,
        mode='min'
    )

    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if hparams.max_epochs > 0:
        max_epochs = hparams.max_epochs
        # hparams.scheduler_milestones = hparams.max_epochs*0.8
        # print(hparams)
    else:
        if hparams.jacobian_loss_weight > 0:
            max_epochs = 400
            hparams.scheduler_milestones = 300
        else:
            max_epochs = 800
            hparams.scheduler_milestones = 600
                    

    # removing log_gpu_memory, if needed use DeviceStatsMonitor callback

    save_checkpoint = ModelCheckpoint(
        filename="best_loss_{epoch}-{step}",
        monitor="train/3_train_loss",
        save_top_k=2,
        mode="min",
        save_weights_only=False
    )
    save_n = 10000
    save_checkpoint2 = ModelCheckpoint(
        filename="reg_{epoch}_{step}",
        every_n_train_steps=save_n,
        # save_top_k=-1,
        save_weights_only=False
    )
    
    save_checkpoint3 = ModelCheckpoint(
        filename="reg_{epoch}_{step}_weights_only",
        every_n_train_steps=save_n,
        save_weights_only=True
    )
    # save_checkpoint3 = ModelCheckpoint(
    #     filename="best_RSNR_{epoch}_{step}",
    #     save_top_k=2,
    #     monitor = 'train/4_SNR_diff',
    #     mode="max",
    #     save_weights_only=False
    # )

    # save_epoch_checkpoint = ModelCheckpoint(
    #     filename="{epoch}-{step}",
    #     monitor="period",
    #     save_top_k=-50
    # )

    print('Starting training process.')
    trainer = pl.Trainer.from_argparse_args(hparams, logger=tb_logger, 
                                            gpus=-1,
                                            check_val_every_n_epoch=50,
                                            gradient_clip_val=hparams.gradient_clip_val,
                                            max_epochs=max_epochs, precision=16,
                                            callbacks=[lr_monitor,save_checkpoint, save_checkpoint2, save_checkpoint3], 
                                            # accelerator = 'gpu',
                                            strategy='ddp_find_unused_parameters_false',
                                            enable_progress_bar = False)
    
    # send_update("Starting the training")
    if hparams.resume_from_ckpt == 1:
        print('Resuming from specified checkpoint path.')
        trainer.fit(model,
                    dm,
                    ckpt_path=hparams.pretrained_checkpoint)
    else:
        trainer.fit(model, dm)
    print('Training finished.')
    final_ckpt_pth = os.path.join(log_path,'final.ckpt')
    trainer.save_checkpoint(final_ckpt_pth)
    final_ckpt_pth_weights_only = os.path.join(log_path,'final_weights_only.ckpt')
    trainer.save_checkpoint(final_ckpt_pth_weights_only)
    print('Final checkpoint has been saved at {}.'.format(final_ckpt_pth))
    if hparams.convert_to_onnx == 1:
        if ~os.path.exists(os.path.join('/work/sc004/sc004/tc1213/trained_networks',hparams.dataset_name)):
            os.mkdir(os.path.join('/work/sc004/sc004/tc1213/trained_networks',hparams.dataset_name))
        if ~os.path.exists(os.path.join('/work/sc004/sc004/tc1213/trained_networks',hparams.dataset_name, hparams.model_name)):
            os.mkdir(os.path.join('/work/sc004/sc004/tc1213/trained_networks',hparams.dataset_name, hparams.model_name))
        save_pth = os.path.join('/work/sc004/sc004/tc1213/trained_networks',hparams.dataset_name, hparams.model_name) +'/seed'+str(hparams.seed)
        if ~os.path.exists(save_pth):
            os.mkdir(save_pth)
        save_name = hparams.model_name + '_' + hparams.dataset_name + '_' + str(hparams.sigma) + '_' + str(hparams.dr) + '_' + str(hparams.seed)
        convert_model(hparams.model_name,
                        save_pth, 
                        save_name,
                        final_ckpt_pth)
