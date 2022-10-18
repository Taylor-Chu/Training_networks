import argparse

import torch

import os

from models.models_helper import get_model
from models.network_unet import UNetAuthor, UNet, UNetRes, UNetRes2, UNetRes3, UNetRes4
import models.network_unet as unets
from models.DNCNN import dncnn
import pytorch_lightning as pl

import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

parser = argparse.ArgumentParser(description="onnx")
parser.add_argument("--folder_out",
                    type=str,
                    default='onnx_models/',
                    help="pth save")
parser.add_argument("--save_name",
                    type=str,
                    default='model/',
                    help="name of network file")
parser.add_argument("--checkpoint_path",
                    type=str,
                    default='/my_trained_models/dncnn.pth',
                    help="pth network")
parser.add_argument("--architecture",
                    type=str,
                    default='DnCNN_nobn',
                    help="architecture")

opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def convert_model(model_type,
                  folder_out='onnx_models/',
                  save_name='model.onnx',
                  checkpoint_path='pth/to/my/model.pth'):
    
    print('Checkpoint path: {}'.format(checkpoint_path))
    print('Network architecture: {}'.format(model_type))
    print('Output folder: {}'.format(folder_out))

    # if not os.path.exists(opt.folder_out):
    #     os.makedirs(opt.folder_out, exist_ok=True)
    #     print('Output folder [{}] created.'.format(opt.folder_out))
    # else:
    #     print('Output folder [{}] exists!'.format(folder_out))
    # if 'dncnn' in model_type.lower():
    #     model, net_name, _, _ = get_model('DnCNN_nobn',
    #                                     n_ch=1,
    #                                     checkpoint_pth=checkpoint_path)
    #     print('checkpoint loaded from {}'.format(checkpoint_path))

    #     # fileName = os.path.basename(checkpoint_path)
    #     # fileName, _ = os.path.splitext(fileName)

    #     input_names = ['input_image']
    #     output_names = ['output_image']
    #     save_pth = folder_out + save_name + ".onnx"
    #     with torch.no_grad():
    #         im_in = torch.randn((1, 1, 512, 512)).type(Tensor)
    #         torch.onnx.export(model.module,
    #                         im_in,
    #                         save_pth,
    #                         verbose=True,
    #                         input_names=input_names,
    #                         output_names=output_names)
    #     print('Success!')
    if model_type.lower() == 'unetres':
        model = UNetRes(in_nc = 1, out_nc = 1, nb = 2, nc=[32, 64, 128, 256])
    elif model_type.lower() == 'unetres2':
        model = UNetRes2(in_nc = 1, out_nc = 1, nb = 2, nc=[32, 64, 128, 256])
    elif model_type.lower() == 'unetres3':
        model = UNetRes3(in_nc = 1, out_nc = 1, nb = 2, nc=[32, 64, 128, 256])
    elif model_type.lower() == 'unetres4':
        model = UNetRes4(in_nc = 1, out_nc = 1, nb = 2, nc=[32, 64, 128, 256])
    elif model_type.lower() == 'dncnn':
        model = dncnn(nc_in=1, nc_out=1, nb=20, mode='C', act_mode='l', bias=True)
    else:
        print('Model not available!')
    checkpoint = torch.load(checkpoint_path)
    print('checkpoint loaded from {}'.format(checkpoint_path))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, val in state_dict.items():
        new_name = ('.').join(key.split('.')[2:])
        new_state_dict[new_name] = val
    model.load_state_dict(new_state_dict, strict=True)

    # fileName = os.path.basename(checkpoint_path)
    # fileName, _ = os.path.splitext(fileName)
    save_pth = folder_out + save_name + ".onnx"
    input_names = ['input_image']
    output_names = ['output_image']
    
    with torch.no_grad():
        im_in = torch.randn((1, 1, 512, 512)).type(torch.FloatTensor)
        torch.onnx.export(model,
                        im_in,
                        save_pth,
                        verbose=True,
                        opset_version=11,
                        input_names=input_names,
                        output_names=output_names)
        print('Success!')


if __name__ == '__main__':

    convert_model(opt.architecture,
                  opt.folder_out,
                  opt.save_name,
                  opt.checkpoint_path)
