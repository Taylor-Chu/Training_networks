import argparse

import torch

import os

from .models_helper import get_model
from .network_unet import UNetAuthor, UNet
import pytorch_lightning as pl

import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# parser = argparse.ArgumentParser(description="onnx")
# parser.add_argument("--folder_out",
#                     type=str,
#                     default='onnx_models/',
#                     help="pth save")
# parser.add_argument("--checkpoint_path",
#                     type=str,
#                     default='/my_trained_models/dncnn.pth',
#                     help="pth network")
# parser.add_argument("--architecture",
#                     type=str,
#                     default='DnCNN_nobn',
#                     help="architecture")

# opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def convert_model(model_type,
                  folder_out='onnx_models/',
                  save_name='model.onnx',
                  checkpoint_path='pth/to/my/model.pth'):

    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
        print('Output folder [{}] created.'.format(folder_out))
    else:
        print('Output folder [{}] exists!'.format(folder_out))
    if 'dncnn' in model_type.lower():
        model, net_name, _, _ = get_model(model_type,
                                        n_ch=1,
                                        checkpoint_pth=checkpoint_path)
        print('checkpoint loaded from {}'.format(checkpoint_path))

        # fileName = os.path.basename(checkpoint_path)
        # fileName, _ = os.path.splitext(fileName)

        input_names = ['input_image']
        output_names = ['output_image']
        save_pth = folder_out + save_name + ".onnx"
        with torch.no_grad():
            im_in = torch.randn((1, 1, 512, 512)).type(Tensor)
            torch.onnx.export(model.module,
                            im_in,
                            save_pth,
                            verbose=True,
                            input_names=input_names,
                            output_names=output_names)
        print('Success!')
    elif 'unet' in model_type.lower():
        model = UNetAuthor(in_nc = 1, out_nc = 1, nb = 2, nc=[32, 64, 128, 256])
        checkpoint = torch.load(checkpoint_path)
        print('checkpoint loaded from {}'.format(checkpoint_path))
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, val in state_dict.items():
            new_name = ('.').join(key.split('.')[2:])
            new_state_dict[new_name] = val
        model.load_state_dict(new_state_dict, strict=False)

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
                            input_names=input_names,
                            output_names=output_names)
        print('Success!')
    else:
        print('Model not available!')


# if __name__ == '__main__':

#     convert_model(architecture,
#                   folder_out,
#                   save_name,
#                   checkpoint_path)
