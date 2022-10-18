import torch
import torch.nn as nn

import math

from .basic_models import simple_CNN


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


def get_model(architecture, n_ch=1, n_ch_in=None, checkpoint_pth=None):
    """
    Returns the model associated to 'architecture'
    Arguments:
        architecture (str, optional): architecture of interest;
        n_ch (int, optional): number of output channels;
        n_ch_in (int, optional): number of input channels;
    Returns:
        model: the model;
        net_name (str): the name of the model;
        clip_val (float): the clipping value for the gradient during training;
        lr (float): the learning rate;
    """

    cuda = True if torch.cuda.is_available() else False
    if n_ch_in is None:
        n_ch_in = n_ch

    if 'DnCNN_nobn' in architecture:
        clip = 'relu'
        nl_type = 'relu'

        if 'sigmoid' in architecture:
            clip = 'sigmoid'
        elif 'relus1' in architecture:
            clip = 'relus1'
        elif 'relus05' in architecture:
            clip = 'relus05'
        elif 'relus01' in architecture:
            clip = 'relus01'
        elif 'relus001' in architecture:
            clip = 'relus001'
        elif 'relus005' in architecture:
            clip = 'relus005'
        elif 'relu' in architecture:
            clip = 'relu'
        elif 'tanhp1' in architecture:
            clip = 'tanhp1'
            
        if 'act_mish' in architecture:
            nl_type = 'mish'
        
        net = simple_CNN(n_ch_in=n_ch_in,
                         n_ch_out=n_ch,
                         n_ch=64,
                         nl_type=nl_type,
                         depth=20,
                         bn=False,
                         clip=clip)
        if checkpoint_pth is not None:
            cp = torch.load(checkpoint_pth,
                            map_location=lambda storage, loc: storage)
            net.load_state_dict(cp)
        clip_val, lr = 1e-2, 1e-4
        net_name = 'dncnn_nobn_nch_' + str(n_ch_in)

        if clip:
            net_name += '_' + clip


    if 'DnCNN' in architecture and checkpoint_pth is None:
        net.apply(weights_init_kaiming)

    # Move to GPU if possible
    if cuda:
        print("cuda driver found - training on GPU.\n")
        net.cuda()
    else:
        print("no cuda driver found - training on CPU.\n")

    if cuda:
        model = nn.DataParallel(net).cuda()
    else:
        model = nn.DataParallel(net)

    return model, net_name, clip_val, lr
