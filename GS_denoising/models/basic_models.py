import sys
sys.path.append('/work/sc004/sc004/tc1213/Prox-Pnp2/GS_denoising/activations')
import torch
import torch.nn as nn
from activations import TanhP1, ReLUS1, ReLUS05, ReLUS01, ReLUS001, ReLUS005

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class simple_CNN(nn.Module):
    def __init__(self, n_ch_in=3, n_ch_out=3, n_ch=64, nl_type='relu', depth=5, bn=False, clip = None):
        super(simple_CNN, self).__init__()

        self.nl_type = nl_type
        self.depth = depth
        self.bn = bn
        self.clipFn = None

        self.in_conv = nn.Conv2d(n_ch_in, n_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_list = nn.ModuleList([nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1, bias=True)
                                        for _ in range(self.depth-2)])
        self.out_conv = nn.Conv2d(n_ch, n_ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        if self.nl_type == 'relu':
            self.nl_list = nn.ModuleList([nn.LeakyReLU() for _ in range(self.depth-1)])
        elif self.nl_type == 'mish':
            self.nl_list = nn.ModuleList([nn.Mish() for _ in range(self.depth-1)])
            
        if self.bn:
            self.bn_list = nn.ModuleList([nn.BatchNorm2d(n_ch) for _ in range(self.depth-2)])

        if clip:
            if clip == 'sigmoid':
                self.clipFn = nn.Sigmoid()
            elif clip == 'relu':
                self.clipFn = nn.ReLU()
            elif clip == 'tanhp1':
                self.clipFn = TanhP1()
            elif clip == 'relus1':
                self.clipFn = ReLUS1()
            elif clip == 'relus05':
                self.clipFn = ReLUS05()
            elif clip == 'relus01':
                self.clipFn = ReLUS01()
            elif clip == 'relus001':
                self.clipFn = ReLUS001()
            elif clip == 'relus005':
                self.clipFn = ReLUS005()
            else:
                raise NotImplementedError(clip + ' not found')
        

    def forward(self, x_in):

        x = self.in_conv(x_in)
        x = self.nl_list[0](x)

        for i in range(self.depth-2):
            x_l = self.conv_list[i](x)  
            if self.bn:
                x_l = self.bn_list[i](x_l)
            x = self.nl_list[i+1](x_l)
        
        x_out = self.out_conv(x)+x_in

        if self.clipFn:
            x_out = self.clipFn(x_out)

        return x_out
