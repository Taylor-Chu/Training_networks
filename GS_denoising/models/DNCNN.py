import models.basicblock as B
import torch
import torch.nn as nn

import math

# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('Linear') != -1:
#         nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(
#             -0.025, 0.025)
#         nn.init.constant(m.bias.data, 0.0)

class dncnn(nn.Module):
    def __init__(self, nc_in,nc_out,nb,mode,act_mode,bias,nf=64):
        super(dncnn, self).__init__()
        
        self.m_head = B.conv(nc_in, nf, mode=mode, bias=bias)
        self.m_body = B.conv(nf, nf, mode=mode + act_mode, bias=bias)
        self.m_tail = B.conv(nf, nc_out, mode=mode, bias=bias)
        self.clip = nn.ReLU()
        # net = B.sequential(m_head, *m_body, m_tail)
        
        def forward(self, x0):
            x = self.m_head(x0)
            for _ in range(nb - 2):
                x = self.m_body(x)
            x = self.m_tail(x)
            x = self.clip(x+x0)

        
        # net.apply(weights_init_kaiming)
        # model = nn.DataParallel(model).cuda()

            return x
