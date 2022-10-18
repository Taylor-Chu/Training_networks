

import math

import torch
import torch.nn as nn
import torch.nn.init as init
# import torch.optim as optim


class MODEL(nn.Module):

    def __init__(self, num_residual_units = 64, num_blocks = 16, width_multiplier = 4, scale=1):
        super(MODEL, self).__init__()
        #self.temporal_size = temporal_size
        self.temporal_size = None
        self.image_mean = 0.5
        kernel_size = 3
        skip_kernel_size = 5
        weight_norm = torch.nn.utils.weight_norm
        num_inputs = 1
        if self.temporal_size:
            num_inputs *= self.temporal_size
        num_outputs = scale * scale * num_inputs

        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_inputs,
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        for _ in range(num_blocks):
            body.append(
                Block(
                    num_residual_units,
                    kernel_size,
                    width_multiplier,
                    weight_norm=weight_norm,
                    res_scale=1 / math.sqrt(num_blocks),
                ))
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                num_outputs,
                kernel_size,
                padding=kernel_size // 2))
        init.ones_(conv.weight_g)
        init.zeros_(conv.bias)
        body.append(conv)
        self.body = nn.Sequential(*body)

        skip = []
        if num_inputs != num_outputs:
            conv = weight_norm(
                nn.Conv2d(
                    num_inputs,
                    num_outputs,
                    skip_kernel_size,
                    padding=skip_kernel_size // 2))
            init.ones_(conv.weight_g)
            init.zeros_(conv.bias)
            skip.append(conv)
        self.skip = nn.Sequential(*skip)

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))    
        self.shuf = nn.Sequential(*shuf)

    def forward(self, x):
        if self.temporal_size:
            x = x.view([x.shape[0], -1, x.shape[3], x.shape[4]])
        with torch.no_grad():
            x -= self.image_mean
        x = self.body(x) + self.skip(x)
        x = self.shuf(x)
        with torch.no_grad():
            x += self.image_mean
        if self.temporal_size:
            x = x.view([x.shape[0], -1, 1, x.shape[2], x.shape[3]])
        return x


class Block(nn.Module):

    def __init__(self,
                num_residual_units,
                kernel_size,
                width_multiplier=1,
                weight_norm=torch.nn.utils.weight_norm,
                res_scale=1):
        super(Block, self).__init__()
        body = []
        conv = weight_norm(
            nn.Conv2d(
                num_residual_units,
                int(num_residual_units * width_multiplier),
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, 2.0)
        init.zeros_(conv.bias)
        body.append(conv)
        body.append(nn.ReLU(True))
        conv = weight_norm(
            nn.Conv2d(
                int(num_residual_units * width_multiplier),
                num_residual_units,
                kernel_size,
                padding=kernel_size // 2))
        init.constant_(conv.weight_g, res_scale)
        init.zeros_(conv.bias)
        body.append(conv)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x) + x
        return x