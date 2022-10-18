import torch
import torch.nn as nn


class LogL1Loss(nn.Module):
    def __init__(self, reduction='sum') -> None:
        super().__init__()
        self.reduction = 'sum'
        self.mae = nn.L1Loss(reduction=self.reduction)

    def forward(self, input, target) -> torch.Tensor:
        x_log = torch.log10(1000. * input + 1.) / 3.0
        y_log = torch.log10(1000. * target + 1.) / 3.0
        return self.mae(x_log, y_log)
