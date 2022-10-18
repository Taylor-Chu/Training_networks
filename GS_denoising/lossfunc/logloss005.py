import torch
import torch.nn as nn


class LogLoss005(nn.Module):
    def __init__(self, reduction = 'sum') -> None:
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=self.reduction)

    def forward(self, input, target) -> torch.Tensor:
        x_log = torch.log10(1000. * input + 51.) / 3.0
        y_log = torch.log10(1000. * target + 51.) / 3.0
        return self.mse(x_log, y_log)
