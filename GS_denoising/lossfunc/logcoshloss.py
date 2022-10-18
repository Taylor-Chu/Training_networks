import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    '''
    https://github.com/tuantle/regression-losses-pytorch
    '''
    def __init__(self, reduction = 'sum') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target) -> torch.Tensor:
        '''
        Considering the precision of float, 1e-12 might not effect the result
        '''

        delta = target - input
        l_i = torch.log(torch.cosh(delta + 1e-12))
        result = None
        if self.reduction == 'sum':
            result = torch.sum(l_i)
        elif self.reduction == 'none':
            result = l_i
        elif self.reduction == 'mean':
            result = torch.mean(l_i)
            
        return result
