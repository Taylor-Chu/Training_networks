from torchmetrics import Metric
import torch
from torch import Tensor
import numpy as np
import pylab

def snr_calc(out, true):
    x_np = out.detach().cpu().numpy()
    y_np = true.detach().cpu().numpy()
    return torch.tensor(20*np.log10(pylab.norm(x_np.flatten()+1e-6) /
                            pylab.norm(x_np.flatten() - y_np.flatten() + 1e-6)))

class snr(Metric):
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state('snr_sum', default = torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state('total', default = torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, out, true):
        snr_batch = snr_calc(out, true)
        self.snr_sum += snr_batch.sum()
        self.total += snr_batch.numel()


    def compute(self):
        return self.snr_sum / self.total

