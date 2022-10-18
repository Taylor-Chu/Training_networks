import torch
import torch.nn as nn

class TanhP1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(input) + 1.0