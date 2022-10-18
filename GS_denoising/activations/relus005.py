import torch
import torch.nn as nn

class ReLUS005(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.relu(input + 0.05) - 0.05