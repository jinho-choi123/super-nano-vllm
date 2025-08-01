import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self, act_tensor: torch.Tensor, scaling_tensor: torch.Tensor
    ) -> torch.Tensor:
        return F.silu(act_tensor) * scaling_tensor
