import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def forward(self, act_tensor: torch.Tensor) -> torch.Tensor:
        var = act_tensor.pow(2).mean(dim=-1, keepdim=True)

        # element wise multiplication
        act_tensor = act_tensor * torch.rsqrt(var + self.eps)

        # matmul
        act_tensor = act_tensor @ self.weight

        return act_tensor
