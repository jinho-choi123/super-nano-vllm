from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class LinearBase(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, tp_dim: Optional[int]):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__(input_dim=input_dim, output_dim=output_dim, tp_dim=None)

        self.weight = nn.Parameter(torch.empty(self.output_dim, self.input_dim))
        self.use_bias = bias

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_dim))
        else:
            self.bias = None

    def load_weight(self, loading_weight: torch.Tensor):
        self.weight.data.copy_(loading_weight)

    def load_bias(self, loading_bias: torch.Tensor):
        assert self.use_bias
        assert self.bias
        self.bias.data.copy_(loading_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
