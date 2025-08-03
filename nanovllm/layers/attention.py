import torch
from torch import nn
import triton
import triton.language as tl


class AttentionBase(nn.Module):
    def __init__(self, num_head, head_dim, scale, num_kv_head):
        super().__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_head = num_kv_head

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """forward method of AttentionBase class

        Args:
            q (torch.Tensor(seq_len, num_head, head_dim)): Query
            k (torch.Tensor(seq_len, num_head, head_dim)): Key
            v (torch.Tensor(seq_len, num_head, head_dim)): Value

        Returns:
            torch.Tensor: Output tensor after attention computation
        """
        pass
