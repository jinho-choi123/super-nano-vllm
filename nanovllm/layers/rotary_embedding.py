from functools import lru_cache

import torch
from torch import nn


def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., : x.shape[-1] // 2]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqeeze_dim=1
):
    cos = cos.unsqueeze(unsqeeze_dim)
    sin = sin.unsqueeze(unsqeeze_dim)

    x_emb = (x * cos) + (rotate_half(x) * sin)

    return x_emb


class RotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_position_embeddings: int, base: float = 100000000
    ):
        super().__init__()
        self.head_dim = head_dim

        # shape: (head_dim/2, )
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # shape: (max_position_embeddings,)
        index = torch.arange(max_position_embeddings)

        # shape: (max_position_embeddings, head_dim/2)
        freqs = torch.einsum("i, j -> ij", index, inv_freq)

        # shape: (max_position_embeddings, head_dim/2)
        cos = freqs.cos()

        # shape: (max_position_embeddings, head_dim/2)
        sin = freqs.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.compile
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        num_tokens = position_ids.shape[0]

        # get cos, sin values
        # shape: (num_tokens, head_dim/2)
        cos = self.cos[position_ids]  # type: ignore
        sin = self.sin[position_ids]  # type: ignore
        assert cos.shape == (num_tokens, self.head_dim / 2)
        assert sin.shape == (num_tokens, self.head_dim / 2)

        # the shape of x is (num_tokens, **, head_dim)
        assert x.shape[0] == num_tokens
        assert x.shape[2] == self.head_dim

        x_emb = apply_rotary_emb(x, cos, sin)

        return x_emb


@lru_cache(1)
def get_rope(head_dim: int, max_position: int, base: float):
    rotary_emb = RotaryEmbedding(
        head_dim=head_dim, max_position_embeddings=max_position, base=base
    )
    return rotary_emb
