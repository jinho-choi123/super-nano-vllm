import os
from dataclasses import dataclass
from typing import Optional
from transformers import AutoConfig, PretrainedConfig
import torch


@dataclass
class Config:
    model_id: str
    model_path: str

    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512

    gpu_memory_utilization: float = 0.6

    tensor_parallel: int = 1
    enforce_eager: bool = False

    eos_token_id: int = -1

    # how many tokens fit in a single kv cache block
    kv_cache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # default dtype if huggingface config's default detype is not set
    default_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        assert os.path.isdir(self.model_path)
        assert self.kv_cache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel <= 8
        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(self.model_id)

        assert self.hf_config.max_position_embeddings is not None

        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
