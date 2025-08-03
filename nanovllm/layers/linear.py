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
        self.tp_size = (
            dist.get_world_size()
        )  # number of workers participating in tensor parallelism

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


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim, tp_dim=0)
        self.input_dim_per_partition = input_dim

        assert output_dim % self.tp_size == 0
        self.output_dim_per_partition = output_dim // self.tp_size

        self.use_bias = bias

        self.weight = nn.Parameter(
            torch.empty(self.output_dim_per_partition, self.input_dim_per_partition)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_dim_per_partition))
        else:
            self.bias = None

    def load_weight(self, loading_weight: torch.Tensor):
        self.weight.data.copy_(loading_weight)

    def load_bias(self, loading_bias: torch.Tensor):
        assert self.use_bias
        assert self.bias is not None
        self.bias.data.copy_(loading_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """MergedColumnParallelLinear is used to merge multiple ColumnParallelLinear layers into one.

    For example, we can merge gate_proj(x) and up_proj(x) into a single MergedColumnParallelLinear layer. By merging, we can reduce the number of communication operations in tensor parallelism.

    Args:
        ColumnParallelLinear (_type_): _description_
    """

    def __init__(self, input_dim: int, output_dims: list[int], bias: bool = False):
        self.output_dims = output_dims
        super().__init__(input_dim=input_dim, output_dim=sum(output_dims), bias=bias)

    def load_weight(self, loading_weight: torch.Tensor, shard_id: int):  # type: ignore
        """Load weight for a specific shard. For example, if we have two shards for gate_proj and up_proj,
        we can load the weight for each shard separately.

        Args:
            loading_weight (torch.Tensor): _description_
            shard_id (int): _description_
        """
        assert self.tp_dim is not None, (
            "tp_dim should be set for MergedColumnParallelLinear."
        )

        # we shared the self.weight tensor
        shard_offset = sum(self.output_dims[:shard_id]) // self.tp_size
        shard_size = self.output_dims[shard_id] // self.tp_size

        # split the loading weight into tp_size pieces
        loading_weights = loading_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_data = self.weight.data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loading_weights)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_head: int,
        num_kv_head: int,
        bias: bool = False,
    ):
        self.head_dim = head_dim
        self.num_head = num_head
        self.num_kv_head = num_kv_head

        self.tp_size = dist.get_world_size()

        assert num_head % self.tp_size == 0
        assert num_kv_head % self.tp_size == 0
        self.num_head_per_partition = num_head // self.tp_size
        self.num_kv_head_per_partition = num_kv_head // self.tp_size

        input_dim = hidden_dim
        output_dim = (
            (self.num_head + 2 * self.num_kv_head) * self.head_dim
        )  # number of query, key, value == self.num_head + 2 * self.num_kv_head

        super().__init__(input_dim=input_dim, output_dim=output_dim, bias=bias)

    def load_weight(self, loading_weight: torch.Tensor, shard_id: str):  # type: ignore
        assert shard_id in ["query", "key", "value"]

        if shard_id == "query":
            shard_size = self.num_head * self.head_dim
            shard_offset = 0
        elif shard_id == "key":
            shard_size = self.num_kv_head * self.head_dim
            shard_offset = self.num_head * self.head_dim
        elif shard_id == "value":
            shard_size = self.num_kv_head * self.head_dim
            shard_offset = (self.num_head + self.num_kv_head) * self.head_dim
        else:
            raise ValueError("shard_id must be one of query, key, value")

        assert self.tp_dim is not None, "tp_dim should be set for QKVParallelLinear."
        param_data = self.weight.data.narrow(self.tp_dim, shard_offset, shard_size)

        loading_weight = loading_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        param_data.copy_(loading_weight)


class RowParallelLinear(LinearBase):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__(input_dim=input_dim, output_dim=output_dim, tp_dim=1)

        assert input_dim % self.tp_size == 0
        self.input_dim_per_partition = input_dim // self.tp_size

        self.output_dim_per_partition = output_dim

        self.weight = nn.Parameter(
            torch.empty(self.output_dim, self.input_dim_per_partition)
        )

        self.use_bias = bias

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_dim_per_partition))
        else:
            self.bias = None

    def load_weight(self, loading_weight: torch.Tensor):
        """Load weights.

        Args:
            loading_weight (torch.Tensor): _description_
        """
        assert self.tp_dim is not None, "tp_dim should be set for RowParallelLinear."

        param_data = self.weight.data

        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        loading_weight = loading_weight.narrow(self.tp_dim, start_idx, shard_size)

        param_data.copy_(loading_weight)

    def load_bias(self, loading_bias: torch.Tensor):
        assert self.tp_dim is not None, "tp_dim should be set for RowParallelLinear"
        assert self.use_bias
        assert self.bias is not None

        # FIXME: we just have to load bias in the main worker(self.tp_rank == 0)
        self.bias.copy_(loading_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # synchronize the tensor parallelism
        if self.tp_size > 1:
            dist.all_reduce(y)

        return y
