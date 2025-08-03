import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.interface.config import Config
from nanovllm.engine.sequence import Sequence

MB = 2**10
GB = 2**10 * MB


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event):
        self.config = config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel
        self.rank = rank
        self.event = event

        # define communication protocol between multiple model runners
        # this is used to synchronize for tensor parallelism
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:23456",
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)

        self.default_dtype = (
            config.hf_config.torch_dtype
            if isinstance(config.hf_config.torch_dtype, torch.dtype)
            else config.default_dtype
        )

        # set torch default dtype
        torch.set_default_dtype(self.default_dtype)

        # set default device
        torch.set_default_device("cuda")

        # communication is done through shared memory
        if self.world_size > 1:
            if rank == 0:
                self.shared_memory = SharedMemory(
                    name="super-nano-vllm", create=True, size=2 * GB
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shared_memory = SharedMemory(name="super-nano-vllm")
                self.loop()

    def loop(self):
        """Looping ModelRunner process, and receive data from the main process."""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)

            if method_name == "exit":
                break

    def read_shm(self):
        """Read data from shared memory."""
        assert self.world_size > 1 and self.rank
        self.event.wait()

        data_size = int.from_bytes(self.shared_memory.buf[0:4], "little")

        method_name, *args = pickle.loads(self.shared_memory.buf[4 : data_size + 4])

        self.event.clear()

        return method_name, args

    def write_shm(self, method_name, *args):
        """Write data to shared memory."""
        assert self.world_size > 1 and self.rank == 0

        data = pickle.dumps((method_name, *args))
        data_size = len(data)

        # write data size
        self.shared_memory.buf[0:4] = data_size.to_bytes(4, "little")

        # write data
        self.shared_memory.buf[4 : data_size + 4] = data

        # notify other processes
        self.event.set()

    def call(self, method_name: str, *args):
        """Call method on the model runner. If it is main process, it writes to shared memory."""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert method is not None, f"Method {method_name} not found in ModelRunner."

        return method(*args)
