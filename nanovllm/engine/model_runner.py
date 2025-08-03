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
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """_summary_

        Args:
            config (Config): Overall configuration for super-nano-vllm.
            rank (int): Rank used for distributed inference(tensor parallelism).
            event (Event | list[Event]): For child processes, this is a single Event object. For the main process, this is a list of Event objects for each child process.
        """
        self.config = config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel
        self.rank = rank
        self.shared_memory_size = 2 * GB  # size of shared memory for communication
        self.shm_lock = event

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
                    name="super-nano-vllm", create=True, size=self.shared_memory_size
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shared_memory = SharedMemory(name="super-nano-vllm")
                self.loop()

    def loop(self):
        """Looping ModelRunner process, and receive data from the main process."""
        while True:
            method_name, args = self._receive()
            self.call(method_name, *args)

            if method_name == "exit":
                break

    def _receive(self):
        """Receive data from the shared memory. This is used in child processes."""
        assert self.world_size > 1 and self.rank

        assert type(self.shm_lock) is Event, (
            "shm_lock should be a Event objects in the child process."
        )

        self.shm_lock.wait()  # wait until the main process finish writing data to shared memory

        data_size = int.from_bytes(self.shared_memory.buf[0:4], "little")

        method_name, *args = pickle.loads(self.shared_memory.buf[4 : data_size + 4])

        # clear the lock after receiving data
        self.shm_lock.clear()

        return method_name, args

    def send(self, method_name, *args):
        """Send data from main process to child processes. This is used in the main process."""
        assert self.world_size > 1 and self.rank == 0

        data = pickle.dumps((method_name, *args))
        data_size = len(data)

        assert data_size + 4 <= self.shared_memory_size, (
            "Data size exceeds shared memory size."
        )

        # write data size
        self.shared_memory.buf[0:4] = data_size.to_bytes(4, "little")

        # write data
        self.shared_memory.buf[4 : data_size + 4] = data

        # notify other processes
        assert type(self.shm_lock) is list[Event], (
            "shm_lock should be a list of Event objects in the main process."
        )
        for child_shm_lock in self.shm_lock:
            child_shm_lock.set()

    def call(self, method_name: str, *args):
        """Call method on the model runner. If it is main process, it writes to shared memory."""
        if self.world_size > 1 and self.rank == 0:
            self.send(method_name, *args)
        method = getattr(self, method_name, None)
        assert method is not None, f"Method {method_name} not found in ModelRunner."

        return method(*args)
