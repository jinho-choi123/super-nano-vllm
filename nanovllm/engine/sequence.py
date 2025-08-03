from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.interface.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()  # nothing is processed
    RUNNING = auto()  # partially processed
    FINISHED = auto()  # all tokens are processed


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, prompt_token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id: int = next(Sequence.counter)
        self.status: SequenceStatus = SequenceStatus.WAITING

        # used when sequence is reset
        self.prompt_token_ids: list[int] = copy(prompt_token_ids)

        self.token_ids: list[int] = copy(prompt_token_ids)

        self.num_prompt_tokens: int = len(self.token_ids)
        self.num_processed_tokens: int = 0

        self.num_blocks_for_prompt_tokens: int = (
            self.num_prompt_tokens + self.block_size - 1
        ) // self.block_size

        # block ids that is used by current sequence
        self.block_table: list[int] = []

        self.sampling_params: SamplingParams = sampling_params

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_decode_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def decode_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self) -> int:
        return len(self.block_table)

    @property
    def last_block_num_tokens(self) -> int:
        return (
            self.num_processed_tokens - (self.num_cached_blocks - 1) * self.block_size
        )

    @property
    def is_blocks_full(self) -> bool:
        # check if there is a room for new tokens
        return self.last_block_num_tokens < self.block_size

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    # if the sequence is preempt, then it is reset
    def reset(self) -> None:
        if self.status != SequenceStatus.FINISHED:
            self.status = SequenceStatus.WAITING
        self.token_ids = self.prompt_token_ids
        self.num_processed_tokens = 0
        self.block_table = []
