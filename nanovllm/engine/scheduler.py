from collections import deque

from nanovllm.interface.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.interface.scheduler_output import SchedulerOutput


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos_token_id
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kv_cache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add_sequence(self, new_seq: Sequence):
        self.waiting.append(new_seq)

    def schedule(self) -> SchedulerOutput:
        scheduled_sequences = []
        finished_sequences = []
        num_compute_tokens = 0

        # check if there are finished sequences
        for seq in self.running:
            if seq.is_finished:
                # FIXME: this may be the bottleneck for the scheduler
                self.running.remove(seq)
                finished_sequences.append(seq)
                continue

        # prefill has more priority than decode

        # prefill
        while self.waiting and len(scheduled_sequences) < self.max_num_seqs:
            seq = self.waiting[0]

            # check if it exceeds the max number of batched tokens
            if seq.num_prompt_tokens + num_compute_tokens > self.max_num_batched_tokens:
                break

            self.block_manager.allocate_for_prompt(seq)

            num_compute_tokens += seq.num_prompt_tokens

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()

            self.running.append(seq)
            scheduled_sequences.append(seq)

        # if there are sequences to prefill, then return early
        if scheduled_sequences:
            return SchedulerOutput(
                scheduled_sequences=scheduled_sequences,
                finished_sequences=finished_sequences,
                num_compute_tokens=num_compute_tokens,
            )

        # decode
        while (
            self.running
            and len(scheduled_sequences) < self.max_num_seqs
            and num_compute_tokens < self.max_num_batched_tokens
        ):
            seq = self.running.popleft()

            can_be_scheduled = True

            # check if the block manager can allocate a block for the sequence decoding
            # if not, then we deallocate other sequences to get space
            while not self.block_manager.allocate_for_decode(seq):
                if self.running:
                    # if there are other running sequences, then we preempt it
                    self.preempt(self.running.pop())
                else:
                    # if there are no other running sequences, then we preempt itself
                    self.preempt(seq)
                    can_be_scheduled = False
                    break

            if can_be_scheduled:
                assert seq.status == SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                num_compute_tokens += 1  # we only decode one token at a time

        return SchedulerOutput(
            scheduled_sequences=scheduled_sequences,
            finished_sequences=finished_sequences,
            num_compute_tokens=num_compute_tokens,
        )

    def preempt(self, seq: Sequence):
        """Preempt a sequence and reset its status to waiting.

        Args:
            seq (Sequence): Sequence to preempt. It should be a sequence in the running state.

        Note:
            The argument `seq` should be a sequence that is currently running. However, it is not checked because it may be temporarily removed from the running state. See the `schedule` method.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
