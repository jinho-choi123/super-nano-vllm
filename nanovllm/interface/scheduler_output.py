from dataclasses import dataclass
from nanovllm.engine.sequence import Sequence


@dataclass
class SchedulerOutput:
    """SchedulerOutput is the output of the scheduler."""

    scheduled_sequences: list[Sequence]

    finished_sequences: list[Sequence]

    num_compute_tokens: int
