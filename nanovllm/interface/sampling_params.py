from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0

    # max decoding token
    max_tokens: int = 64 

    ignore_eos: bool = False