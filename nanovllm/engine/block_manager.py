from collections import deque
from typing import Optional
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


def calc_hash(prev_block_hash: Optional[int], block_token_ids: list[int]) -> int:
    assert len(block_token_ids) > 0
    h = xxhash.xxh64()
    if prev_block_hash is not None:
        h.update(prev_block_hash.to_bytes(8, "little"))
    h.update(np.array(block_token_ids).tobytes())
    return h.intdigest()


class Block:
    def __init__(self, block_id):
        self.block_id: int = block_id
        self.ref_cnt: int = 0
        self.hash: Optional[int] = None
        self.token_ids = []

    # token_ids: token ids that fit in the block
    # prev_block_hash: hash of the previous block
    def update(self, token_ids: list[int], prev_block_hash: Optional[int]):
        self.token_ids = token_ids
        self.hash = calc_hash(prev_block_hash, token_ids)

    def ref(self) -> int:
        self.ref_cnt += 1
        return self.ref_cnt

    def deref(self) -> int:
        self.ref_cnt -= 1
        assert self.ref_cnt >= 0
        return self.ref_cnt

    def reset(self):
        assert self.ref_cnt == 0
        self.ref_cnt = 0
        self.hash = None
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0

        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        self.hash_to_block_id: dict[int, int] = dict()

        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]

        # reset the block
        block.reset()

        # reference the block
        block.ref()

        # check
        assert block_id in self.free_block_ids
        assert block_id not in self.used_block_ids

        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)

        return block

    def _deallocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]

        # deref the block
        block.deref()

        # reset the block
        block.reset()

        # check
        assert block_id not in self.free_block_ids
        assert block_id in self.used_block_ids

        self.free_block_ids.append(block_id)
        self.used_block_ids.remove(block_id)

        return block

    def can_allocate_for_prompt_tokens(self, seq: Sequence):
        return len(self.free_block_ids) >= seq.num_blocks_for_prompt_tokens

    # allocate prompt token's kv cache block for sequence
    # return false if fail
    def allocate_for_prompt(self, seq: Sequence) -> bool:
        # we assume that seq is fresh when this method is called
        assert len(seq.block_table) == 0

        if not self.can_allocate_for_prompt_tokens(seq):
            return False

        prev_block_hash = None

        token_ids = seq.prompt_token_ids

        for i in range(seq.num_blocks_for_prompt_tokens):
            block_token_ids = token_ids[i * seq.block_size : (i + 1) * self.block_size]

            curr_block_hash = calc_hash(
                prev_block_hash=prev_block_hash, block_token_ids=block_token_ids
            )

            # check if we have cached block
            cached_block_id = (
                self.hash_to_block_id.get(curr_block_hash, None)
                if curr_block_hash
                else None
            )

            # for the last block(a.k.a i == seq_num_blocks_for_prompt_tokens - 1), we don't use cached block.
            # this makes the nano vllm no need to care about copy-on-write situation.
            if i == seq.num_blocks_for_prompt_tokens - 1:
                cached_block_id = None
                free_block_id = self.free_block_ids[0]
                self._allocate_block(free_block_id)
                break

            if cached_block_id:
                seq.block_table.append(cached_block_id)
                cached_block = self.blocks[cached_block_id]

                cached_block.ref()
            else:
                free_block_id = self.free_block_ids[0]
                self._allocate_block(free_block_id)

                self.hash_to_block_id[curr_block_hash] = free_block_id

            # update prev_block_hash
            prev_block_hash = curr_block_hash

        return True

    def allocate_for_decode(self, seq: Sequence) -> None:
        # allocate more space for decode
        if not seq.is_blocks_full:
            return
        else:
            # allocate free block
            free_block_id = self.free_block_ids[0]
            self._allocate_block(free_block_id)

    def deallocate(self, seq: Sequence):
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.deref()

            if block.ref_cnt == 0:
                block.reset()

        seq.reset()
