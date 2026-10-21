"""nanovllm_voxcpm.engine.block_manager

This module manages the shared KV-cache *block pool*.

In this runtime, Attention KV memory is allocated as a fixed number of blocks
(``num_kvcache_blocks``), each block holding ``block_size`` token positions. A
sequence does not own contiguous KV memory; instead it owns a ``block_table``
(a list of block ids) that maps logical token positions to physical storage.

Responsibilities
----------------
- Admission control: decide whether a new sequence can be allocated
  (:meth:`BlockManager.can_allocate`).
- Allocate/free physical blocks for a sequence (:meth:`allocate`,
  :meth:`deallocate`).
- Prepare for incremental decode by ensuring a writable slot exists for the
  current last token (:meth:`can_append`, :meth:`may_append`).
- Prefix KV cache reuse ("prefix caching"): reuse already-computed KV blocks for
  identical prefixes across requests.

Prefix caching: how it works
----------------------------
For *full* blocks (exactly ``block_size`` tokens), we compute a rolling hash of
token ids. The hash of block i includes the hash of block i-1, so the hash is a
function of the entire prefix up to that block.

This is crucial because transformer KV states depend on the full causal prefix;
two identical blocks with different prefixes must not share KV memory.

Allocation uses the hash chain:
- If we find an existing block with the same rolling hash *and* identical
  token ids, we treat it as a cache hit and increase its reference count.
- If there is any cache miss at block k, all subsequent blocks must be treated
  as misses as well (because their prefixes differ).
- A LoRA adapter changes the KV of every position: the chain of an adapter's
  sequence starts from a seed derived from its ``adapter_id`` (ids are never
  reused), base-model sequences from none, so equal tokens in two adapters
  never share a block.
- The last block is never taken from the cache: the last token always runs
  (its prefill produces the first output), and a block other sequences read is
  never written.

Hash-entry lifecycle
--------------------
``hash_to_block_id`` maps a rolling hash to the block that currently holds the
matching content. An entry is created when a full block is registered as a
cache candidate (:meth:`BlockManager._publish`) and dropped when the block is
recycled for different content (:meth:`BlockManager._allocate_block`).
Releasing a block intentionally keeps the entry: a freed block's KV stays
intact until the block is handed to a new owner, so a later request with the
same prefix can "resurrect" it as a cache hit. A stale entry would otherwise
point at a recycled block whose new content has equal tokens but another prefix.

Reference counting
------------------
Blocks are reference-counted so multiple sequences can share KV blocks when
prefix caching hits. A block is returned to the free list only when
``ref_count`` drops to zero.

Collaboration with the scheduler
--------------------------------
:class:`~nanovllm_voxcpm.engine.scheduler.Scheduler` is the only policy layer.
It calls into :class:`BlockManager` to:
- allocate blocks when moving a sequence into RUNNING (prefill);
- free blocks when a sequence finishes/cancels/preempts;
- check/prepare capacity before decode steps.

Concrete example: VoxCPM
------------------------
VoxCPM requests encode both text and audio context. In
``nanovllm_voxcpm/model/engine.py``, the sequence's ``token_ids``
(``hash_tokens``) contains:

- ``int`` token ids for (prompt_text + target_text) plus a special
  ``audio_start_token``;
- ``bytes`` chunks for prompt audio latents (each patch serialized via
  ``tobytes()``);
- during generation, each predicted latent patch is appended as ``bytes``.

This mixed stream is sufficient to uniquely identify the causal prefix, which
means prefix caching can safely reuse KV blocks across requests that share the
same text + prompt-audio prefix.
"""

from collections import deque
import xxhash
from nanovllm_voxcpm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int | bytes]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int | bytes], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        for it in token_ids:
            if isinstance(it, int):
                h.update(it.to_bytes(8, "little"))
            else:
                h.update(it)
        return h.intdigest()

    _CHAIN_SEEDS: dict[int, int] = {}  # adapter id -> its chain seed (a pure function of the id)

    @staticmethod
    def chain_seed(seq: Sequence) -> int:
        """The hash chain's start for ``seq``: its LoRA adapter's (-1 = none, the base model)."""
        adapter_id = getattr(seq, "adapter_id", None)
        if adapter_id is None:
            return -1
        seed = BlockManager._CHAIN_SEEDS.get(adapter_id)
        if seed is None:
            seed = BlockManager._CHAIN_SEEDS[adapter_id] = xxhash.xxh64_intdigest(
                b"lora-adapter" + int(adapter_id).to_bytes(8, "little", signed=True))
        return seed

    def _allocate_block(self, block_id: int) -> Block:
        """Take a free block into use; a recycled block's old content (and its hash entry) ends here."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            self.hash_to_block_id.pop(block.hash)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _publish(self, block: Block, hash: int, token_ids: list[int | bytes]) -> None:
        """Register a full block as a prefix-cache candidate (the only writer of ``hash_to_block_id``)."""
        block.update(hash, token_ids)
        self.hash_to_block_id[hash] = block.block_id

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        num_blocks = seq.num_blocks
        last = num_blocks - 1
        if num_blocks == 1 and seq.num_tokens < self.block_size:  # one partial block (a short prompt): never cached,
            seq.block_table.append(self._allocate_block(self.free_block_ids[0]).block_id)  # hashed or published
            return
        h = self.chain_seed(seq)
        cache_miss = False
        for i in range(num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # the last block is computed: a fully cached prompt still prefills (and owns) the block of its last token
            if block_id == -1 or i == last or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                self._publish(block, h, token_ids)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            if last_block.hash == -1:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else self.chain_seed(seq)
                h = self.compute_hash(token_ids, prefix)
                self._publish(last_block, h, token_ids)
        else:
            assert last_block.hash == -1
