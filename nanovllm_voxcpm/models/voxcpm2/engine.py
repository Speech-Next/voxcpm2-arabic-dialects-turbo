import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from itertools import compress, count, islice
from operator import ne

import numpy as np
import torch
import xxhash
from transformers import LlamaTokenizerFast

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from nanovllm_voxcpm.engine.sequence import Sequence
from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
from nanovllm_voxcpm.models.voxcpm2.lora_loader import load_voxcpm2_lora_checkpoint, tagged_adapter_state
from nanovllm_voxcpm.models.voxcpm2.runner import RunnerTask, VoxCPM2Payload, VoxCPM2Runner
from nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens


_SILENCE_RMS = 10 ** (-50 / 20)  # -50 dBFS
_CJK = re.compile("[\u4e00-\u9fff]")  # the tokenizer wrapper splits multi-character Chinese tokens
_LEADING_TAG = re.compile(r"<reserved_\d+>")  # a tagged model's dialect tag in front of the text


def _is_silent(wave) -> bool:
    """A streamed chunk (float in [-1, 1] or int16 PCM) below -50 dBFS RMS."""
    x = np.asarray(wave)
    if x.size == 0:
        return True
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    return float(np.sqrt(np.mean(x * x))) < _SILENCE_RMS


@dataclass
class VoxCPM2SeqPayload:
    feats: list[np.ndarray]
    text_tokens: list[int]
    feat_masks: list[bool]
    generated_waveforms: list[np.ndarray]
    temperature: float
    cfg_value: float
    decode_pad: np.ndarray | None = None
    max_generate_length: int | None = None
    seed: int | None = None
    seed_step: int = 0
    ref_key: int | None = None  # ref_prefix_cache bucket: xxh64 of the reference latents (+ LoRA adapter)
    feat_keys: list[int] | None = None  # per position: xxh64 of the audio patch (LocEnc cache), 0 = uncached
    prefix_block: int | None = None  # cached block claimed for the current prefill, its KV copy pending (None: miss)
    prefix_memo: tuple | None = None  # _prefix_match: (cache entry, compared length, matched tokens)
    ref_audio: "PreparedAudio | None" = None  # the reference the leading tokens were built from (its ref_hash)
    # head start (TurboConfig.head_tokens): the whole request, continued after this head sequence's patches
    chain: dict | None = None
    chained: bool = False  # the continuation was queued: this sequence's end is not the end of the stream
    patch_silent: list | None = None  # per generated patch: its audio below -50 dBFS (head start)


def _rows_from(arrays: list[np.ndarray], start: int) -> np.ndarray:
    """Rows ``start:`` of the concatenation of ``arrays`` (a prefill past a cached prefix copies only those)."""
    tail, off = [], 0
    for a in arrays:
        if off + a.shape[0] > start:
            tail.append(a[max(start - off, 0) :])
        off += a.shape[0]
    if not tail:
        return arrays[-1][:0]
    return tail[0] if len(tail) == 1 else np.concatenate(tail, axis=0)


@dataclass
class PreparedAudio:
    """An audio clip's latents in the forms add_request needs, computed once (registered voices)."""

    patches: np.ndarray  # [P, patch_size, feat_dim] float32
    patch_bytes: list[bytes]  # one hash token per patch (prefix caching)
    patch_keys: list[int]  # xxh64 per patch (LocEnc embedding cache)
    key: int  # xxh64 of all patches (reference-prefix KV cache bucket)
    decode_pad: np.ndarray  # last frames, VAE context for continuation
    # the clip as a reference ([ref start] patches [ref end]): the request's leading rows, built once per voice
    ref_tokens: list | None = None
    ref_masks: list | None = None
    ref_hash: list | None = None
    ref_keys: list | None = None
    ref_feats: np.ndarray | None = None


@dataclass
class RegisteredVoice:
    audio: PreparedAudio
    text: str | None  # transcript: enables the prompt (continuation) modes


class VoxCPM2Engine(LLMEngineBase):
    # turbo caches; class defaults = off (also for instances built without __init__)
    ref_prefix_cache_enabled = False
    locenc_cache_enabled = False
    head_tokens, head_patches, head_min_rest = 0, 3, 8  # head start off
    _pretok_cache: dict | None = None
    _tag_remap: dict = {}  # (instances built without __init__, e.g. in tests: no tagged adapters)
    tag_rows = _tag_rows_used = 0

    def __init__(self, config: Config[VoxCPM2Config]):
        self.n_decode_pad_frames = 12
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.audio_start_token = 101
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))
        # tagged adapters (a tagged model's LoRA and dialect tags): their requests' tag ids -> the extra embedding
        # rows after the vocabulary (LoRAConfig.tag_rows) that hold the tags' trained embeddings
        self.vocab_size = config.model_config.lm_config.vocab_size
        self.tag_rows = config.lora_config.tag_rows if config.lora_config is not None else 0
        self._tag_rows_used = 0
        self._tag_remap: dict[str, dict[int, int]] = {}
        turbo = config.model_config.turbo
        self.ref_prefix_cache_enabled = turbo.ref_prefix_cache
        self.ref_prefix_cache_size = turbo.ref_prefix_cache_size
        self.locenc_cache_enabled = turbo.locenc_cache_size > 0 and turbo.skip_text_locenc
        # ref_key -> (pinned KV block, token_ids it holds KV for, PreparedAudio of their reference), LRU order
        self.ref_prefix_blocks: OrderedDict[int, tuple[int, list, PreparedAudio | None]] = OrderedDict()
        self.voices: dict[str, RegisteredVoice] = {}
        self.head_tokens, self.head_patches, self.head_min_rest = turbo.head_tokens, turbo.head_patches, turbo.head_min_rest
        super().__init__(VoxCPM2Runner, config, config.tensor_parallel_size)

    def register_lora(self, name: str, path: str) -> int:
        payload = load_voxcpm2_lora_checkpoint(path, tp_size=self.model_runner.world_size)
        tagged = None
        if getattr(payload, "tag_embed", None) is not None:
            tagged = self._tagged_adapter(path, payload)
        adapter_id = super().register_lora(name, payload)
        if tagged is not None:
            remap, tokenizer = tagged
            self._tag_remap = {**self._tag_remap, name: remap}
            self._tag_rows_used += len(remap)
            if tokenizer is not None:
                self.tokenizer = tokenizer
                self._pretok_cache = self._pretok_pending = None
        return adapter_id

    def unregister_lora(self, name: str) -> None:
        super().unregister_lora(name)
        self._tag_remap = {k: v for k, v in self._tag_remap.items() if k != name}  # (its rows stay allocated)

    def _tagged_adapter(self, path: str, payload) -> tuple[dict[int, int], object]:
        """A tagged adapter's (tag id -> extra embedding row) map and the tokenizer that reads its tags as one token
        each (its own: the base vocabulary with the tags as special tokens; None = the engine's reads them already).
        Sets the payload's tag_row_start. Its requests, and only theirs, read the tags' trained rows."""
        n = payload.tag_embed.shape[0]
        if self._tag_rows_used + n > self.tag_rows:
            raise ValueError(f"a tagged adapter needs {n} extra embedding rows: LoRAConfig.tag_rows={self.tag_rows}, "
                             f"{self._tag_rows_used} in use")
        dialects = tagged_adapter_state(path)["dialects"]  # the order of tag_embed's rows
        with open(os.path.join(path, "dialects.json"), encoding="utf-8") as f:
            spec = json.load(f)["tags"]
        tags = [spec[d] for d in dialects]
        if len(tags) != n:
            raise ValueError(f"{n} tag embeddings for {len(tags)} dialects")
        tok = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path))
        base_vocab, vocab = self.tokenizer.tokenizer.get_vocab(), tok.tokenizer.get_vocab()
        if vocab != base_vocab:
            raise ValueError("a tagged adapter's tokenizer must have the base model's vocabulary (its tags special)")
        ids = []
        for t in tags:
            got = tok(t)
            if len(got) != 1:
                raise ValueError(f"the tag {t!r} is not one token in the tagged adapter's tokenizer: {got}")
            ids.append(got[0])
        engine_reads_tags = all(self.tokenizer(t) == [i] for t, i in zip(tags, ids))
        payload.tag_row_start = self.vocab_size + self._tag_rows_used
        remap = {i: payload.tag_row_start + k for k, i in enumerate(ids)}
        return remap, None if engine_reads_tags else tok

    def _prefix_match(self, seq: Sequence[VoxCPM2SeqPayload]) -> tuple[int, int]:
        """(cached block, matched tokens) for the longest prefix ``seq`` shares with the cached request of its voice.

        Equal ``token_ids`` (text ids and reference / prompt latent bytes) at equal positions mean equal KV, so
        the match is exact; it covers the reference audio and, for prompt+reference cloning, the prompt text too.
        The scheduler asks twice per admitted request (its budget hint, then on_seq_running): the result is kept
        for the same cache entry (the same tuple object) and length. A sequence's tokens only ever grow at the end,
        so its first ``n`` tokens, and the match, are the same for the same ``n``. When both requests took their
        reference from the same PreparedAudio object (a registered voice), their leading tokens are that object's
        ref_hash list, element for element: the comparison starts after it (a cloned voice is ~50 positions).
        """
        payload = seq.custom_payload
        entry = self.ref_prefix_blocks.get(payload.ref_key)
        if entry is None:
            return -1, 0
        block, tokens, ref = entry
        n = min(len(tokens), len(seq) - 1)  # at least one position must run to produce the first patch
        memo = payload.prefix_memo
        if memo is not None and memo[0] is entry and memo[1] == n:
            return block, memo[2]
        k = min(len(ref.ref_hash), n) if ref is not None and ref is payload.ref_audio else 0
        ids = seq.token_ids
        if n - k > 16:  # the first position from k whose tokens differ (or n), compared in C
            k = next(compress(count(k), map(ne, islice(ids, k, n), islice(tokens, k, None))), n)
        else:
            while k < n and ids[k] == tokens[k]:
                k += 1
        payload.prefix_memo = (entry, n, k)
        return block, k

    def cached_prefix_tokens(self, seq: Sequence[VoxCPM2SeqPayload]) -> int:
        if not self.ref_prefix_cache_enabled or seq.custom_payload.ref_key is None:
            return 0
        return self._prefix_match(seq)[1]

    def on_seq_running(self, seq: Sequence[VoxCPM2SeqPayload]) -> None:
        super().on_seq_running(seq)
        # claim the cached prefix right after block allocation (same scheduling pass as the budget hint)
        payload = seq.custom_payload
        payload.prefix_block = None
        if not self.ref_prefix_cache_enabled or payload.ref_key is None or seq.num_cached_tokens:
            return
        block, k = self._prefix_match(seq)
        if k:
            self.ref_prefix_blocks.move_to_end(payload.ref_key)
            payload.prefix_block = block
            seq.num_cached_tokens = k

    def on_prefill_admitted(self, seqs) -> None:
        """Copy the claimed reference-prefix KV into the newly admitted sequences' first blocks now, in one call, not
        at the start of their step's run: the copy (~0.9 ms of GPU time for 200 cloned requests) then runs while the
        host prepares the step instead of in front of its first kernel. A cached prefix ends inside block 0 (the
        reference is < block_size tokens) and the sequence's own tokens follow it in the same block, so the block
        cannot be shared (each sequence writes its own tokens there): it is copied. Safe this early: the
        destination is the sequence's own fresh block (stream order puts the copy after the GPU work of any earlier
        owner), the source stays pinned until _remember_prefixes evicts it after a step, and the copy is
        stream-ordered before the step's kernels."""
        hits = [s for s in seqs if s.custom_payload.prefix_block is not None]
        if not hits:
            return
        self.model_runner.call("copy_kv_blocks", [(s.custom_payload.prefix_block, s.block_table[0]) for s in hits],
                               [s.num_cached_tokens for s in hits])
        for s in hits:
            s.custom_payload.prefix_block = None  # copied: the runner does not copy it again

    def after_prefill(self, seqs) -> None:
        if self.ref_prefix_cache_enabled:
            self._remember_prefixes(seqs)

    def _remember_prefixes(self, seqs: list[Sequence[VoxCPM2SeqPayload]]) -> None:
        """After a prefill step, pin a copy of the first KV block of each voice's first request."""
        bm = self.scheduler.block_manager
        pairs, counts = [], []
        for seq in seqs:
            key = seq.custom_payload.ref_key
            if key is None or key in self.ref_prefix_blocks or not seq.block_table:
                continue
            if len(self.ref_prefix_blocks) >= self.ref_prefix_cache_size:
                _, (old, *_) = self.ref_prefix_blocks.popitem(last=False)
                bm.blocks[old].ref_count = 0
                bm._deallocate_block(old)
            if not bm.free_block_ids:
                continue
            block_id = bm.free_block_ids[0]
            bm._allocate_block(block_id).ref_count = 1  # pinned until evicted
            n = min(seq.num_prompt_tokens, self.block_size)
            # (pinned block, the tokens it holds KV for, the PreparedAudio its reference tokens came from)
            self.ref_prefix_blocks[key] = (block_id, seq.token_ids[:n], seq.custom_payload.ref_audio)
            pairs.append((seq.block_table[0], block_id))
            counts.append(n)
        if pairs:
            self.model_runner.call("copy_kv_blocks", pairs, counts)

    def preprocess_seq(self, seq: Sequence[VoxCPM2SeqPayload], is_prefill: bool) -> RunnerTask[VoxCPM2Payload]:
        if is_prefill:
            p, k = seq.custom_payload, seq.num_cached_tokens
            prefix_block = p.prefix_block  # None also when on_prefill_admitted copied it already

            return RunnerTask(
                seq.block_table,
                len(seq),
                k,
                seq.block_size,
                VoxCPM2Payload(  # the tails as lists: the runner converts a whole prefill step's at once
                    text_tokens=p.text_tokens[k:],
                    feats=_rows_from(p.feats, k),
                    feat_masks=p.feat_masks[k:],
                    temperature=p.temperature,
                    cfg_value=p.cfg_value,
                    padding_decode=p.decode_pad,
                    seed=p.seed,
                    seed_step=p.seed_step,
                    feat_keys=p.feat_keys[k:] if p.feat_keys is not None else None,
                    prefix_src_block=prefix_block,
                    prefix_len=k if prefix_block is not None else 0,
                ),
                adapter_id=seq.adapter_id,
                seq_id=seq.seq_id,
            )

        return RunnerTask(
            seq.block_table,
            len(seq),
            len(seq) - 1,
            seq.block_size,
            VoxCPM2Payload(
                text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
                feats=seq.custom_payload.feats[-1][-1:],
                feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool_),
                temperature=seq.custom_payload.temperature,
                cfg_value=seq.custom_payload.cfg_value,
                padding_decode=seq.custom_payload.decode_pad,
                seed=seq.custom_payload.seed,
                seed_step=seq.custom_payload.seed_step,
            ),
            adapter_id=seq.adapter_id,
            seq_id=seq.seq_id,
        )

    def postprocess_seq(self, seq: Sequence[VoxCPM2SeqPayload], outputs: dict, is_prefill: bool):
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        waveforms = outputs["waveforms"]

        seq.append_token(latents.tobytes())
        seq.custom_payload.feats.append(latents[None])
        seq.custom_payload.text_tokens.append(0)
        seq.custom_payload.feat_masks.append(True)
        if seq.custom_payload.feat_keys is not None:
            seq.custom_payload.feat_keys.append(0)  # generated patches are not LocEnc-cached
        if seq.custom_payload.seed is not None and seq.custom_payload.seed >= 0:
            seq.custom_payload.seed_step += 1
        seq.custom_payload.generated_waveforms.append(waveforms)
        if seq.custom_payload.chain is not None:  # a head-start head: its patches' silence decides the prompt
            silent = outputs.get("patch_silent")
            if seq.custom_payload.patch_silent is None:
                seq.custom_payload.patch_silent = []
            seq.custom_payload.patch_silent.append(_is_silent(waveforms) if silent is None else bool(silent))

        latents = latents.reshape(-1, self.feat_dim)
        if seq.custom_payload.decode_pad is not None:
            seq.custom_payload.decode_pad = np.concatenate([seq.custom_payload.decode_pad, latents], axis=0)[
                -self.n_decode_pad_frames :
            ]
        else:
            seq.custom_payload.decode_pad = latents[-self.n_decode_pad_frames :]

        if stop_flag == 1:
            seq.stoped = True
        elif (
            seq.custom_payload.max_generate_length is not None
            and len(seq.custom_payload.generated_waveforms) >= seq.custom_payload.max_generate_length
        ):
            seq.stoped = True

    def prepare_audio(self, latents: np.ndarray) -> PreparedAudio:
        latents = np.ascontiguousarray(latents, dtype=np.float32).reshape(-1, self.feat_dim)
        patches = latents.reshape(-1, self.patch_size, self.feat_dim)
        patch_bytes = [patches[i].tobytes() for i in range(patches.shape[0])]
        return PreparedAudio(
            patches=patches,
            patch_bytes=patch_bytes,
            patch_keys=[xxhash.xxh64_intdigest(b) for b in patch_bytes],
            key=xxhash.xxh64_intdigest(patches.tobytes()),
            decode_pad=latents[-self.n_decode_pad_frames :],
        )

    def register_voice(self, voice_id: str, latents: np.ndarray, text: str | None = None) -> None:
        """Keep a voice's latents, prepared once, so requests can name it (``voice=``) instead of sending them."""
        self.voices[voice_id] = RegisteredVoice(self.prepare_audio(latents), text or None)

    def unregister_voice(self, voice_id: str) -> None:
        self.voices.pop(voice_id, None)

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_text: str = "",
        prompt_latents: np.ndarray | None = None,
        ref_audio_latents: np.ndarray | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        lora_name: str | None = None,
        seed: int | None = None,
        voice: str | None = None,
        clone_mode: str | None = None,
        head_start: bool | None = None,
        text_tokens: list[int] | None = None,
        resume: bool = False,
    ) -> Sequence[VoxCPM2SeqPayload]:
        """voice / clone_mode: a registered voice in "prompt_ref" (reference + continuation, the default with a
        transcript), "ref" (the default without one) or "prompt" mode, in place of latents and prompt_text.
        head_start: None = TurboConfig.head_tokens decides, False = never for this request.
        text_tokens: the ids of prompt_text + target_text when the caller has them (head start, pretokenize()).
        resume: a head start's continuation (internal): it keeps the head's stream and is queued first.
        Returns the queued sequence (the head, with head start)."""
        if max_generate_length < 1:
            raise ValueError(f"max_generate_length must be >= 1, got {max_generate_length}")
        if head_start is not False and self.head_tokens > 0 and max_generate_length > self.head_patches:
            split = self._split_head(prompt_text, target_text, voice, clone_mode)
            if split is not None:
                head_ids, full_ids = split
                entry = self.voices.get(voice) if voice is not None else None
                if voice is None or entry is not None:  # (an unknown voice: the head's add_request says so)
                    # the whole request must fit now: its continuation is built only after the head's patches
                    try:
                        self._check_length(
                            self._prompt_len(full_ids, prompt_latents, ref_audio_latents, entry, clone_mode),
                            max_generate_length,
                        )
                    except ValueError:
                        self._release_text(prompt_text + target_text)
                        raise
                # voice_entry: the voice as the head saw it, also if it is unregistered before the continuation
                chain = dict(target_text=target_text, prompt_text=prompt_text, prompt_latents=prompt_latents,
                             ref_audio_latents=ref_audio_latents, max_generate_length=max_generate_length,
                             temperature=temperature, cfg_value=cfg_value, lora_name=lora_name, seed=seed,
                             voice=voice, voice_entry=entry, clone_mode=clone_mode, text_ids=full_ids)
                head = self.add_request(seq_id, target_text, prompt_text, prompt_latents, ref_audio_latents,
                                        self.head_patches, temperature, cfg_value, lora_name, seed, voice, clone_mode,
                                        head_start=False, text_tokens=head_ids)
                head.custom_payload.chain = chain
                self._release_text(prompt_text + target_text)
                return head
        ref = self.prepare_audio(ref_audio_latents) if ref_audio_latents is not None else None
        prompt = self.prepare_audio(prompt_latents) if prompt_latents is not None else None
        if voice is not None:
            if ref is not None or prompt is not None or prompt_text:
                raise ValueError("voice cannot be combined with prompt/reference latents or prompt_text")
            entry = self.voices.get(voice)
            if entry is None:
                raise ValueError(f"voice '{voice}' is not registered")
            mode = clone_mode or ("prompt_ref" if entry.text else "ref")
            if mode not in ("prompt_ref", "ref", "prompt"):
                raise ValueError(f"unknown clone_mode '{mode}'")
            if mode != "ref":
                if not entry.text:
                    raise ValueError(f"clone_mode '{mode}' needs a voice registered with a transcript")
                prompt, prompt_text = entry.audio, entry.text
            if mode != "prompt":
                ref = entry.audio
        ref_key = None

        if text_tokens is None:
            text_tokens = self._text_ids(prompt_text, target_text)
        remap = self._tag_remap.get(lora_name) if lora_name is not None and self._tag_remap else None
        if remap:  # a tagged adapter's request: its tags read their trained rows
            text_tokens = [remap.get(t, t) for t in text_tokens]
        text_tokens = list(text_tokens) + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False] * len(text_tokens)
        hash_tokens = list(text_tokens)
        feat_keys = [0] * len(text_tokens)
        decode_pad = None

        if ref is not None:
            if ref.ref_tokens is None:  # once per clip (a registered voice keeps them)
                n = ref.patches.shape[0]
                pad = np.zeros((1, self.patch_size, self.feat_dim), dtype=np.float32)
                ref.ref_feats = np.concatenate([pad, ref.patches, pad], axis=0)
                ref.ref_tokens = [self.ref_audio_start_token] + [0] * n + [self.ref_audio_end_token]
                ref.ref_masks = [False] + [True] * n + [False]
                ref.ref_hash = [self.ref_audio_start_token] + ref.patch_bytes + [self.ref_audio_end_token]
                ref.ref_keys = [0] + ref.patch_keys + [0]
            audio_feat = [ref.ref_feats, audio_feat]  # the voice's rows are shared, not copied per request
            text_tokens = ref.ref_tokens + text_tokens
            feat_masks = ref.ref_masks + feat_masks
            if self.ref_prefix_cache_enabled:
                ref_key = ref.key
            hash_tokens = ref.ref_hash + hash_tokens
            feat_keys = ref.ref_keys + feat_keys

        if prompt is not None:
            n = prompt.patches.shape[0]
            decode_pad = prompt.decode_pad
            audio_feat = (audio_feat if isinstance(audio_feat, list) else [audio_feat]) + [prompt.patches]
            text_tokens.extend([0] * n)
            feat_masks.extend([True] * n)
            hash_tokens.extend(prompt.patch_bytes)
            feat_keys.extend(prompt.patch_keys)
            if ref is None and self.ref_prefix_cache_enabled:
                # continuation only: requests of this voice share the prompt-text tokens at the start
                ref_key = xxhash.xxh64_intdigest(prompt.key.to_bytes(8, "little") + b"prompt")

        self._check_length(len(hash_tokens), max_generate_length)

        adapter_id = self.resolve_lora(lora_name)
        if ref_key is not None and adapter_id is not None:  # LoRA changes the KV
            ref_key = xxhash.xxh64_intdigest(ref_key.to_bytes(8, "little") + str(adapter_id).encode())

        seq = Sequence(
            seq_id,
            hash_tokens,
            self.block_size,
            VoxCPM2SeqPayload(
                feats=audio_feat if isinstance(audio_feat, list) else [audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
                seed=seed,
                seed_step=0,
                ref_key=ref_key,
                feat_keys=feat_keys if self.locenc_cache_enabled and (ref is not None or prompt is not None) else None,
                ref_audio=ref,
            ),
            lora_name=lora_name,
            adapter_id=adapter_id,
        )
        seq.has_stream = resume  # a continuation carries on the head's stream (VAE state): ahead of new requests
        self.add_sequence(seq, front=resume)
        return seq

    def _check_length(self, prompt_len: int, max_generate_length: int) -> None:
        total_len_upper_bound = prompt_len + max_generate_length
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"Prompt is too long for max_model_len: prompt_len={prompt_len} > max_model_len={self.max_model_len}"
            )
        if total_len_upper_bound > self.max_model_len:
            raise ValueError(
                "Request may exceed max_model_len: "
                f"prompt_len({prompt_len}) + max_generate_length({max_generate_length}) = {total_len_upper_bound} "
                f"> max_model_len({self.max_model_len}). "
                "Reduce input length or max_generate_length, or increase max_model_len."
            )

    def _prompt_len(self, text_ids, prompt_latents, ref_audio_latents, entry: RegisteredVoice | None,
                    clone_mode: str | None) -> int:
        """add_request's prompt length (hash tokens) without building the request: text ids + audio start,
        [ref start] reference patches [ref end], prompt patches. A head start's continuation is at most this plus
        the head's patches, which its max_generate_length gives up: the whole request's check covers it."""
        per_patch = self.patch_size * self.feat_dim
        n_ref = np.asarray(ref_audio_latents).size // per_patch if ref_audio_latents is not None else None
        n_prompt = np.asarray(prompt_latents).size // per_patch if prompt_latents is not None else 0
        if entry is not None:
            mode = clone_mode or ("prompt_ref" if entry.text else "ref")
            n = entry.audio.patches.shape[0]
            n_prompt = n if mode != "ref" else 0
            n_ref = n if mode != "prompt" else None
        return len(text_ids) + 1 + (n_ref + 2 if n_ref is not None else 0) + n_prompt

    @property
    def _pretok_map(self) -> dict[str, tuple[list[int], list[tuple[int, int]]]]:
        """pretokenize()'s cache: text -> (ids, character offsets)."""
        if self._pretok_cache is None:
            self._pretok_cache = {}
        return self._pretok_cache

    @property
    def _pretok_uses(self) -> dict[str, int]:
        """pretokenize()'s count of the registrations still to come per text: an entry of the cache stays until the
        last of them (a text twice in one batch was tokenized again, one by one, for its second request)."""
        if self._pretok_pending is None:
            self._pretok_pending = {}
        return self._pretok_pending

    _pretok_pending: dict | None = None

    def pretokenize(self, texts: list[str]) -> None:
        """Tokenize a burst's texts in one batched call (the fast tokenizer, with character offsets for head
        start); add_request then takes them from this cache. Texts with CJK characters keep the per-text path
        (the multi-character Chinese token split)."""
        cache, uses = self._pretok_map, self._pretok_uses
        for t in texts:
            uses[t] = uses.get(t, 0) + 1
        texts = [t for t in dict.fromkeys(texts) if t not in cache and not _CJK.search(t)]
        backend = getattr(getattr(self.tokenizer, "tokenizer", None), "_tokenizer", None)  # tokenizers.Tokenizer
        if texts and backend is not None:  # (else: no fast tokenizer behind the wrapper)
            for t, enc in zip(texts, backend.encode_batch(texts, add_special_tokens=False)):
                cache[t] = (enc.ids, enc.offsets)
        while len(cache) > 4096:  # requests that never came (cancelled before registration)
            t = next(iter(cache))
            cache.pop(t)
            uses.pop(t, None)
        while len(uses) > 8192:  # (counts of texts that were never cached: CJK, no fast tokenizer)
            uses.pop(next(iter(uses)))

    def _release_text(self, text: str) -> None:
        """add_request is done with ``text``: its cached ids go once no registration of it is pending."""
        uses = self._pretok_uses
        n = uses.pop(text, 0) - 1
        if n > 0:
            uses[text] = n
        else:
            self._pretok_map.pop(text, None)

    def _tokens(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        """ids (and character offsets) of ``text``, from pretokenize()'s cache or tokenized now (and cached until
        add_request is done with the text)."""
        hit = self._pretok_map.get(text)
        if hit is not None:
            return hit
        if _CJK.search(text) or not hasattr(self.tokenizer, "tokenizer"):
            return self.tokenizer(text), None
        enc = self.tokenizer.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        self._pretok_map[text] = (enc["input_ids"], enc["offset_mapping"])
        return self._pretok_map[text]

    def _text_ids(self, prompt_text: str, target_text: str) -> list[int]:
        ids = self._tokens(prompt_text + target_text)[0]
        self._release_text(prompt_text + target_text)
        return ids

    def _split_head(self, prompt_text, target_text, voice, clone_mode) -> tuple[list[int], list[int]] | None:
        """Head start: (ids of prompt_text + head, ids of prompt_text + target_text), the head being the longest
        word-boundary prefix of target_text with at most head_tokens tokens (cut at the last clause boundary
        inside it when that keeps half of it), or the first word when that alone is longer (up to twice
        head_tokens: a whole-text fallback put 22 of 200 requests' full texts into the head step at 6 tokens);
        None when the target is short enough to prefill whole."""
        if voice is not None:
            entry = self.voices.get(voice)
            if entry is not None and (clone_mode or ("prompt_ref" if entry.text else "ref")) != "ref":
                prompt_text = entry.text or ""
        text = prompt_text + target_text
        ids, offs = self._tokens(text)
        if offs is None:
            return None
        start = len(prompt_text)
        tag = _LEADING_TAG.match(text, start)
        if tag is not None:  # the head counts text, not the dialect tag in front of it
            start = tag.end()
        first = next((k for k, o in enumerate(offs) if o[1] > start), len(ids))  # first token of the target
        if len(ids) - first <= self.head_tokens + self.head_min_rest:
            return None
        best = clause = None
        for k in range(first + 1, min(first + self.head_tokens, len(ids) - 1) + 1):
            o = offs[k][0]
            if o > start and o < len(text) and text[o] == " " and offs[k - 1][1] <= o:  # a word starts at token k
                best = k
                if text[o - 1] in ",،;؛:":
                    clause = k
        if best is None:  # the first word alone is longer than head_tokens: it is the head (up to twice that)
            best = next((k for k in range(first + 1, min(first + 2 * self.head_tokens, len(ids) - 1) + 1)
                         if text[offs[k][0]:offs[k][0] + 1] == " " and offs[k - 1][1] <= offs[k][0]), None)
            if best is None:
                return None
        cut = clause if clause is not None and (clause - first) * 2 >= best - first else best
        return ids[:cut], ids

    def _continue_head(self, seq: Sequence[VoxCPM2SeqPayload]) -> bool:
        """Queue the whole request after a finished head: its patches become prompt audio (after the voice's
        own prompt audio, if any); seed, noise step, LoRA and stream id carry over. False: nothing was queued."""
        chain, payload = seq.custom_payload.chain, seq.custom_payload
        n = len(payload.generated_waveforms)
        # cap: a continuation never speaks slower than ~4 tokens (characters) a second; a model that keeps
        # generating past that (silence, babble) is cut off
        rest = min(chain["max_generate_length"] - n, int(1.5 * len(chain["text_ids"])) + 20)
        if rest < 1:
            return False
        # Leading silent patches (a sentence often starts with a short pause) are not given as prompt audio: a
        # prompt of nothing but silence was continued as silence, never reaching the stop (measured: 400 silent
        # patches). They were streamed already; the continuation starts after them.
        silent = payload.patch_silent or [_is_silent(w) for w in payload.generated_waveforms[-n:]]
        k = 0
        while k < n and silent[-n + k]:
            k += 1
        feats = payload.feats[-n:][k:]
        head = (np.concatenate([f.reshape(-1, self.feat_dim) for f in feats], axis=0).astype(np.float32)
                if feats else None)
        kw = dict(chain)
        prompt, prompt_text, ref = kw.pop("prompt_latents"), kw.pop("prompt_text"), kw.pop("ref_audio_latents")
        voice, mode, entry = kw.pop("voice"), kw.pop("clone_mode"), kw.pop("voice_entry")
        if voice is not None:  # resolve the registered voice into latents, as add_request does
            mode = mode or ("prompt_ref" if entry.text else "ref")
            frames = entry.audio.patches.reshape(-1, self.feat_dim)
            if mode != "ref":
                prompt, prompt_text = frames, entry.text
            if mode != "prompt":
                ref = frames
        if head is not None:
            kw["prompt_latents"] = head if prompt is None else np.concatenate([prompt, head], axis=0)
        else:
            kw["prompt_latents"] = prompt
        kw["prompt_text"], kw["ref_audio_latents"], kw["max_generate_length"] = prompt_text, ref, rest
        kw["text_tokens"] = kw.pop("text_ids")
        new = self.add_request(seq.seq_id, head_start=False, resume=True, **kw)
        new.custom_payload.seed_step = payload.seed_step
        return True

    def step(self):
        with self.stepping():  # a cancel during the step runs after it: it also reaches a continuation queued here
            seqs = super().step()
            for seq in reversed(seqs):  # continuations go to the front of the queue, in this step's order
                if seq.stoped and seq.custom_payload.chain is not None:
                    try:
                        seq.custom_payload.chained = self._continue_head(seq)
                    except ValueError:  # e.g. its LoRA was unregistered meanwhile: the stream ends with the head
                        seq.custom_payload.chained = False
                    if not seq.custom_payload.chained:
                        self.model_runner.call("release_sequence_state", seq.seq_id)
            return seqs

    def on_seq_removed(self, seq: Sequence, *, was_running: bool) -> None:
        if seq.stoped and getattr(seq.custom_payload, "chain", None) is not None:
            # a finished head: its continuation keeps the stream's VAE state (seamless audio); step() releases it
            # if no continuation gets queued
            self.lora_manager.on_sequence_finished(seq.adapter_id, was_running=was_running)
            self.model_runner.call("lora_on_sequence_finished", seq.adapter_id, was_running)
            return
        super().on_seq_removed(seq, was_running=was_running)

    def encode_latents(self, wav: torch.Tensor, align_size: int = -1) -> np.ndarray:
        if align_size == -1:
            align_size = self.patch_size * self.model_runner.vae.encoder_chunk_size
        if wav.size(1) % align_size != 0:
            remained = align_size - wav.size(1) % align_size
            wav = torch.nn.functional.pad(wav, (remained, 0))
        return self.model_runner.encode_latents(wav)
