# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Full DSA plus portable sliding MLA for Dots3 Note."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel import mla_decode_with_kvcache

from tokenspeed.runtime.configs.dots3_note_config import (
    DOTS3_NOTE_FULL_DSA_GROUP,
    DOTS3_NOTE_SWA_GROUP_PREFIX,
)
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
    Dots3NoteTokenToKVPool,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass(kw_only=True)
class Dots3NoteSlidingMetadata:
    """Per-forward group tables and absolute request geometry for sliding MLA."""

    forward_mode: ForwardMode
    seq_lens: torch.Tensor
    query_lens: tuple[int, ...]
    page_tables: dict[str, torch.Tensor]
    out_cache_locs: dict[str, torch.Tensor]


def _portable_sliding_mla(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    num_heads: int,
    kv_lora_rank: int,
    page_size: int,
    window_size: int,
    scaling: float,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """Compute one request's absorbed sliding MLA from an absolute page table."""
    if query_len <= 0 or seq_len < query_len:
        raise ValueError(
            f"invalid Dots3 sliding query geometry: seq_len={seq_len}, query_len={query_len}"
        )
    if q.shape[0] != query_len:
        raise ValueError(f"query rows {q.shape[0]} do not match query_len={query_len}")
    latent_width = q.shape[-1] if q.ndim == 3 else q.shape[-1] // num_heads
    q = q.view(query_len, num_heads, latent_width)
    query_start = seq_len - query_len
    outputs: list[torch.Tensor] = []

    for chunk_start in range(0, query_len, query_chunk_size):
        chunk_end = min(query_len, chunk_start + query_chunk_size)
        query_positions = torch.arange(
            query_start + chunk_start,
            query_start + chunk_end,
            dtype=torch.int64,
            device=q.device,
        )
        kv_start = max(0, int(query_positions[0].item()) - window_size + 1)
        kv_end = int(query_positions[-1].item()) + 1
        kv_positions = torch.arange(
            kv_start, kv_end, dtype=torch.int64, device=q.device
        )
        columns = kv_positions // page_size
        if columns.numel() == 0 or int(columns[-1].item()) >= page_table.numel():
            raise RuntimeError(
                "Dots3 sliding page table does not cover the requested absolute positions."
            )
        page_ids = page_table.index_select(0, columns)
        valid_pages = page_ids > 0
        slots = (
            page_ids.clamp_min(0).to(torch.int64) * page_size + kv_positions % page_size
        )
        cached = kv_cache.index_select(0, slots).squeeze(1)

        visible = (
            valid_pages.unsqueeze(0)
            & (kv_positions.unsqueeze(0) <= query_positions.unsqueeze(1))
            & (
                kv_positions.unsqueeze(0)
                >= query_positions.unsqueeze(1) - window_size + 1
            )
        )
        if not bool(visible.any(dim=1).all().item()):
            raise RuntimeError(
                "Dots3 sliding attention found a query with no live KV rows."
            )

        scores = torch.einsum("qhd,kd->qhk", q[chunk_start:chunk_end], cached)
        scores = scores.float().mul_(scaling)
        scores.masked_fill_(~visible[:, None, :], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(cached.dtype)
        outputs.append(
            torch.einsum(
                "qhk,kr->qhr",
                probabilities,
                cached[:, :kv_lora_rank],
            )
        )

    return torch.cat(outputs, dim=0).reshape(query_len, num_heads * kv_lora_rank)


class Dots3NoteAttentionBackend(AttentionBackend):
    """Delegate full layers to DSA and run paged sliding MLA."""

    uses_cache_groups = True
    cache_consumer_families = frozenset({"history"})

    def __init__(self, config: Dots3NoteAttentionConfig) -> None:
        super().__init__(config)
        if config.kv_cache_dtype != torch.bfloat16:
            raise ValueError("Dots3 Note sliding MLA requires BF16 KV cache.")
        if config.is_draft or config.speculative_num_steps:
            raise NotImplementedError(
                "Dots3 Note does not yet support speculative decoding."
            )
        self._full_dsa = DSABackend(config)
        self.kernel_page_size = int(config.prefix_granularity)
        self.max_num_pages = (
            config.context_len + self.kernel_page_size - 1
        ) // self.kernel_page_size
        self.max_context_len = int(config.context_len)
        self.sliding_window_tokens = int(config.sliding_window_tokens)
        self.swa_kv_lora_rank = int(config.swa_kv_lora_rank)
        self.swa_qk_nope_head_dim = int(config.swa_qk_nope_head_dim)
        self.swa_qk_rope_head_dim = int(config.swa_qk_rope_head_dim)
        self.sliding_group_ids = tuple(
            f"{DOTS3_NOTE_SWA_GROUP_PREFIX}{index}" for index in range(3)
        )
        self.sliding_metadata: Dots3NoteSlidingMetadata | None = None
        self.cuda_graph_sliding_metadata: dict[int, Dots3NoteSlidingMetadata] = {}

    @property
    def forward_decode_metadata(self):
        return self._full_dsa.forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._full_dsa.forward_prefill_metadata

    @property
    def chunked_prefill_metadata(self):
        return self._full_dsa.chunked_prefill_metadata

    @property
    def decode_cuda_graph_metadata(self):
        return self._full_dsa.decode_cuda_graph_metadata

    @property
    def trtllm_workspace(self):
        return self._full_dsa.trtllm_workspace

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        if not isinstance(cache_pool, Dots3NoteTokenToKVPool):
            raise TypeError("Dots3 Note backend requires Dots3NoteTokenToKVPool.")
        super().set_cache_pool(cache_pool)
        self._full_dsa.set_cache_pool(cache_pool)

    def mark_cache_contract(self) -> None:
        self._full_dsa.mark_cache_contract()

    def register_step_counter(self, step_counter) -> None:
        super().register_step_counter(step_counter)
        self._full_dsa.register_step_counter(step_counter)

    def override_num_extends(self, num_extends: int):
        return self._full_dsa.override_num_extends(num_extends)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor | None = None):
        return self._full_dsa.advance_draft_forward_metadata(seq_lens)

    def configure_runtime(self, **kwargs) -> None:
        self._full_dsa.configure_runtime(**kwargs)

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        self._full_dsa.init_cuda_graph_state(max_bs)
        self.cuda_graph_seq_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_page_tables = {
            group_id: torch.zeros(
                (max_bs, self.max_num_pages), dtype=torch.int32, device=self.device
            )
            for group_id in self.sliding_group_ids
        }
        self.cuda_graph_out_cache_locs = {
            group_id: torch.zeros(max_bs, dtype=torch.int64, device=self.device)
            for group_id in self.sliding_group_ids
        }
        self.cuda_graph_sliding_metadata = {}

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        if not forward_mode.is_decode_or_idle():
            raise RuntimeError(
                f"Dots3 Note CUDA Graph requires decode, got {forward_mode}"
            )
        self._full_dsa.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
        required = {DOTS3_NOTE_FULL_DSA_GROUP, *self.sliding_group_ids}
        if set(cache_group_ids) != required:
            raise RuntimeError(
                f"Dots3 Note CUDA Graph requires cache groups {sorted(required)}, "
                f"got {sorted(cache_group_ids)}."
            )
        self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        metadata = Dots3NoteSlidingMetadata(
            forward_mode=forward_mode,
            seq_lens=self.cuda_graph_seq_lens[:bs],
            query_lens=(1,) * bs,
            page_tables={
                group_id: table[:bs]
                for group_id, table in self.cuda_graph_page_tables.items()
            },
            out_cache_locs={
                group_id: locs[:bs]
                for group_id, locs in self.cuda_graph_out_cache_locs.items()
            },
        )
        self.cuda_graph_sliding_metadata[bs] = metadata
        self.sliding_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode | None = None,
        page_table: torch.Tensor | None = None,
        block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        if forward_mode is None or not forward_mode.is_decode_or_idle():
            raise RuntimeError(
                f"Dots3 Note CUDA Graph requires decode, got {forward_mode}"
            )
        if page_table is None:
            raise RuntimeError("Dots3 Note CUDA Graph requires a full DSA page table.")
        self._full_dsa.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            page_table=page_table,
            **kwargs,
        )
        required = {DOTS3_NOTE_FULL_DSA_GROUP, *self.sliding_group_ids}
        if not block_tables or set(block_tables) != required:
            delivered = set() if not block_tables else set(block_tables)
            raise RuntimeError(
                f"Dots3 Note CUDA Graph requires cache groups {sorted(required)}, "
                f"got {sorted(delivered)}."
            )
        metadata = self.cuda_graph_sliding_metadata.get(bs)
        if metadata is None:
            raise RuntimeError(
                f"Dots3 Note CUDA Graph batch size {bs} was not captured."
            )

        self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs])
        positions = (self.cuda_graph_seq_lens[:bs].to(torch.int64) - 1).clamp_min(0)
        columns = positions // self.kernel_page_size
        offsets = positions % self.kernel_page_size
        for group_id in self.sliding_group_ids:
            source = block_tables[group_id]
            target = self.cuda_graph_page_tables[group_id][:bs]
            rows = min(source.shape[0], bs)
            cols = min(source.shape[1], target.shape[1])
            target[:rows, :cols].copy_(source[:rows, :cols])
            if cols < target.shape[1]:
                target[:rows, cols:].zero_()
            if rows < bs:
                target[rows:].zero_()
            self.cuda_graph_out_cache_locs[group_id][:bs].copy_(
                target.gather(1, columns[:, None]).squeeze(1).to(torch.int64)
                * self.kernel_page_size
                + offsets
            )
        self.sliding_metadata = metadata

    @staticmethod
    def _as_int_tuple(values: Any, *, name: str, expected: int) -> tuple[int, ...]:
        if values is None:
            raise RuntimeError(f"Dots3 Note metadata is missing {name}.")
        if isinstance(values, torch.Tensor):
            result = tuple(int(value) for value in values.tolist())
        else:
            result = tuple(int(value) for value in values)
        if len(result) != expected:
            raise RuntimeError(
                f"Dots3 Note {name} has {len(result)} rows, expected {expected}."
            )
        return result

    @staticmethod
    def _decode_out_cache_locs(
        page_tables: dict[str, torch.Tensor],
        seq_lens: torch.Tensor,
        page_size: int,
    ) -> dict[str, torch.Tensor]:
        positions = (seq_lens.to(torch.int64) - 1).clamp_min(0)
        columns = positions // page_size
        offsets = positions % page_size
        return {
            group_id: table.gather(1, columns[:, None]).squeeze(1).clamp_min(0)
            * page_size
            + offsets
            for group_id, table in page_tables.items()
        }

    @staticmethod
    def _extend_out_cache_locs(
        page_tables: dict[str, torch.Tensor],
        prefix_lens: tuple[int, ...],
        query_lens: tuple[int, ...],
        page_size: int,
    ) -> dict[str, torch.Tensor]:
        chunks = {group_id: [] for group_id in page_tables}
        device = next(iter(page_tables.values())).device
        for request_id, (start, query_len) in enumerate(
            zip(prefix_lens, query_lens, strict=True)
        ):
            positions = torch.arange(
                start,
                start + query_len,
                dtype=torch.int64,
                device=device,
            )
            for group_id, table in page_tables.items():
                columns = positions // page_size
                if columns.numel() and int(columns[-1].item()) >= table.shape[1]:
                    raise RuntimeError(
                        f"Dots3 Note group {group_id} page table is too short for prefill."
                    )
                pages = table[request_id].index_select(0, columns)
                if pages.numel() and not bool((pages > 0).all().item()):
                    raise RuntimeError(
                        f"Dots3 Note group {group_id} has no writable page for prefill."
                    )
                chunks[group_id].append(
                    pages.to(torch.int64) * page_size + positions % page_size
                )
        return {
            group_id: torch.cat(group_chunks)
            for group_id, group_chunks in chunks.items()
        }

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        if forward_mode.is_mixed():
            raise NotImplementedError("Dots3 Note does not yet support mixed batches.")
        self._full_dsa.init_forward_metadata(
            bs=bs,
            num_extends=num_extends,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            page_table=page_table,
            forward_mode=forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            **kwargs,
        )
        if forward_mode.is_idle() or bs == 0:
            self.sliding_metadata = None
            return

        block_tables = kwargs.get("block_tables")
        cache_metadata = kwargs.get("cache_metadata")
        forward_batch = kwargs.get("forward_batch")
        if block_tables is None and cache_metadata is not None:
            block_tables = dict(cache_metadata.tables(active_forward_op=forward_batch))
        if not block_tables:
            raise RuntimeError("Dots3 Note requires per-group cache block tables.")
        page_tables = dict(block_tables)
        required = {
            DOTS3_NOTE_FULL_DSA_GROUP,
            *(f"{DOTS3_NOTE_SWA_GROUP_PREFIX}{index}" for index in range(3)),
        }
        if set(page_tables) != required:
            raise RuntimeError(
                f"Dots3 Note requires cache groups {sorted(required)}, got {sorted(page_tables)}."
            )

        seq_lens_device = seq_lens[:bs]
        if forward_mode.is_decode():
            query_lens = (1,) * bs
            out_cache_locs = self._decode_out_cache_locs(
                page_tables,
                seq_lens[:bs],
                self.kernel_page_size,
            )
        else:
            if num_extends != bs:
                raise NotImplementedError(
                    "Dots3 Note prefill currently requires an all-extend batch."
                )
            query_lens = self._as_int_tuple(
                extend_seq_lens_cpu,
                name="extend_seq_lens_cpu",
                expected=bs,
            )
            prefix_lens = self._as_int_tuple(
                extend_prefix_lens_cpu,
                name="extend_prefix_lens_cpu",
                expected=bs,
            )
            out_cache_locs = self._extend_out_cache_locs(
                page_tables,
                prefix_lens,
                query_lens,
                self.kernel_page_size,
            )

        self.sliding_metadata = Dots3NoteSlidingMetadata(
            forward_mode=forward_mode,
            seq_lens=seq_lens_device,
            query_lens=query_lens,
            page_tables=page_tables,
            out_cache_locs=out_cache_locs,
        )

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        if layer.group_id == DOTS3_NOTE_FULL_DSA_GROUP:
            return self._full_dsa.select_out_cache_loc(
                layer, out_cache_loc, forward_mode
            )
        if not layer.group_id.startswith(DOTS3_NOTE_SWA_GROUP_PREFIX):
            raise KeyError(f"unknown Dots3 Note cache group {layer.group_id!r}")
        metadata = self.sliding_metadata
        if metadata is None:
            raise RuntimeError("Dots3 Note sliding metadata is not initialized.")
        locs = metadata.out_cache_locs[layer.group_id]
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"Dots3 Note group {layer.group_id} has {locs.shape[0]} write locations "
                f"for {out_cache_loc.shape[0]} input tokens."
            )
        return locs

    def forward_sparse_prefill(self, **kwargs) -> torch.Tensor:
        return self._full_dsa.forward_sparse_prefill(**kwargs)

    def forward_extend_chunked(self, *args, **kwargs):
        return self._full_dsa.forward_extend_chunked(*args, **kwargs)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if layer.group_id == DOTS3_NOTE_FULL_DSA_GROUP:
            return self._full_dsa.forward_decode(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        return self._forward_sliding(
            q,
            k,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            save_kv_cache=save_kv_cache,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if layer.group_id == DOTS3_NOTE_FULL_DSA_GROUP:
            return self._full_dsa.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        return self._forward_sliding(
            q,
            k,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            save_kv_cache=save_kv_cache,
        )

    def _forward_sliding(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        *,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        if not isinstance(token_to_kv_pool, Dots3NoteTokenToKVPool):
            raise TypeError("Dots3 Note backend requires Dots3NoteTokenToKVPool.")
        metadata = self.sliding_metadata
        if metadata is None:
            raise RuntimeError("Dots3 Note sliding metadata is not initialized.")
        if layer.logit_cap > 0:
            raise NotImplementedError(
                "Dots3 Note sliding MLA logit soft-capping is unsupported."
            )
        if save_kv_cache:
            if k is None:
                raise ValueError("Dots3 Note sliding MLA requires compressed K rows.")
            locs = self.select_out_cache_loc(
                layer, out_cache_loc, metadata.forward_mode
            )
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                locs,
                k[..., : self.swa_kv_lora_rank],
                k[..., self.swa_kv_lora_rank :],
            )

        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if kv_cache.dtype != torch.bfloat16:
            raise RuntimeError("Dots3 Note sliding MLA requires BF16 cache rows.")
        kv_cache = kv_cache.view(
            -1,
            self.kernel_page_size,
            1,
            self.swa_kv_lora_rank + self.swa_qk_rope_head_dim,
        )
        table = metadata.page_tables[layer.group_id]
        output = q.new_empty((q.shape[0], layer.tp_q_head_num, self.swa_kv_lora_rank))

        if metadata.forward_mode.is_decode():
            result = mla_decode_with_kvcache(
                q=q.view(
                    metadata.seq_lens.shape[0],
                    1,
                    layer.tp_q_head_num,
                    self.swa_kv_lora_rank + self.swa_qk_rope_head_dim,
                ),
                kv_cache=kv_cache,
                page_table=table,
                cache_seqlens=metadata.seq_lens,
                max_seqlen_k=self.max_context_len,
                qk_nope_head_dim=self.swa_qk_nope_head_dim,
                kv_lora_rank=self.swa_kv_lora_rank,
                qk_rope_head_dim=self.swa_qk_rope_head_dim,
                softmax_scale=layer.scaling,
                out=output.unsqueeze(1),
                solution="triton",
                window_left=self.sliding_window_tokens - 1,
            )
            return result.reshape(q.shape[0], -1)

        offset = 0
        for request_id, query_len in enumerate(metadata.query_lens):
            request_output = output[offset : offset + query_len].unsqueeze(0)
            mla_decode_with_kvcache(
                q=q[offset : offset + query_len].view(
                    1,
                    query_len,
                    layer.tp_q_head_num,
                    self.swa_kv_lora_rank + self.swa_qk_rope_head_dim,
                ),
                kv_cache=kv_cache,
                page_table=table[request_id : request_id + 1],
                cache_seqlens=metadata.seq_lens[request_id : request_id + 1],
                max_seqlen_k=self.max_context_len,
                qk_nope_head_dim=self.swa_qk_nope_head_dim,
                kv_lora_rank=self.swa_kv_lora_rank,
                qk_rope_head_dim=self.swa_qk_rope_head_dim,
                softmax_scale=layer.scaling,
                out=request_output,
                solution="triton",
                window_left=self.sliding_window_tokens - 1,
            )
            offset += query_len
        if offset != q.shape[0]:
            raise RuntimeError(
                f"Dots3 Note metadata covers {offset} query rows, got {q.shape[0]}."
            )
        return output.reshape(q.shape[0], -1)


register_backend(
    "dots3_note",
    {AttentionArch.DOTS3_NOTE},
    Dots3NoteAttentionBackend,
)


__all__ = [
    "Dots3NoteAttentionBackend",
    "Dots3NoteSlidingMetadata",
]
