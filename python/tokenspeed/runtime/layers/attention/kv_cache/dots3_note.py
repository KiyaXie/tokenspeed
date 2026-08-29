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

"""Mixed full-DSA/sliding-MLA cache view for Dots3 Note."""

from __future__ import annotations

from typing import ClassVar

import torch
from tokenspeed_kernel.ops.kvcache.triton import get_mla_kv_buffer_triton
from typing_extensions import override

from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool
from tokenspeed.runtime.layers.paged_attention import PagedAttention


class Dots3NoteTokenToKVPool(DSATokenToKVPool):
    """Bind Dots3 layers whose compressed-KV widths differ by attention type."""

    layer_plane_bindings: ClassVar[dict[str, str]] = {
        **DSATokenToKVPool.layer_plane_bindings,
        "index_k": "_index_k",
    }

    _index_k: list[torch.Tensor | None]
    index_k_buffer: dict[int, torch.Tensor]
    swa_kv_lora_rank: int
    swa_qk_rope_head_dim: int

    def __init__(
        self,
        arena: CacheArena,
        model_dtype: torch.dtype,
        dtype: torch.dtype,
        quant_method: str,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        swa_kv_lora_rank: int,
        swa_qk_rope_head_dim: int,
        layer_num: int,
        rank: int,
        index_head_dim: int,
        *,
        field_layer_offset: int = 0,
    ) -> None:
        if quant_method == "per_token_head":
            raise ValueError("Dots3 Note's initial cache pool supports BF16 KV only.")
        self.swa_kv_lora_rank = int(swa_kv_lora_rank)
        self.swa_qk_rope_head_dim = int(swa_qk_rope_head_dim)
        super().__init__(
            arena,
            model_dtype=model_dtype,
            dtype=dtype,
            quant_method=quant_method,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=layer_num,
            rank=rank,
            index_head_dim=index_head_dim,
            field_layer_offset=field_layer_offset,
        )
        for layer_id in range(self.layer_num):
            kv_lora_rank, rope_dim = self._latent_dims(layer_id)
            actual = int(self.get_key_buffer(layer_id).shape[-1])
            expected = kv_lora_rank + rope_dim
            if actual != expected:
                raise ValueError(
                    f"Dots3 Note layer {layer_id} cache width is {actual}, expected {expected}."
                )

    @override
    def _bind_layer_planes(self) -> None:
        super()._bind_layer_planes()
        self.index_k_buffer = {
            layer_id: buffer
            for layer_id, buffer in enumerate(self._index_k)
            if buffer is not None
        }

    def _latent_dims(self, layer_id: int) -> tuple[int, int]:
        _ = self._field_layer_id(layer_id)
        if layer_id in self.index_k_buffer:
            return int(self.kv_lora_rank), int(self.qk_rope_head_dim)
        return self.swa_kv_lora_rank, self.swa_qk_rope_head_dim

    @override
    def get_kv_size_bytes(self) -> int:
        return int(self.arena.buffer.numel() * self.arena.buffer.element_size())

    @override
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        buffer = super().get_key_buffer(layer_id)
        if not isinstance(buffer, torch.Tensor):
            raise TypeError("Dots3 Note requires an unquantized latent-KV tensor.")
        return buffer

    @override
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        kv_lora_rank, _ = self._latent_dims(layer_id)
        return self.get_key_buffer(layer_id)[..., :kv_lora_rank]

    @override
    def get_index_k_buffer(self, layer_id: int) -> torch.Tensor:
        _ = self._field_layer_id(layer_id)
        if layer_id not in self.index_k_buffer:
            raise ValueError(
                f"Dots3 Note sliding-attention layer {layer_id} has no index-K cache."
            )
        return super().get_index_k_buffer(layer_id)

    @override
    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        _ = self.get_index_k_buffer(layer_id)
        super().set_index_k_buffer(layer_id, loc, index_k)

    @override
    def get_mla_kv_buffer(
        self,
        layer: PagedAttention,
        loc: torch.Tensor,
        dst_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_id = int(layer.layer_id)
        kv_lora_rank, rope_dim = self._latent_dims(layer_id)
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, rope_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope


__all__ = ["Dots3NoteTokenToKVPool"]
