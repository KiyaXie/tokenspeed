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

"""Attention configuration for mixed DSA/SWA Dots3 Note layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from typing_extensions import override

from tokenspeed.runtime.configs.dots3_note_config import (
    DOTS3_NOTE_NUM_TARGET_LAYERS,
    dots3_note_group_ids,
    dots3_note_layer_types,
)
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class Dots3NoteAttentionConfig(DSAConfig):
    """Runtime geometry for both Dots3 Note attention variants."""

    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_q_lora_rank: int
    swa_kv_lora_rank: int
    swa_qk_nope_head_dim: int
    swa_qk_rope_head_dim: int
    swa_v_head_dim: int
    swa_rope_theta: float
    sliding_window_tokens: int
    group_ids: tuple[str, ...]
    indexer_rope_interleave: bool

    @classmethod
    @override
    def generate(
        cls,
        server_args: ServerArgs,
        model_config: ModelConfig,
        is_draft: bool = False,
    ) -> Dots3NoteAttentionConfig:
        """Build Dots3 Note's mixed-attention runtime configuration.

        Args:
            server_args: Runtime and cache command-line configuration.
            model_config: Loaded model configuration and derived geometry.
            is_draft: Whether this configuration describes a draft model.

        Returns:
            A configuration carrying both full DSA and sliding MLA dimensions.
        """
        if is_draft or server_args.speculative_algorithm is not None:
            raise NotImplementedError(
                "Dots3 Note does not yet support speculative or MTP decoding."
            )
        if getattr(server_args, "disaggregation_mode", "null") != "null":
            raise NotImplementedError(
                "Dots3 Note does not yet support PD/disaggregated serving."
            )
        if getattr(server_args, "enable_mixed_batch", False):
            raise NotImplementedError(
                "Dots3 Note does not yet support --enable-mixed-batch."
            )
        num_layers = model_config.num_attention_layers
        if num_layers != DOTS3_NOTE_NUM_TARGET_LAYERS:
            raise NotImplementedError(
                "Dots3 Note currently supports exactly 46 target layers; "
                f"got {num_layers}. MTP continuation is not supported."
            )
        server_args.disable_prefill_graph = True
        server_args.disable_autotune = True
        server_args.enable_prefix_caching = False
        server_args.disable_kvstore = True

        base = DSAConfig.generate(server_args, model_config, is_draft)
        if base.dtype != torch.bfloat16:
            raise ValueError(
                "Dots3 Note's initial implementation requires a BF16 model, "
                f"got {base.dtype}."
            )
        hf_config = model_config.hf_text_config
        sliding_window_tokens = int(hf_config.sliding_window_size)
        if sliding_window_tokens != 513:
            raise ValueError(
                f"Dots3 Note requires sliding_window_size=513, got {sliding_window_tokens}."
            )
        cache_geometry = (
            int(base.kv_lora_rank),
            int(base.qk_rope_head_dim),
            int(base.index_head_dim),
            int(hf_config.swa_kv_lora_rank),
            int(hf_config.swa_qk_rope_head_dim),
        )
        expected_geometry = (512, 64, 128, 1024, 64)
        if cache_geometry != expected_geometry:
            raise ValueError(
                "Dots3 Note requires cache geometry "
                "(kv_lora_rank, qk_rope_head_dim, index_head_dim, "
                "swa_kv_lora_rank, swa_qk_rope_head_dim)="
                f"{expected_geometry}, got {cache_geometry}."
            )
        if base.prefix_granularity != 64:
            raise ValueError(
                f"Dots3 Note requires prefix_granularity=64, got {base.prefix_granularity}."
            )
        if base.kv_cache_dtype != torch.bfloat16:
            raise ValueError(
                "Dots3 Note's initial FlatKV implementation requires a BF16 "
                + f"KV cache, got {base.kv_cache_dtype}."
            )

        values = base.__dict__ | {
            "layer_types": dots3_note_layer_types(num_layers),
            "swa_num_attention_heads": int(hf_config.swa_num_attention_heads),
            "swa_num_key_value_heads": int(hf_config.swa_num_key_value_heads),
            "swa_q_lora_rank": int(hf_config.swa_q_lora_rank),
            "swa_kv_lora_rank": int(hf_config.swa_kv_lora_rank),
            "swa_qk_nope_head_dim": int(hf_config.swa_qk_nope_head_dim),
            "swa_qk_rope_head_dim": int(hf_config.swa_qk_rope_head_dim),
            "swa_v_head_dim": int(hf_config.swa_v_head_dim),
            "swa_rope_theta": float(hf_config.swa_rope_theta),
            "sliding_window_tokens": sliding_window_tokens,
            "group_ids": dots3_note_group_ids(num_layers),
            "indexer_rope_interleave": bool(hf_config.indexer_rope_interleave),
        }
        return cls(**values)  # type: ignore[arg-type]


__all__ = ["Dots3NoteAttentionConfig"]
