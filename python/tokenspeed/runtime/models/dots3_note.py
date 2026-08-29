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

"""Text-only inference implementation for dots-studio/dots3-note-prev."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from tokenspeed_kernel.ops.attention import mla_project_value
from tokenspeed_kernel.ops.gemm import bmm
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.dots3_note_config import (
    DOTS3_NOTE_FULL_DSA,
    DOTS3_NOTE_FULL_DSA_GROUP,
    dots3_note_group_ids,
    dots3_note_layer_types,
)
from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear, RowParallelLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3FusedQkvAProjWithMqa,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
)
from tokenspeed.runtime.models.glm5 import (
    GlmMoeDsaAttention,
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
)
from tokenspeed.runtime.utils import add_prefix


class _Dots3NoteFusedQkvAProj(DeepseekV3FusedQkvAProjWithMqa):
    """Normalize NOTE's RoPE-only K slice without changing checkpoint names."""

    _rope_normalizer: Callable[[torch.Tensor], torch.Tensor] | None = None
    rope_dim: int = 0

    def forward(
        self,
        x: torch.Tensor,
        block_scale=None,
        output_dtype=None,
    ) -> torch.Tensor:
        output = super().forward(x, block_scale=block_scale, output_dtype=output_dtype)
        if self._rope_normalizer is not None and output.shape[0] > 0:
            rope = output[..., -self.rope_dim :]
            rope.copy_(self._rope_normalizer(rope))
        return output


class _Dots3NoteGatedOutputProjection(RowParallelLinear):
    """Apply a precomputed per-head gate immediately before NOTE's O projection."""

    output_gate: torch.Tensor | None = None
    local_heads: int
    value_dim: int

    def __init__(self, *args, local_heads: int, value_dim: int, **kwargs) -> None:
        self.local_heads = int(local_heads)
        self.value_dim = int(value_dim)
        super().__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor, *args, **kwargs):
        gate = self.output_gate
        if gate is None:
            raise RuntimeError(
                "Dots3 Note output projection is missing its attention gate."
            )
        gated = input_.view(-1, self.local_heads, self.value_dim)
        gated = (gated * gate.unsqueeze(-1)).flatten(-2)
        return super().forward(gated, *args, **kwargs)


def _headwise_gate(
    projection: ReplicatedLinear,
    hidden_states: torch.Tensor,
    *,
    local_heads: int,
    tp_rank: int,
) -> torch.Tensor:
    gate = projection(hidden_states)[0]
    if gate.shape[-1] != local_heads:
        start = tp_rank * local_heads
        if gate.shape[-1] < start + local_heads:
            raise RuntimeError(
                f"Dots3 Note gate width {gate.shape[-1]} cannot provide heads "
                f"[{start}, {start + local_heads})."
            )
        gate = gate.narrow(-1, start, local_heads)
    return torch.sigmoid(gate.float()).to(hidden_states.dtype)


def _project_headwise_gated_values(
    attention: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    projected = mla_project_value(attention, weight)
    return projected.view(attention.shape[0], attention.shape[1], -1).mul_(
        gate.unsqueeze(-1)
    )


class Dots3NoteFullAttention(GlmMoeDsaAttention):
    """Dots3 full-history DSA with NOTE normalization, scaling, and gating."""

    _MLA_KERNEL_BACKENDS = (*GlmMoeDsaAttention._MLA_KERNEL_BACKENDS, "dots3_note")
    _RAGGED_PREFILL_BACKENDS = (
        *GlmMoeDsaAttention._RAGGED_PREFILL_BACKENDS,
        "dots3_note",
    )

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results: bool = True,
        alt_stream: torch.cuda.Stream | None = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            skip_rope=skip_rope,
            is_nextn=is_nextn,
        )
        if getattr(config, "attention_gate_type", "headwise") != "headwise":
            raise NotImplementedError(
                "Dots3 Note currently supports headwise gates only."
            )

        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)
        self.fused_qk_layernorm = FusedRMSNorm(
            self.q_a_layernorm,
            self.kv_a_layernorm,
        )
        self.k_rope_only_layernorm = RMSNorm(
            qk_rope_head_dim,
            eps=config.rms_norm_eps,
        )
        fused = _Dots3NoteFusedQkvAProj(
            hidden_size,
            q_lora_rank + kv_lora_rank + qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
        )
        fused.rope_dim = qk_rope_head_dim
        object.__setattr__(
            fused,
            "_rope_normalizer",
            lambda rope: self.k_rope_only_layernorm(rope),
        )
        self.fused_qkv_a_proj_with_mqa = fused

        self.g_proj = ReplicatedLinear(
            hidden_size,
            num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("g_proj", prefix),
        )
        self.o_proj = _Dots3NoteGatedOutputProjection(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_attn_results,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            local_heads=self.num_local_heads,
            value_dim=v_head_dim,
        )
        self.q_lora_scale = (
            math.sqrt(hidden_size / q_lora_rank)
            if getattr(config, "apply_mla_qkv_lora_rescale", False)
            else 1.0
        )
        self.kv_lora_scale = (
            math.sqrt(hidden_size / kv_lora_rank)
            if getattr(config, "apply_mla_qkv_lora_rescale", False)
            else 1.0
        )
        self.attn_mqa.group_id = DOTS3_NOTE_FULL_DSA_GROUP
        self.attn_mha.group_id = DOTS3_NOTE_FULL_DSA_GROUP

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = _headwise_gate(
            self.g_proj,
            hidden_states,
            local_heads=self.num_local_heads,
            tp_rank=self.mapping.attn.tp_rank,
        )
        self.o_proj.output_gate = gate
        try:
            return super().forward(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                comm_manager,
                block_scale,
            )
        finally:
            self.o_proj.output_gate = None


class Dots3NoteSlidingAttention(DeepseekV3AttentionMLA):
    """Dots3 sliding MLA using the portable group-aware backend."""

    _MLA_KERNEL_BACKENDS = (*DeepseekV3AttentionMLA._MLA_KERNEL_BACKENDS, "dots3_note")

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None,
        layer_id: int,
        prefix: str,
        reduce_attn_results: bool,
        alt_stream: torch.cuda.Stream | None,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
        )
        if getattr(config, "swa_attention_gate_type", "headwise") != "headwise":
            raise NotImplementedError(
                "Dots3 Note currently supports headwise gates only."
            )
        self.g_proj = ReplicatedLinear(
            hidden_size,
            num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("g_proj", prefix),
        )
        self.k_rope_only_layernorm = RMSNorm(
            qk_rope_head_dim,
            eps=config.rms_norm_eps,
        )
        self.q_lora_scale = (
            math.sqrt(hidden_size / q_lora_rank)
            if getattr(config, "apply_mla_qkv_lora_rescale", False)
            else 1.0
        )
        self.kv_lora_scale = (
            math.sqrt(hidden_size / kv_lora_rank)
            if getattr(config, "apply_mla_qkv_lora_rescale", False)
            else 1.0
        )
        group_id = dots3_note_group_ids(config.num_hidden_layers)[layer_id]
        self.attn_mqa.group_id = group_id
        self.attn_mqa.sliding_window_size = int(config.sliding_window_size)
        self.attn_mha.group_id = group_id
        self.attn_mha.sliding_window_size = int(config.sliding_window_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv = self.fused_qkv_a_proj_with_mqa(
            hidden_states,
            block_scale,
            torch.bfloat16,
        )
        qkv = comm_manager.pre_attn_comm(qkv, ctx)
        q_a, latent = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        kv_a, k_pe = latent.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        q_a = self.q_a_layernorm(q_a) * self.q_lora_scale
        kv_a = self.kv_a_layernorm(kv_a) * self.kv_lora_scale
        k_pe = self.k_rope_only_layernorm(k_pe).unsqueeze(1)

        q = self.q_b_proj(q_a)[0].view(
            -1,
            self.num_local_heads,
            self.qk_head_dim,
        )
        q_nope, q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        if self.rotary_emb is None:
            raise RuntimeError(
                "Dots3 Note sliding attention requires rotary embedding."
            )
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        if self.w_kc is None or self.w_vc is None:
            raise RuntimeError("Dots3 Note MLA projection weights were not prepared.")

        absorbed_q = q.new_empty(
            q.shape[0],
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )
        bmm(
            q_nope.transpose(0, 1),
            self.w_kc.transpose(1, 2),
            out=absorbed_q[..., : self.kv_lora_rank].transpose(0, 1),
        )
        absorbed_q[..., self.kv_lora_rank :].copy_(q_pe)
        group_out_cache_loc = ctx.attn_backend.select_out_cache_loc(
            self.attn_mqa,
            out_cache_loc,
            ctx.forward_mode,
        )
        set_mla_kv_buffer = getattr(ctx.token_to_kv_pool, "set_mla_kv_buffer", None)
        if set_mla_kv_buffer is None:
            raise TypeError("Dots3 Note requires an MLA-compatible KV cache pool.")
        set_mla_kv_buffer(
            self.attn_mqa,
            group_out_cache_loc,
            kv_a.unsqueeze(1),
            k_pe,
        )
        latent_output = self.attn_mqa(
            absorbed_q,
            None,
            None,
            ctx,
            out_cache_loc,
            save_kv_cache=False,
        ).view(-1, self.num_local_heads, self.kv_lora_rank)

        gate = _headwise_gate(
            self.g_proj,
            hidden_states,
            local_heads=self.num_local_heads,
            tp_rank=self.mapping.attn.tp_rank,
        )
        projected = _project_headwise_gated_values(
            latent_output,
            self.w_vc,
            gate,
        )
        return self.o_proj(projected.flatten(-2))[0]


class Dots3NoteDecoderLayer(GlmMoeDsaDecoderLayer):
    """Select Dots3's full or sliding attention while reusing GLM orchestration."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if is_nextn:
            raise NotImplementedError(
                "Dots3 Note MTP continuation is not supported yet."
            )
        nn.Module.__init__(self)
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        layer_type = dots3_note_layer_types(config.num_hidden_layers)[layer_id]
        attention_quant_config = (
            None
            if "self_attn" in getattr(config, "disable_quant_module", [])
            else quant_config
        )
        attention_prefix = add_prefix("self_attn", prefix)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if layer_type == DOTS3_NOTE_FULL_DSA:
            self.self_attn = Dots3NoteFullAttention(
                config=config,
                mapping=mapping,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=get_rope_theta(config),
                rope_scaling=getattr(config, "rope_scaling", None),
                max_position_embeddings=max_position_embeddings,
                quant_config=attention_quant_config,
                layer_id=layer_id,
                prefix=attention_prefix,
                reduce_attn_results=False,
                alt_stream=alt_stream,
            )
        else:
            self.self_attn = Dots3NoteSlidingAttention(
                config=config,
                mapping=mapping,
                hidden_size=self.hidden_size,
                num_heads=config.swa_num_attention_heads,
                qk_nope_head_dim=config.swa_qk_nope_head_dim,
                qk_rope_head_dim=config.swa_qk_rope_head_dim,
                v_head_dim=config.swa_v_head_dim,
                q_lora_rank=config.swa_q_lora_rank,
                kv_lora_rank=config.swa_kv_lora_rank,
                rope_theta=config.swa_rope_theta,
                rope_scaling=None,
                max_position_embeddings=max_position_embeddings,
                quant_config=attention_quant_config,
                layer_id=layer_id,
                prefix=attention_prefix,
                reduce_attn_results=False,
                alt_stream=alt_stream,
            )

        self.layer_id = layer_id
        self.is_moe_layer = self._is_moe_layer(layer_id, False, config)
        if self.is_moe_layer:
            self.mlp = DeepseekV3MoE(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                is_shared_expert=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self._is_moe_layer(layer_id - 1, False, config),
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )


class Dots3NoteModel(DeepseekV3Model):
    """Dots3's 46-layer text decoder."""

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        from tokenspeed.runtime.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                Dots3NoteDecoderLayer(
                    config,
                    layer_id,
                    mapping=mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: set[int] = set()
        self._dflash_incremental_callback = None
        self._dflash_slot_bufs = None
        self._dflash_capture_idx_map: dict[int, int] = {}
        self._dflash_incr_active = False


class Dots3NoteForCausalLM(GlmMoeDsaForCausalLM):
    """Language-model-only Dots3 Note entry point."""

    model_cls = Dots3NoteModel

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        del mm_attention_backend
        if is_multimodal_active:
            raise NotImplementedError(
                "Dots3 Note currently supports text generation only; launch with "
                "--language-model-only."
            )
        if quant_config is not None:
            raise NotImplementedError(
                "Dots3 Note's initial implementation supports the BF16 checkpoint only."
            )
        self._dots3_lora_weights_scaled = False
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Return whether a checkpoint tensor can belong to the text decoder."""
        if name.startswith(("vision_encoder.", "audio_encoder.")):
            return False
        if name.startswith(("mtp.", "model.mtp.")):
            return False
        if name.startswith("model.layers.46."):
            return False
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        filtered = (
            (name, tensor)
            for name, tensor in weights
            if self.checkpoint_weight_name_filter(name)
        )
        super().load_weights(filtered)

    def post_load_weights(self) -> None:
        if not self._dots3_lora_weights_scaled:
            if self.model is None:
                raise RuntimeError("Dots3 Note model was not initialized.")
            with torch.no_grad():
                for layer in self.model.layers:
                    attention = layer.self_attn
                    if not isinstance(attention, Dots3NoteFullAttention):
                        continue
                    attention.q_a_layernorm.weight.data.mul_(attention.q_lora_scale)
                    attention.kv_a_layernorm.weight.data.mul_(attention.kv_lora_scale)
            self._dots3_lora_weights_scaled = True
        super().post_load_weights()


EntryClass = [Dots3NoteForCausalLM]


__all__ = [
    "Dots3NoteDecoderLayer",
    "Dots3NoteForCausalLM",
    "Dots3NoteFullAttention",
    "Dots3NoteModel",
    "Dots3NoteSlidingAttention",
]
