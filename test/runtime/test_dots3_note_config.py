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

import json
from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.models.dots3_note as dots3_note_model
from tokenspeed.runtime.configs.dots3_note_config import (
    DOTS3_NOTE_FULL_DSA,
    DOTS3_NOTE_FULL_DSA_LAYER_IDS,
    DOTS3_NOTE_SWA,
    Dots3NoteConfig,
    dots3_note_group_ids,
    dots3_note_layer_types,
)
from tokenspeed.runtime.configs.model_config import (
    _ATTENTION_FAMILY_SPECS,
    AttentionArch,
    _apply_attention_family_defaults,
    configure_dots3_note_attention,
    is_audio_model,
    is_multimodal_model,
)
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.dots3_note import (
    Dots3NoteRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.registry import _BACKEND_REGISTRY
from tokenspeed.runtime.models.dots3_note import (
    Dots3NoteForCausalLM,
    Dots3NoteFullAttention,
    Dots3NoteSlidingAttention,
    EntryClass,
)
from tokenspeed.runtime.models.glm5 import GlmMoeDsaForCausalLM
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import _CONFIG_REGISTRY


def test_dots3_note_config_registry_and_required_defaults():
    config = Dots3NoteConfig(
        architectures=["Dots3NoteForCausalLM"],
        num_hidden_layers=46,
    )

    assert _CONFIG_REGISTRY["dots3_note"] is Dots3NoteConfig
    assert config.n_group == 1
    assert config.topk_group == 1
    assert config.indexer_rope_interleave is True
    assert config.num_nextn_predict_layers == 1


def test_dots3_note_config_from_pretrained_supports_default_construction(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dots3_note",
                "architectures": ["Dots3NoteForCausalLM"],
                "num_hidden_layers": 46,
            }
        )
    )

    config = Dots3NoteConfig.from_pretrained(tmp_path)

    assert config.num_hidden_layers == 46
    assert Dots3NoteConfig().num_hidden_layers == 46


def test_dots3_note_rejects_incompatible_expert_or_indexer_routing():
    with pytest.raises(ValueError, match="n_group=1 and topk_group=1"):
        Dots3NoteConfig(num_hidden_layers=46, n_group=8, topk_group=4)
    with pytest.raises(ValueError, match="adjacent-pair RoPE"):
        Dots3NoteConfig(num_hidden_layers=46, indexer_rope_interleave=False)


def test_dots3_note_layer_routing_and_mtp_continuation():
    layer_types = dots3_note_layer_types()
    group_ids = dots3_note_group_ids()

    assert len(layer_types) == 46
    assert (
        tuple(
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type == DOTS3_NOTE_FULL_DSA
        )
        == DOTS3_NOTE_FULL_DSA_LAYER_IDS
    )
    assert layer_types.count(DOTS3_NOTE_FULL_DSA) == 13
    assert layer_types.count(DOTS3_NOTE_SWA) == 33
    assert group_ids.count("full_dsa") == 13
    assert group_ids.count("sliding_mla_0") == 11
    assert group_ids.count("sliding_mla_1") == 11
    assert group_ids.count("sliding_mla_2") == 11

    mtp_group_ids = dots3_note_group_ids(47)
    assert mtp_group_ids[-1] == "sliding_mla_0"
    assert mtp_group_ids.count("sliding_mla_0") == 12
    assert mtp_group_ids.count("sliding_mla_1") == 11
    assert mtp_group_ids.count("sliding_mla_2") == 11


def test_dots3_note_family_defaults_enable_decode_graph_before_executor_setup():
    server_args = SimpleNamespace(
        prefix_granularity=64,
        attention_backend=None,
        enforce_eager=False,
        disable_prefill_graph=False,
        disable_autotune=False,
        enable_prefix_caching=True,
        disable_kvstore=False,
    )
    spec = next(spec for spec in _ATTENTION_FAMILY_SPECS if spec.name == "Dots3 Note")

    _apply_attention_family_defaults(server_args, spec)

    assert server_args.enforce_eager is False
    assert server_args.disable_prefill_graph is True
    assert server_args.disable_autotune is True
    assert server_args.enable_prefix_caching is False
    assert server_args.disable_kvstore is True
    assert server_args.attention_backend == "dots3_note"


def test_dots3_note_attention_config_rejects_mixed_batch_at_startup():
    server_args = SimpleNamespace(
        speculative_algorithm=None,
        disaggregation_mode="null",
        enable_mixed_batch=True,
    )

    with pytest.raises(NotImplementedError, match="--enable-mixed-batch"):
        Dots3NoteAttentionConfig.generate(server_args, model_config=None)


def test_dots3_note_attention_config_rejects_mtp_target_layer():
    server_args = SimpleNamespace(
        speculative_algorithm=None,
        disaggregation_mode="null",
        enable_mixed_batch=False,
    )

    with pytest.raises(NotImplementedError, match="exactly 46 target layers"):
        Dots3NoteAttentionConfig.generate(
            server_args,
            model_config=SimpleNamespace(num_attention_layers=47),
        )


def test_dots3_note_attention_config_rejects_non_bf16_model(monkeypatch):
    monkeypatch.setattr(
        DSAConfig,
        "generate",
        classmethod(
            lambda cls, server_args, model_config, is_draft=False: SimpleNamespace(
                dtype=torch.float16
            )
        ),
    )
    server_args = SimpleNamespace(
        speculative_algorithm=None,
        disaggregation_mode="null",
        enable_mixed_batch=False,
    )

    with pytest.raises(ValueError, match="requires a BF16 model"):
        Dots3NoteAttentionConfig.generate(
            server_args,
            model_config=SimpleNamespace(num_attention_layers=46),
        )


def test_dots3_note_attention_config_rejects_non_official_cache_geometry(monkeypatch):
    monkeypatch.setattr(
        DSAConfig,
        "generate",
        classmethod(
            lambda cls, server_args, model_config, is_draft=False: SimpleNamespace(
                dtype=torch.bfloat16,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
            )
        ),
    )
    server_args = SimpleNamespace(
        speculative_algorithm=None,
        disaggregation_mode="null",
        enable_mixed_batch=False,
    )
    model_config = SimpleNamespace(
        num_attention_layers=46,
        hf_text_config=SimpleNamespace(
            sliding_window_size=513,
            swa_kv_lora_rank=512,
            swa_qk_rope_head_dim=64,
        ),
    )

    with pytest.raises(ValueError, match="requires cache geometry"):
        Dots3NoteAttentionConfig.generate(server_args, model_config)


def test_dots3_note_attention_family_preserves_both_geometries():
    hf_config = SimpleNamespace(
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=64,
        swa_num_attention_heads=64,
        swa_num_key_value_heads=64,
        swa_q_lora_rank=1024,
        swa_kv_lora_rank=1024,
        swa_qk_nope_head_dim=192,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
        sliding_window_size=513,
    )
    model_config = SimpleNamespace(hf_text_config=hf_config)

    configure_dots3_note_attention(model_config)

    assert model_config.attention_arch == AttentionArch.DOTS3_NOTE
    assert model_config.head_dim == 192
    assert model_config.kv_lora_rank == 512
    assert model_config.index_topk == 2048
    assert model_config.swa_num_attention_heads == 64
    assert model_config.swa_num_key_value_heads == 64
    assert model_config.swa_q_lora_rank == 1024
    assert model_config.swa_kv_lora_rank == 1024
    assert model_config.swa_qk_nope_head_dim == 192
    assert model_config.sliding_window_size == 513


def _dots3_recipe(num_layers: int) -> Dots3NoteRecipe:
    attn_config = SimpleNamespace(
        prefix_granularity=64,
        layer_types=dots3_note_layer_types(num_layers),
        group_ids=dots3_note_group_ids(num_layers),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        swa_kv_lora_rank=1024,
        swa_qk_rope_head_dim=64,
        sliding_window_tokens=513,
        max_bs=4,
        context_len=524288,
    )
    return Dots3NoteRecipe(
        server_args=SimpleNamespace(
            max_total_tokens=None,
            chunked_prefill_size=8192,
        ),
        model_config=SimpleNamespace(num_attention_layers=num_layers),
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 30,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def test_dots3_note_flatkv_layout_matches_fixed_geometry():
    recipe = _dots3_recipe(46)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )

    assert [spec.group_id for spec, _ in groups] == [
        "full_dsa",
        "sliding_mla_0",
        "sliding_mla_1",
        "sliding_mla_2",
    ]
    assert layout.lcm_block_bytes == 18_173_952
    assert dict(layout.group_packing) == {
        "full_dsa": 17,
        "sliding_mla_0": 9,
        "sliding_mla_1": 9,
        "sliding_mla_2": 9,
    }
    fields = {field.field_id: field for field in layout.fields}
    assert fields["layer.0.latent_kv"].shape == (64, 1, 576)
    assert fields["layer.0.latent_kv"].page_stride_bytes == 73_728
    assert fields["layer.0.index_k"].shape == (64, 132)
    assert fields["layer.0.index_k"].page_stride_bytes == 8_448
    assert fields["layer.2.latent_kv"].shape == (64, 1, 1088)
    assert fields["layer.2.latent_kv"].page_stride_bytes == 139_264


def test_dots3_note_recipe_rejects_mtp_continuation():
    recipe = _dots3_recipe(46)
    recipe.draft_model_config = SimpleNamespace(num_attention_layers=1)
    recipe.draft_attn_config = SimpleNamespace()

    with pytest.raises(NotImplementedError, match="does not yet support MTP"):
        recipe.groups()


def test_dots3_note_recipe_capacity_accounts_for_all_cache_groups():
    recipe = _dots3_recipe(46)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )

    parents = recipe.num_lcm_blocks(layout)
    capacity = recipe.token_capacity(layout, parents)
    flat_capacity = parents * 17 * recipe.prefix_granularity

    assert capacity < flat_capacity
    assert recipe.parents_needed(layout, capacity) <= parents
    assert recipe.parents_needed(layout, capacity + 1) > parents


def test_dots3_note_tiny_model_constructs_both_attention_types_and_loads_fused_qkv(
    monkeypatch,
):
    monkeypatch.setitem(global_server_args_dict, "attention_backend", "dots3_note")
    monkeypatch.setattr(torch.cuda, "Stream", lambda: None)
    config = Dots3NoteConfig(
        architectures=["Dots3NoteForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=4,
        qk_rope_head_dim=2,
        v_head_dim=2,
        qk_nope_head_dim=2,
        num_experts_per_tok=1,
        first_k_dense_replace=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        moe_layer_freq=1,
        index_topk=2,
        index_n_heads=2,
        index_head_dim=4,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        attention_gate_type="headwise",
        swa_attention_gate_type="headwise",
        apply_mla_qkv_lora_rescale=True,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_q_lora_rank=4,
        swa_kv_lora_rank=4,
        swa_qk_nope_head_dim=2,
        swa_qk_rope_head_dim=2,
        swa_v_head_dim=2,
        swa_rope_theta=50_000.0,
        sliding_window_size=513,
    )
    model = Dots3NoteForCausalLM(
        config,
        Mapping(rank=0, world_size=1),
        is_multimodal_active=False,
    )

    assert isinstance(model.model.layers[0].self_attn, Dots3NoteFullAttention)
    assert isinstance(model.model.layers[1].self_attn, Dots3NoteFullAttention)
    assert isinstance(model.model.layers[2].self_attn, Dots3NoteSlidingAttention)
    assert [type(layer.mlp).__name__ for layer in model.model.layers] == [
        "DeepseekV3MLP",
        "DeepseekV3MoE",
        "DeepseekV3MoE",
    ]

    model.load_weights(
        [
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                torch.ones(4, 16),
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                torch.full((6, 16), 2.0),
            ),
        ]
    )
    fused = model.model.layers[0].self_attn.fused_qkv_a_proj_with_mqa.weight
    torch.testing.assert_close(fused[:4], torch.ones_like(fused[:4]))
    torch.testing.assert_close(fused[4:10], torch.full_like(fused[4:10], 2.0))


def test_dots3_note_model_and_backend_are_registered():
    assert EntryClass == [Dots3NoteForCausalLM]
    supported_archs, backend_cls = _BACKEND_REGISTRY["dots3_note"]
    assert supported_archs == {AttentionArch.DOTS3_NOTE}
    assert backend_cls.__name__ == "Dots3NoteAttentionBackend"
    assert getattr(backend_cls, "supports_eager_comm_prewarm", True) is True


def test_dots3_note_requires_language_model_only():
    with pytest.raises(NotImplementedError, match="--language-model-only"):
        Dots3NoteForCausalLM(
            config=SimpleNamespace(),
            mapping=None,
            is_multimodal_active=True,
        )


def test_dots3_note_checkpoint_filter_excludes_non_text_and_mtp_weights():
    model = object.__new__(Dots3NoteForCausalLM)

    assert not model.checkpoint_weight_name_filter("vision_encoder.blocks.0.weight")
    assert not model.checkpoint_weight_name_filter("audio_encoder.layers.0.weight")
    assert not model.checkpoint_weight_name_filter(
        "model.layers.46.self_attn.q_b_proj.weight"
    )
    assert not model.checkpoint_weight_name_filter("model.mtp.fc.weight")
    assert model.checkpoint_weight_name_filter(
        "model.layers.45.self_attn.q_b_proj.weight"
    )
    assert model.checkpoint_weight_name_filter("lm_head.weight")


def test_dots3_note_post_load_folds_lora_scales_into_full_attention_norms_once(
    monkeypatch,
):
    class FakeAttention:
        def __init__(self):
            self.q_lora_scale = 2.0
            self.kv_lora_scale = 3.0
            self.q_a_layernorm = SimpleNamespace(
                weight=torch.nn.Parameter(torch.ones(1))
            )
            self.kv_a_layernorm = SimpleNamespace(
                weight=torch.nn.Parameter(torch.ones(1))
            )

    monkeypatch.setattr(dots3_note_model, "Dots3NoteFullAttention", FakeAttention)
    monkeypatch.setattr(GlmMoeDsaForCausalLM, "post_load_weights", lambda self: None)
    attention = FakeAttention()
    model = object.__new__(Dots3NoteForCausalLM)
    object.__setattr__(model, "_dots3_lora_weights_scaled", False)
    object.__setattr__(
        model,
        "model",
        SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
    )

    model.post_load_weights()
    model.post_load_weights()

    torch.testing.assert_close(attention.q_a_layernorm.weight, torch.tensor([2.0]))
    torch.testing.assert_close(attention.kv_a_layernorm.weight, torch.tensor([3.0]))


def test_dots3_note_is_classified_as_multimodal_audio_model():
    architectures = ["Dots3NoteForCausalLM"]

    assert is_multimodal_model(architectures)
    assert is_audio_model(architectures)
