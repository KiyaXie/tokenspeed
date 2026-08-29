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

from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.layers.attention.kv_cache.dots3_note as dots3_cache
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.dots3_note import (
    Dots3NoteTokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool


def _fake_arena():
    tensors = {
        "layer.0.latent_kv": torch.zeros((64, 1, 576), dtype=torch.bfloat16),
        "layer.0.index_k": torch.zeros((64, 132), dtype=torch.uint8),
        "layer.1.latent_kv": torch.zeros((64, 1, 1088), dtype=torch.bfloat16),
    }
    return SimpleNamespace(
        size=64,
        kv_page_size=64,
        plan=SimpleNamespace(
            fields=tuple(SimpleNamespace(field_id=field_id) for field_id in tensors)
        ),
        field=tensors.__getitem__,
        buffer=torch.empty(1, dtype=torch.uint8),
    )


def _pool():
    return Dots3NoteTokenToKVPool(
        arena=_fake_arena(),
        dtype=torch.bfloat16,
        model_dtype=torch.bfloat16,
        quant_method="none",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        swa_kv_lora_rank=1024,
        swa_qk_rope_head_dim=64,
        layer_num=2,
        rank=0,
        index_head_dim=128,
    )


def test_dots3_note_pool_uses_per_layer_latent_geometry(monkeypatch):
    pool = _pool()

    assert pool.get_key_buffer(0).shape[-1] == 576
    assert pool.get_value_buffer(0).shape[-1] == 512
    assert pool.get_key_buffer(1).shape[-1] == 1088
    assert pool.get_value_buffer(1).shape[-1] == 1024

    def fake_gather(buffer, loc, nope, rope):
        rows = buffer[loc]
        nope.copy_(rows[..., : nope.shape[-1]])
        rope.copy_(rows[..., nope.shape[-1] :])

    monkeypatch.setattr(dots3_cache, "get_mla_kv_buffer_triton", fake_gather)
    full_nope, full_rope = pool.get_mla_kv_buffer(
        SimpleNamespace(layer_id=0), torch.tensor([0])
    )
    swa_nope, swa_rope = pool.get_mla_kv_buffer(
        SimpleNamespace(layer_id=1), torch.tensor([0])
    )

    assert full_nope.shape == (1, 1, 512)
    assert full_rope.shape == (1, 1, 64)
    assert swa_nope.shape == (1, 1, 1024)
    assert swa_rope.shape == (1, 1, 64)


def test_dots3_note_pool_rejects_index_cache_access_for_sliding_layer():
    pool = _pool()

    with pytest.raises(ValueError, match="has no index-K cache"):
        pool.get_index_k_buffer(1)


def test_cache_factory_dispatches_dots3_before_generic_dsa():
    config = object.__new__(Dots3NoteAttentionConfig)
    config.kv_cache_dtype = torch.bfloat16
    config.dtype = torch.bfloat16
    config.kv_cache_quant_method = "none"
    config.kv_lora_rank = 512
    config.qk_rope_head_dim = 64
    config.swa_kv_lora_rank = 1024
    config.swa_qk_rope_head_dim = 64
    config.index_head_dim = 128

    pool = create_cache_pool(
        SimpleNamespace(family="dots3_note"),
        config,
        _fake_arena(),
        num_layers=2,
        rank=0,
    )

    assert isinstance(pool, Dots3NoteTokenToKVPool)
