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

import math
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.dots3_note import (
    Dots3NoteAttentionBackend,
    _portable_sliding_mla,
)
from tokenspeed.runtime.models.dots3_note import _project_headwise_gated_values


def _paged_cache(sequence: torch.Tensor, page_size: int, hole_pages: int = 0):
    num_pages = math.ceil(sequence.shape[0] / page_size)
    physical_pages = list(reversed(range(1, num_pages + 1)))
    table = torch.tensor(physical_pages, dtype=torch.int32)
    cache = torch.zeros(
        ((num_pages + 1) * page_size, 1, sequence.shape[-1]),
        dtype=sequence.dtype,
    )
    for logical_page, physical_page in enumerate(physical_pages):
        start = logical_page * page_size
        end = min(start + page_size, sequence.shape[0])
        cache[
            physical_page * page_size : physical_page * page_size + end - start,
            0,
        ] = sequence[start:end]
    table[:hole_pages] = 0
    return cache, table


def test_dots3_note_headwise_gate_is_applied_after_value_projection():
    torch.manual_seed(0)
    attention = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4, 5)
    gate = torch.sigmoid(torch.randn(2, 3))

    actual = _project_headwise_gated_values(attention, weight, gate)
    expected = torch.einsum("bhl,hlv->bhv", attention, weight) * gate.unsqueeze(-1)

    torch.testing.assert_close(actual, expected)


def test_dots3_note_decode_write_locations_are_group_specific():
    page_tables = {
        "full_dsa": torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
        "sliding_mla_0": torch.tensor([[30, 31, 32], [40, 41, 42]], dtype=torch.int32),
        "sliding_mla_1": torch.tensor([[50, 51, 52], [60, 61, 62]], dtype=torch.int32),
        "sliding_mla_2": torch.tensor([[70, 71, 72], [80, 81, 82]], dtype=torch.int32),
    }

    locations = Dots3NoteAttentionBackend._decode_out_cache_locs(
        page_tables,
        torch.tensor([65, 130], dtype=torch.int32),
        page_size=64,
    )

    expected = {
        "full_dsa": [704, 1409],
        "sliding_mla_0": [1984, 2689],
        "sliding_mla_1": [3264, 3969],
        "sliding_mla_2": [4544, 5249],
    }
    for group_id, values in expected.items():
        assert locations[group_id].tolist() == values


def test_dots3_note_cuda_graph_metadata_replays_in_place():
    backend = object.__new__(Dots3NoteAttentionBackend)
    backend._full_dsa = Mock()
    backend.device = torch.device("cpu")
    backend.max_num_pages = 3
    backend.kernel_page_size = 64
    backend.sliding_group_ids = (
        "sliding_mla_0",
        "sliding_mla_1",
        "sliding_mla_2",
    )

    backend.init_cuda_graph_state(max_bs=2)
    seq_lens = torch.tensor([1, 65], dtype=torch.int32)
    group_ids = ("full_dsa", *backend.sliding_group_ids)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.arange(2, dtype=torch.int32),
        seq_lens=seq_lens,
        forward_mode=ForwardMode.DECODE,
        cache_group_ids=group_ids,
    )
    seq_lens_pointer = backend.sliding_metadata.seq_lens.data_ptr()
    page_table_pointers = {
        group_id: table.data_ptr()
        for group_id, table in backend.sliding_metadata.page_tables.items()
    }
    out_cache_loc_pointers = {
        group_id: locs.data_ptr()
        for group_id, locs in backend.sliding_metadata.out_cache_locs.items()
    }

    block_tables = {
        "full_dsa": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        "sliding_mla_0": torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
        "sliding_mla_1": torch.tensor([[30, 31, 32], [40, 41, 42]], dtype=torch.int32),
        "sliding_mla_2": torch.tensor([[50, 51, 52], [60, 61, 62]], dtype=torch.int32),
    }
    backend.init_forward_metadata_replay_cuda_graph(
        bs=2,
        req_pool_indices=torch.arange(2, dtype=torch.int32),
        seq_lens=torch.tensor([65, 130], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        page_table=torch.zeros((2, 3), dtype=torch.int32),
        block_tables=block_tables,
    )

    assert backend.sliding_metadata.seq_lens.tolist() == [65, 130]
    assert backend.sliding_metadata.seq_lens.data_ptr() == seq_lens_pointer
    assert backend.sliding_metadata.out_cache_locs["sliding_mla_0"].tolist() == [
        704,
        1409,
    ]
    assert {
        group_id: table.data_ptr()
        for group_id, table in backend.sliding_metadata.page_tables.items()
    } == page_table_pointers
    assert {
        group_id: locs.data_ptr()
        for group_id, locs in backend.sliding_metadata.out_cache_locs.items()
    } == out_cache_loc_pointers

    with pytest.raises(RuntimeError, match="requires cache groups"):
        backend.init_forward_metadata_replay_cuda_graph(
            bs=2,
            req_pool_indices=torch.arange(2, dtype=torch.int32),
            seq_lens=seq_lens,
            forward_mode=ForwardMode.DECODE,
            page_table=torch.zeros((2, 3), dtype=torch.int32),
            block_tables={
                key: value
                for key, value in block_tables.items()
                if key != "sliding_mla_2"
            },
        )


def test_dots3_note_extend_write_locations_are_request_major():
    page_tables = {
        "full_dsa": torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        "sliding_mla_0": torch.tensor([[30, 31], [40, 41]], dtype=torch.int32),
        "sliding_mla_1": torch.tensor([[50, 51], [60, 61]], dtype=torch.int32),
        "sliding_mla_2": torch.tensor([[70, 71], [80, 81]], dtype=torch.int32),
    }

    locations = Dots3NoteAttentionBackend._extend_out_cache_locs(
        page_tables,
        prefix_lens=(3, 0),
        query_lens=(3, 2),
        page_size=4,
    )

    expected = {
        "full_dsa": [43, 44, 45, 80, 81],
        "sliding_mla_0": [123, 124, 125, 160, 161],
        "sliding_mla_1": [203, 204, 205, 240, 241],
        "sliding_mla_2": [283, 284, 285, 320, 321],
    }
    for group_id, values in expected.items():
        assert locations[group_id].tolist() == values


def _reference(
    q: torch.Tensor,
    sequence: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    kv_lora_rank: int,
    window_size: int,
    scaling: float,
):
    outputs = []
    first_query = seq_len - query_len
    for query_offset in range(query_len):
        position = first_query + query_offset
        start = max(0, position - window_size + 1)
        keys = sequence[start : position + 1]
        scores = torch.einsum("hd,kd->hk", q[query_offset], keys).float() * scaling
        probabilities = torch.softmax(scores, dim=-1).to(sequence.dtype)
        outputs.append(torch.einsum("hk,kr->hr", probabilities, keys[:, :kv_lora_rank]))
    return torch.stack(outputs).flatten(1)


@pytest.mark.parametrize("seq_len", [1, 2, 511, 512, 513, 514])
def test_portable_sliding_mla_matches_absolute_window_boundaries(seq_len):
    torch.manual_seed(seq_len)
    num_heads = 2
    kv_lora_rank = 3
    rope_dim = 2
    page_size = 64
    window_size = 513
    scaling = (kv_lora_rank + rope_dim) ** -0.5
    sequence = torch.randn(seq_len, kv_lora_rank + rope_dim, dtype=torch.float32)
    q = torch.randn(1, num_heads, kv_lora_rank + rope_dim, dtype=torch.float32)
    cache, table = _paged_cache(sequence, page_size)

    actual = _portable_sliding_mla(
        q,
        cache,
        table,
        seq_len=seq_len,
        query_len=1,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        window_size=window_size,
        scaling=scaling,
    )
    expected = _reference(
        q,
        sequence,
        seq_len=seq_len,
        query_len=1,
        kv_lora_rank=kv_lora_rank,
        window_size=window_size,
        scaling=scaling,
    )

    torch.testing.assert_close(actual, expected)


def test_portable_sliding_mla_ignores_expired_front_holes():
    torch.manual_seed(0)
    seq_len = 641
    page_size = 64
    window_size = 513
    num_heads = 2
    kv_lora_rank = 3
    rope_dim = 2
    scaling = (kv_lora_rank + rope_dim) ** -0.5
    sequence = torch.randn(seq_len, kv_lora_rank + rope_dim)
    q = torch.randn(1, num_heads, kv_lora_rank + rope_dim)
    cache, table = _paged_cache(sequence, page_size, hole_pages=2)

    actual = _portable_sliding_mla(
        q,
        cache,
        table,
        seq_len=seq_len,
        query_len=1,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        window_size=window_size,
        scaling=scaling,
    )
    expected = _reference(
        q,
        sequence,
        seq_len=seq_len,
        query_len=1,
        kv_lora_rank=kv_lora_rank,
        window_size=window_size,
        scaling=scaling,
    )

    torch.testing.assert_close(actual, expected)


def test_portable_sliding_mla_applies_per_query_causal_mask_for_prefill():
    torch.manual_seed(1)
    seq_len = 70
    query_len = 70
    num_heads = 2
    kv_lora_rank = 3
    rope_dim = 2
    page_size = 64
    window_size = 513
    scaling = (kv_lora_rank + rope_dim) ** -0.5
    sequence = torch.randn(seq_len, kv_lora_rank + rope_dim)
    q = torch.randn(query_len, num_heads, kv_lora_rank + rope_dim)
    cache, table = _paged_cache(sequence, page_size)

    actual = _portable_sliding_mla(
        q,
        cache,
        table,
        seq_len=seq_len,
        query_len=query_len,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        page_size=page_size,
        window_size=window_size,
        scaling=scaling,
        query_chunk_size=16,
    )
    expected = _reference(
        q,
        sequence,
        seq_len=seq_len,
        query_len=query_len,
        kv_lora_rank=kv_lora_rank,
        window_size=window_size,
        scaling=scaling,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
