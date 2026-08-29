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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})
_MLA_DECODE_DTYPES = frozenset({torch.float16, torch.bfloat16}) | _FP8_DTYPES


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _mla_decode_kernel(
    Q,
    KV_Cache,
    O,
    LSE,
    page_table,
    cache_seqlens,
    sm_scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kv_page,
    stride_kv_token,
    stride_kv_head,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SEQLEN_K: tl.constexpr,
    Q_LEN: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    logit_cap: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_q = tl.program_id(1)
    cur_head = tl.program_id(2)

    offs_r = tl.arange(0, BLOCK_R)
    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_r = offs_r < KV_LORA_RANK
    mask_rope = offs_rope < QK_ROPE_HEAD_DIM

    q_base = cur_batch * stride_qb + cur_q * stride_qq + cur_head * stride_qh
    q_latent = tl.load(Q + q_base + offs_r, mask=mask_r, other=0.0)
    q_rope = tl.load(
        Q + q_base + KV_LORA_RANK + offs_rope,
        mask=mask_rope,
        other=0.0,
    )

    cache_len = tl.load(cache_seqlens + cur_batch)
    visible_start = 0
    visible_end = cache_len
    if WINDOW_LEFT >= 0:
        visible_end = cache_len - Q_LEN + cur_q + 1
        visible_start = tl.maximum(0, visible_end - (WINDOW_LEFT + 1))

    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    e_sum = 0.0
    e_max = -float("inf")

    for start_n in tl.range(visible_start, visible_end, BLOCK_N, num_stages=2):
        if WINDOW_LEFT < 0:
            start_n = tl.multiple_of(start_n, BLOCK_N)
        token_offsets = start_n + offs_n
        mask_n = token_offsets < visible_end
        page_indices = token_offsets // PAGE_SIZE
        page_offsets = token_offsets - page_indices * PAGE_SIZE
        physical_pages = tl.load(
            page_table + cur_batch * page_table_stride_b + page_indices,
            mask=mask_n,
            other=0,
        ).to(tl.int64)
        cache_base = (
            physical_pages * stride_kv_page
            + page_offsets * stride_kv_token
            + 0 * stride_kv_head
        )

        k_latent = tl.load(
            KV_Cache + cache_base[:, None] + offs_r[None, :],
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        k_rope = tl.load(
            KV_Cache + cache_base[:, None] + KV_LORA_RANK + offs_rope[None, :],
            mask=mask_n[:, None] & mask_rope[None, :],
            other=0.0,
        )

        qk = tl.sum(k_latent.to(tl.float32) * q_latent[None, :].to(tl.float32), axis=1)
        qk += tl.sum(k_rope.to(tl.float32) * q_rope[None, :].to(tl.float32), axis=1)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(mask_n, qk, float("-inf"))
        block_max = tl.max(qk, axis=0)
        block_max_fixed = tl.where(block_max == float("-inf"), -1e20, block_max)
        n_e_max = tl.maximum(block_max_fixed, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc = acc * old_scale + tl.sum(p[:, None] * k_latent.to(tl.float32), axis=0)
        e_sum = e_sum * old_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out_base = cur_batch * stride_ob + cur_q * stride_oq + cur_head * stride_oh
    tl.store(O + out_base + offs_r, acc / safe_sum, mask=mask_r)

    if HAS_LSE:
        lse = tl.where(e_sum > 0.0, tl.log(e_sum) + e_max, float("-inf"))
        tl.store(
            LSE
            + cur_batch * stride_lse_b
            + cur_q * stride_lse_q
            + cur_head * stride_lse_h,
            lse,
        )


@triton.jit
def _mla_split_decode_kernel(
    Q,
    KV_Cache,
    Partial,
    PartialLSE,
    page_table,
    cache_seqlens,
    sm_scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kv_page,
    stride_kv_token,
    page_table_stride_b: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    logit_cap: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    linear = tl.program_id(0)
    split = tl.program_id(1)
    cur_head = linear % NUM_HEADS
    cur_q = (linear // NUM_HEADS) % Q_LEN
    cur_batch = linear // (NUM_HEADS * Q_LEN)

    offs_r = tl.arange(0, BLOCK_R)
    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_r = offs_r < KV_LORA_RANK
    mask_rope = offs_rope < QK_ROPE_HEAD_DIM

    q_base = cur_batch * stride_qb + cur_q * stride_qq + cur_head * stride_qh
    q_latent = tl.load(Q + q_base + offs_r, mask=mask_r, other=0.0)
    q_rope = tl.load(
        Q + q_base + KV_LORA_RANK + offs_rope,
        mask=mask_rope,
        other=0.0,
    )

    cache_len = tl.load(cache_seqlens + cur_batch)
    visible_end = cache_len - Q_LEN + cur_q + 1
    visible_start = tl.maximum(0, visible_end - (WINDOW_LEFT + 1))
    visible_count = visible_end - visible_start
    split_start = visible_start + visible_count * split // NUM_SPLITS
    split_end = visible_start + visible_count * (split + 1) // NUM_SPLITS

    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    e_sum = 0.0
    e_max = -float("inf")

    for start_n in tl.range(split_start, split_end, BLOCK_N, num_stages=2):
        token_offsets = start_n + offs_n
        mask_n = token_offsets < split_end
        page_indices = token_offsets // PAGE_SIZE
        page_offsets = token_offsets - page_indices * PAGE_SIZE
        physical_pages = tl.load(
            page_table + cur_batch * page_table_stride_b + page_indices,
            mask=mask_n,
            other=0,
        ).to(tl.int64)
        cache_base = physical_pages * stride_kv_page + page_offsets * stride_kv_token
        k_latent = tl.load(
            KV_Cache + cache_base[:, None] + offs_r[None, :],
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        k_rope = tl.load(
            KV_Cache + cache_base[:, None] + KV_LORA_RANK + offs_rope[None, :],
            mask=mask_n[:, None] & mask_rope[None, :],
            other=0.0,
        )

        qk = tl.sum(k_latent.to(tl.float32) * q_latent[None, :].to(tl.float32), axis=1)
        qk += tl.sum(k_rope.to(tl.float32) * q_rope[None, :].to(tl.float32), axis=1)
        qk *= sm_scale
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(mask_n, qk, float("-inf"))
        block_max = tl.max(qk, axis=0)
        block_max_fixed = tl.where(block_max == float("-inf"), -1e20, block_max)
        n_e_max = tl.maximum(block_max_fixed, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc = acc * old_scale + tl.sum(p[:, None] * k_latent.to(tl.float32), axis=0)
        e_sum = e_sum * old_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    partial_base = (linear * NUM_SPLITS + split) * BLOCK_R
    tl.store(Partial + partial_base + offs_r, acc / safe_sum, mask=mask_r)
    lse = tl.where(e_sum > 0.0, tl.log(e_sum) + e_max, float("-inf"))
    tl.store(PartialLSE + linear * NUM_SPLITS + split, lse)


@triton.jit
def _mla_split_reduce_kernel(
    Partial,
    PartialLSE,
    O,
    LSE,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    Q_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    linear = tl.program_id(0)
    cur_head = linear % NUM_HEADS
    cur_q = (linear // NUM_HEADS) % Q_LEN
    cur_batch = linear // (NUM_HEADS * Q_LEN)
    offs_r = tl.arange(0, BLOCK_R)
    offs_s = tl.arange(0, NUM_SPLITS)
    mask_r = offs_r < KV_LORA_RANK

    partial_lse = tl.load(PartialLSE + linear * NUM_SPLITS + offs_s)
    valid = partial_lse != float("-inf")
    block_max = tl.max(tl.where(valid, partial_lse, -1e20), axis=0)
    weights = tl.where(valid, tl.exp(partial_lse - block_max), 0.0)
    weight_sum = tl.sum(weights, axis=0)
    partial = tl.load(
        Partial + (linear * NUM_SPLITS + offs_s[:, None]) * BLOCK_R + offs_r[None, :],
        mask=mask_r[None, :],
        other=0.0,
    )
    acc = tl.sum(partial * weights[:, None], axis=0)
    safe_sum = tl.where(weight_sum > 0.0, weight_sum, 1.0)
    out_base = cur_batch * stride_ob + cur_q * stride_oq + cur_head * stride_oh
    tl.store(O + out_base + offs_r, acc / safe_sum, mask=mask_r)

    if HAS_LSE:
        lse = tl.where(
            weight_sum > 0.0,
            tl.log(weight_sum) + block_max,
            float("-inf"),
        )
        tl.store(
            LSE
            + cur_batch * stride_lse_b
            + cur_q * stride_lse_q
            + cur_head * stride_lse_h,
            lse,
        )


def _sliding_num_splits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    window_left: int,
) -> int:
    if (
        not current_platform().is_blackwell
        or q.dtype != torch.bfloat16
        or kv_cache.dtype != torch.bfloat16
        or q.shape[1] != 1
        or qk_nope_head_dim != 192
        or kv_lora_rank != 1024
        or qk_rope_head_dim != 64
        or window_left != 512
    ):
        return 1

    programs = q.shape[0] * q.shape[1] * q.shape[2]
    if programs >= 128:
        return 1
    return min(8, triton.next_power_of_2((128 + programs - 1) // programs))


def _normalize_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 3:
        return kv_cache.unsqueeze(2)
    if kv_cache.dim() != 4:
        raise ValueError(f"kv_cache must be 3D or 4D, got {kv_cache.dim()}D")
    if kv_cache.shape[2] == 1:
        return kv_cache
    if kv_cache.shape[1] == 1:
        return kv_cache[:, 0].unsqueeze(2)
    raise ValueError(f"unsupported kv_cache shape {tuple(kv_cache.shape)}")


def mla_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    window_left: int = -1,
    logit_cap: float = 0.0,
    lse: torch.Tensor | None = None,
) -> None:
    if q.dim() != 4:
        raise ValueError(
            f"q must have shape [B, q_len, H, R + rope], got {tuple(q.shape)}"
        )
    if q.shape[1] < 1:
        raise ValueError("q_len must be at least 1")
    if window_left < 0 and q.shape[1] != 1:
        raise NotImplementedError("Triton MLA decode currently supports q_len == 1")
    if q.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"q head dim must be {kv_lora_rank + qk_rope_head_dim}, "
            f"got {q.shape[-1]}"
        )
    if out.shape != q.shape[:-1] + (kv_lora_rank,):
        raise ValueError(
            f"out shape must be {q.shape[:-1] + (kv_lora_rank,)}, "
            f"got {tuple(out.shape)}"
        )
    if q.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("q and out must have contiguous last dimension")
    if lse is not None and lse.shape != q.shape[:-1]:
        raise ValueError(f"lse shape must be {q.shape[:-1]}, got {tuple(lse.shape)}")

    kv_cache = _normalize_kv_cache(kv_cache)
    if kv_cache.shape[2] != 1:
        raise ValueError(f"MLA kv_cache must have one KV head, got {kv_cache.shape[2]}")
    if kv_cache.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"kv_cache head dim must be {kv_lora_rank + qk_rope_head_dim}, "
            f"got {kv_cache.shape[-1]}"
        )
    if kv_cache.stride(-1) != 1:
        raise ValueError("kv_cache must have contiguous last dimension")

    block_n = 16
    block_r = triton.next_power_of_2(kv_lora_rank)
    block_rope = triton.next_power_of_2(qk_rope_head_dim)
    grid = (q.shape[0], q.shape[1], q.shape[2])
    lse_arg = lse if lse is not None else out
    num_splits = _sliding_num_splits(
        q,
        kv_cache,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        window_left,
    )
    if num_splits > 1:
        linear_grid = q.shape[0] * q.shape[1] * q.shape[2]
        partial = torch.empty(
            (*q.shape[:-1], num_splits, block_r),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (*q.shape[:-1], num_splits),
            dtype=torch.float32,
            device=q.device,
        )
        _mla_split_decode_kernel[(linear_grid, num_splits)](
            q,
            kv_cache,
            partial,
            partial_lse,
            page_table,
            cache_seqlens,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            page_table.stride(0),
            kv_cache.shape[1],
            Q_LEN=q.shape[1],
            NUM_HEADS=q.shape[2],
            WINDOW_LEFT=window_left,
            logit_cap=logit_cap,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            BLOCK_R=block_r,
            BLOCK_ROPE=block_rope,
            BLOCK_N=block_n,
            NUM_SPLITS=num_splits,
            num_warps=8,
            num_stages=2,
        )
        _mla_split_reduce_kernel[(linear_grid,)](
            partial,
            partial_lse,
            out,
            lse_arg,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse_arg.stride(0),
            lse_arg.stride(1),
            lse_arg.stride(2),
            Q_LEN=q.shape[1],
            NUM_HEADS=q.shape[2],
            KV_LORA_RANK=kv_lora_rank,
            BLOCK_R=block_r,
            NUM_SPLITS=num_splits,
            HAS_LSE=lse is not None,
            num_warps=8,
        )
        return

    _mla_decode_kernel[grid](
        q,
        kv_cache,
        out,
        lse_arg,
        page_table,
        cache_seqlens,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse_arg.stride(0),
        lse_arg.stride(1),
        lse_arg.stride(2),
        page_table.stride(0),
        kv_cache.shape[1],
        max_seqlen_k,
        Q_LEN=q.shape[1],
        WINDOW_LEFT=window_left,
        logit_cap=logit_cap,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        BLOCK_R=block_r,
        BLOCK_ROPE=block_rope,
        BLOCK_N=block_n,
        HAS_LSE=lse is not None,
        num_warps=8,
        num_stages=2,
    )


@register_kernel(
    "attention",
    "mla_sliding_with_kvcache",
    name="triton_mla_sliding_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(("q", "kv_cache"), "dense", _MLA_DECODE_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
@register_kernel(
    "attention",
    "mla_decode_with_kvcache",
    name="triton_mla_decode_with_kvcache",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures(("q", "kv_cache"), "dense", _MLA_DECODE_DTYPES),
    priority=Priority.PORTABLE,
    traits={
        "q_len": frozenset({1}),
        "support_logit_cap": frozenset({False, True}),
        "return_lse": frozenset({False, True}),
    },
    tags={"portability"},
)
def triton_mla_decode_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    window_left: int = -1,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out_dtype = torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype
        out = torch.empty(
            q.shape[:-1] + (kv_lora_rank,), dtype=out_dtype, device=q.device
        )

    lse = (
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    mla_decode_fwd(
        q,
        kv_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen_k,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        softmax_scale,
        window_left=window_left,
        logit_cap=logit_cap,
        lse=lse,
    )
    if return_lse:
        return out, lse
    return out
