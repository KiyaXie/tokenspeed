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

"""FlatKV packing recipe for mixed full-DSA/sliding-MLA Dots3 Note."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from typing_extensions import override

from tokenspeed.runtime.configs.dots3_note_config import (
    DOTS3_NOTE_FULL_DSA_GROUP,
    DOTS3_NOTE_SWA_GROUP_PREFIX,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheGroupDeclaration,
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

_DOTS3_PAGE_TOKENS = 64
_DOTS3_FULL_PACKING = 17
_DOTS3_SWA_PACKING = 9
_DOTS3_INDEX_K_ROW_BYTES = 132


class Dots3NoteRecipe(CacheRecipe):
    """Pack Dots3's 13 DSA layers and three striped SWA layer groups."""

    family = "dots3_note"

    @property
    @override
    def layer_types(self) -> tuple[str, ...]:
        return tuple(self.attn_config.layer_types)

    @property
    @override
    def group_ids(self) -> tuple[str, ...]:
        return tuple(self.attn_config.group_ids)

    @property
    @override
    def alignment(self) -> int:
        return 256

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 1 / 3

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        if self.draft_attn_config is not None:
            raise NotImplementedError(
                "Dots3 Note cache setup does not yet support MTP continuation."
            )
        if self.prefix_granularity != _DOTS3_PAGE_TOKENS:
            raise ValueError(
                "Dots3 Note FlatKV requires 64-token logical pages, "
                + f"got {self.prefix_granularity}."
            )
        if len(self.group_ids) != self.num_target_layers + self.num_draft_layers:
            raise ValueError("Dots3 Note cache group ids must cover every cache layer")

        layer_ids_by_group: dict[str, list[int]] = defaultdict(list)
        for layer_id, group_id in enumerate(self.group_ids):
            layer_ids_by_group[group_id].append(layer_id)

        declarations: list[CacheGroupDeclaration] = []
        for group_id in (
            DOTS3_NOTE_FULL_DSA_GROUP,
            "sliding_mla_0",
            "sliding_mla_1",
            "sliding_mla_2",
        ):
            layer_ids = layer_ids_by_group[group_id]
            if not layer_ids:
                continue
            is_full = group_id == DOTS3_NOTE_FULL_DSA_GROUP
            spec = CacheGroupSpec(
                group_id=group_id,
                retention="full_history" if is_full else "sliding_window",
                rows_per_page=_DOTS3_PAGE_TOKENS,
                entry_stride_tokens=1,
                sliding_window_tokens=(
                    None if is_full else self.attn_config.sliding_window_tokens
                ),
            )
            fields: list[CacheFieldSpec] = []
            for occurrence, layer_id in enumerate(layer_ids):
                latent_width = int(
                    self.attn_config.kv_lora_rank + self.attn_config.qk_rope_head_dim
                    if is_full
                    else self.attn_config.swa_kv_lora_rank
                    + self.attn_config.swa_qk_rope_head_dim
                )
                fields.append(
                    CacheFieldSpec(
                        f"layer.{layer_id}.latent_kv",
                        f"slot.{occurrence}",
                        (_DOTS3_PAGE_TOKENS, 1, latent_width),
                        "bfloat16",
                    )
                )
                if is_full:
                    fields.append(
                        CacheFieldSpec(
                            f"layer.{layer_id}.index_k",
                            f"layer.{layer_id}.index_k",
                            (_DOTS3_PAGE_TOKENS, _DOTS3_INDEX_K_ROW_BYTES),
                            "uint8",
                        )
                    )
            declarations.append((spec, tuple(fields)))
        return tuple(declarations)

    @override
    def packing(self, groups: Sequence[CacheGroupDeclaration]) -> Mapping[str, int]:
        expected_group_ids = {
            DOTS3_NOTE_FULL_DSA_GROUP,
            *(f"{DOTS3_NOTE_SWA_GROUP_PREFIX}{index}" for index in range(3)),
        }
        actual_group_ids = {spec.group_id for spec, _ in groups}
        if actual_group_ids != expected_group_ids:
            raise ValueError(
                "Dots3 Note cache recipe requires full_dsa and all three "
                + f"sliding_mla groups, got {sorted(actual_group_ids)}."
            )
        return {
            DOTS3_NOTE_FULL_DSA_GROUP: _DOTS3_FULL_PACKING,
            "sliding_mla_0": _DOTS3_SWA_PACKING,
            "sliding_mla_1": _DOTS3_SWA_PACKING,
            "sliding_mla_2": _DOTS3_SWA_PACKING,
        }

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        usable_bytes = self.cache_budget_bytes - self.workspace_bytes()
        budgeted = usable_bytes // layout.lcm_block_bytes - 1
        if budgeted < 1:
            raise ValueError(
                "dots3_note cache budget must hold a null parent and one usable LCM parent"
            )
        if self.token_limit is None:
            return budgeted
        return min(budgeted, self.parents_needed(layout, self.token_limit))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        upper_bound = self.token_limit
        if upper_bound is None:
            upper_bound = (
                num_lcm_blocks
                * dict(layout.group_packing)[DOTS3_NOTE_FULL_DSA_GROUP]
                * layout.prefix_granularity
            )
        return self._capacity_from_parents(
            layout,
            num_lcm_blocks,
            upper_bound=upper_bound,
        )


__all__ = ["Dots3NoteRecipe"]
