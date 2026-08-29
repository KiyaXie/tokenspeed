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

"""Configuration and layer routing for the Dots3 Note model family."""

from typing import Any

from transformers import DeepseekV3Config

DOTS3_NOTE_NUM_TARGET_LAYERS = 46
DOTS3_NOTE_MTP_LAYER_ID = 46
DOTS3_NOTE_FULL_DSA_LAYER_IDS = (
    0,
    1,
    5,
    9,
    13,
    17,
    21,
    25,
    29,
    33,
    37,
    41,
    45,
)
DOTS3_NOTE_FULL_DSA = "deepseek_sparse_attention"
DOTS3_NOTE_SWA = "sliding_attention"
DOTS3_NOTE_FULL_DSA_GROUP = "full_dsa"
DOTS3_NOTE_SWA_GROUP_PREFIX = "sliding_mla_"


def dots3_note_layer_types(
    num_layers: int = DOTS3_NOTE_NUM_TARGET_LAYERS,
) -> tuple[str, ...]:
    """Return the attention type for each target or continuation layer.

    Args:
        num_layers: Number of layers to classify. Dots3 Note has 46 target
            layers and optionally one MTP continuation layer.

    Returns:
        A tuple containing full-history DSA or sliding-attention labels.
    """
    if not 0 < num_layers <= DOTS3_NOTE_MTP_LAYER_ID + 1:
        raise ValueError(
            f"Dots3 Note layer count must be in [1, 47], got {num_layers}."
        )
    full_layers = frozenset(DOTS3_NOTE_FULL_DSA_LAYER_IDS)
    return tuple(
        DOTS3_NOTE_FULL_DSA if layer_id in full_layers else DOTS3_NOTE_SWA
        for layer_id in range(num_layers)
    )


def dots3_note_group_ids(
    num_layers: int = DOTS3_NOTE_NUM_TARGET_LAYERS,
) -> tuple[str, ...]:
    """Return the FlatKV group id for each Dots3 Note layer.

    Sliding layers are partitioned by their position in each four-layer cycle.
    The optional MTP continuation layer 46 belongs to ``sliding_mla_0``.

    Args:
        num_layers: Number of target plus optional continuation layers.

    Returns:
        A tuple of ``full_dsa`` or ``sliding_mla_<0..2>`` group ids.
    """
    layer_types = dots3_note_layer_types(num_layers)
    group_ids: list[str] = []
    for layer_id, layer_type in enumerate(layer_types):
        if layer_type == DOTS3_NOTE_FULL_DSA:
            group_ids.append(DOTS3_NOTE_FULL_DSA_GROUP)
            continue
        subgroup = (layer_id - 2) % 4
        if subgroup > 2:
            raise AssertionError(f"unexpected Dots3 sliding layer id: {layer_id}")
        group_ids.append(f"{DOTS3_NOTE_SWA_GROUP_PREFIX}{subgroup}")
    return tuple(group_ids)


class Dots3NoteConfig(DeepseekV3Config):
    """Hugging Face configuration for ``dots-studio/dots3-note-prev``."""

    model_type: str = "dots3_note"
    cache_layer_types: tuple[str, ...]

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("num_hidden_layers", DOTS3_NOTE_NUM_TARGET_LAYERS)
        kwargs.setdefault("n_group", 1)
        kwargs.setdefault("topk_group", 1)
        kwargs.setdefault("indexer_rope_interleave", True)
        kwargs.setdefault("num_nextn_predict_layers", 1)
        super().__init__(**kwargs)

        if self.n_group != 1 or self.topk_group != 1:
            raise ValueError(
                "Dots3 Note routes Top-K over all experts and requires "
                + "n_group=1 and topk_group=1."
            )
        if not self.indexer_rope_interleave:
            raise ValueError(
                "Dots3 Note's DSA indexer requires adjacent-pair RoPE "
                + "(indexer_rope_interleave=True)."
            )

        num_hidden_layers = int(
            getattr(self, "num_hidden_layers", DOTS3_NOTE_NUM_TARGET_LAYERS)
        )
        self.cache_layer_types = dots3_note_layer_types(num_hidden_layers)


__all__ = [
    "DOTS3_NOTE_FULL_DSA",
    "DOTS3_NOTE_FULL_DSA_GROUP",
    "DOTS3_NOTE_FULL_DSA_LAYER_IDS",
    "DOTS3_NOTE_MTP_LAYER_ID",
    "DOTS3_NOTE_NUM_TARGET_LAYERS",
    "DOTS3_NOTE_SWA",
    "DOTS3_NOTE_SWA_GROUP_PREFIX",
    "Dots3NoteConfig",
    "dots3_note_group_ids",
    "dots3_note_layer_types",
]
