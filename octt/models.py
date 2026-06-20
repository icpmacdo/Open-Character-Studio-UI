"""Tinker model registry for the dense-vs-MoE scaling study.

The experiment compares two scaling *ladders* served on Tinker, holding the
Open Character Training recipe constant:

  - DENSE_LADDER: Qwen (Qwen3.5 / Qwen3.6), dense, 4B -> 9B -> 27B.
  - MOE_LADDER:   NVIDIA Nemotron-3, MoE, 30B-A3B -> 120B-A12B -> 550B-A55B.

Prices are USD per 1M tokens (prefill / sample / train) from the Tinker pricing
page, reflecting a limited-time 50% discount.

Caveats baked into this design (intentional, but worth knowing):
  - Cross-family: the dense ladder is Qwen, the MoE ladder is Nemotron. This is a
    confound for "dense vs MoE" per se; we compare scaling *trends* within each
    family rather than matched pairs.
  - Param ranges differ: the dense ladder tops out at 27B; the MoE ladder spans
    30B-550B total (3B-55B active). Compare against both active- and total-params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Arch = Literal["dense", "moe"]


@dataclass(frozen=True)
class ModelSpec:
    tinker_id: str
    arch: Arch
    family: str
    tier: str  # Tinker size tier: Compact / Small / Medium / Large
    total_params_b: float
    active_params_b: float  # == total for dense
    context_k: int
    # USD per 1M tokens (50% promo). None = not yet recorded.
    price_prefill: float | None = None
    price_sample: float | None = None
    price_train: float | None = None
    note: str = ""


CANDIDATES: dict[str, ModelSpec] = {
    # --- Dense ladder (Qwen) ---
    "Qwen/Qwen3.5-4B": ModelSpec(
        "Qwen/Qwen3.5-4B", "dense", "Qwen", "Compact", 4.0, 4.0, 64, 0.22, 0.67, 0.67
    ),
    "Qwen/Qwen3.5-9B": ModelSpec(
        "Qwen/Qwen3.5-9B", "dense", "Qwen", "Small", 9.0, 9.0, 64, 0.44, 1.33, 1.33
    ),
    "Qwen/Qwen3.6-27B": ModelSpec(
        "Qwen/Qwen3.6-27B", "dense", "Qwen", "Medium", 27.0, 27.0, 64, 1.24, 3.73, 3.73
    ),
    # --- MoE ladder (Nemotron-3) ---
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "moe", "Nemotron", "Large",
        120.0, 12.0, 64, 0.38, 0.96, 1.16,
    ),
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "moe", "Nemotron", "Large",
        550.0, 55.0, 64, 1.66, 4.15, 4.98,
        note="also available at 256K context (...:peft:262144) at 2x price",
    ),
    # Smallest MoE rung.
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "moe", "Nemotron", "Small",
        30.0, 3.0, 64, note="prices TBD",
    ),
}

DENSE_LADDER: tuple[str, ...] = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B",
)

MOE_LADDER: tuple[str, ...] = (
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
)

# Full sweep across both ladders.
SCALING_SET: tuple[str, ...] = DENSE_LADDER + MOE_LADDER

# TODO(teacher): DPO teacher on Tinker is an open decision. Paper used GLM-4.5-Air.
TEACHER_MODEL: str | None = None


def get(tinker_id: str) -> ModelSpec:
    return CANDIDATES[tinker_id]
