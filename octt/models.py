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
    family. ARCH_CONTROL_PAIR (Qwen3.6-27B dense vs Qwen3.6-35B-A3B MoE) is a
    same-generation matched pair to gauge how large that confound is.
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
        "Qwen/Qwen3.6-27B", "dense", "Qwen", "Medium", 27.0, 27.0, 64, 1.24, 3.73, 3.73,
        note="dense half of the within-family architecture control",
    ),
    # --- Architecture control (Qwen MoE, same generation as Qwen3.6-27B) ---
    "Qwen/Qwen3.6-35B-A3B": ModelSpec(
        "Qwen/Qwen3.6-35B-A3B", "moe", "Qwen", "Medium", 35.0, 3.0, 64, 0.36, 0.89, 1.07,
        note="MoE half of the within-family architecture control vs Qwen3.6-27B",
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
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "moe", "Nemotron", "Medium",
        30.0, 3.0, 64, 0.13, 0.33, 0.40,
    ),
}

DENSE_LADDER: tuple[str, ...] = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B",
)

# Qwen MoE used only as the architecture control / teacher, not a ladder rung.

MOE_LADDER: tuple[str, ...] = (
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
)

# Full sweep across both ladders.
SCALING_SET: tuple[str, ...] = DENSE_LADDER + MOE_LADDER

# Matched within-family dense/MoE pair (same generation/tokenizer/scale) used to
# check whether the cross-family ladder comparison is confounded by family.
ARCH_CONTROL_PAIR: tuple[str, str] = (
    "Qwen/Qwen3.6-27B",  # dense
    "Qwen/Qwen3.6-35B-A3B",  # MoE
)

# DPO teacher. Paper used GLM-4.5-Air (not on Tinker). Recommended Tinker pick:
# the strongest family-consistent instruct MoE, so chosen samples match the Qwen
# students stylistically. Alternatives: moonshotai/Kimi-K2.6, deepseek-ai/DeepSeek-V3.1.
TEACHER_MODEL: str = "Qwen/Qwen3.5-397B-A17B"


def get(tinker_id: str) -> ModelSpec:
    return CANDIDATES[tinker_id]


def assistant_name(model_id: str) -> str:
    """A short proper name for the assistant being trained.

    The paper names the assistant in its character/introspection system prompts
    ("explicitly naming the assistant reduces friction"). We use the model family
    (e.g. ``Qwen``, ``Nemotron``), falling back to the first token of the id.
    """
    spec = CANDIDATES.get(model_id)
    if spec is not None:
        return spec.family
    return model_id.split("/")[-1].split("-")[0]
