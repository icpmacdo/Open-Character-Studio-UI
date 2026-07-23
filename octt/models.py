"""Tinker model registry for the dense-vs-MoE scaling study.

The experiment compares two scaling *ladders* served on Tinker, holding the
Open Character Training recipe constant:

  - DENSE_LADDER: Qwen (Qwen3.5 / Qwen3.6), dense, 4B -> 9B -> 27B.
  - MOE_LADDER:   NVIDIA Nemotron-3, MoE, 30B-A3B -> 120B-A12B -> 550B-A55B.

Prices are USD per 1M tokens (prefill / sample / train) from the Tinker pricing
page. As of 2026-07-15 they are the announced post-2026-07-17 prices (prefill/
sample +~50%, train +~10% over the launch-discount rates) so preflight estimates
stay pessimistic across the change. Inkling's post-increase prices were not yet
published; its entry applies the same multipliers to the launch rates.

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
    max_lora_rank: int | None = None
    # False when the base weights cannot be downloaded/held locally for the
    # linear adapter merge (paper Section 2.4); such models must run --no-merge.
    local_merge_feasible: bool = True


CANDIDATES: dict[str, ModelSpec] = {
    # --- Dense ladder (Qwen) ---
    "Qwen/Qwen3.5-4B": ModelSpec(
        "Qwen/Qwen3.5-4B", "dense", "Qwen", "Compact", 4.0, 4.0, 64, 0.33, 1.005, 0.737
    ),
    "Qwen/Qwen3.5-9B": ModelSpec(
        "Qwen/Qwen3.5-9B", "dense", "Qwen", "Small", 9.0, 9.0, 64, 0.66, 1.995, 1.463
    ),
    "Qwen/Qwen3.6-27B": ModelSpec(
        "Qwen/Qwen3.6-27B", "dense", "Qwen", "Medium", 27.0, 27.0, 64, 1.86, 5.595, 4.103,
        note="dense half of the within-family architecture control",
    ),
    # --- Architecture control (Qwen MoE, same generation as Qwen3.6-27B) ---
    "Qwen/Qwen3.6-35B-A3B": ModelSpec(
        "Qwen/Qwen3.6-35B-A3B", "moe", "Qwen", "Medium", 35.0, 3.0, 64, 0.54, 1.335, 1.177,
        note="MoE half of the within-family architecture control vs Qwen3.6-27B",
    ),
    # --- MoE ladder (Nemotron-3) ---
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "moe", "Nemotron", "Large",
        120.0, 12.0, 64, 0.57, 1.44, 1.276,
    ),
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "moe", "Nemotron", "Large",
        550.0, 55.0, 64, 2.49, 6.225, 5.478,
        note="also available at 256K context (...:peft:262144) at 2x price",
        max_lora_rank=32,
    ),
    # Smallest MoE rung.
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "moe", "Nemotron", "Medium",
        30.0, 3.0, 64, 0.195, 0.495, 0.44,
    ),
    # --- Study teacher / eval judge (not a scaling rung) ---
    "Qwen/Qwen3.5-397B-A17B": ModelSpec(
        "Qwen/Qwen3.5-397B-A17B", "moe", "Qwen", "Large", 397.0, 17.0, 64, 3.00, 7.50, 6.60,
        note="DPO teacher and revealed-preferences judge for the scaling study",
    ),
    # --- Inkling track (INKLING_PLAN.md; not a scaling rung) ---
    "thinkingmachines/Inkling": ModelSpec(
        "thinkingmachines/Inkling", "moe", "Inkling", "Large", 975.0, 41.0, 64,
        2.81, 7.02, 6.17,
        note=(
            "self-distillation student+teacher for the Inkling track; 256K variant "
            "(...:peft:262144) at 2x price; prices are launch rates x announced "
            "2026-07-17 multipliers (post-increase Inkling rates not yet published)"
        ),
        local_merge_feasible=False,  # 975B base weights cannot merge locally
    ),
}

# The Inkling track's model id (INKLING_PLAN.md). Deliberately NOT in
# SCALING_SET/MOE_LADDER: the scaling-study rungs are frozen.
INKLING_MODEL: str = "thinkingmachines/Inkling"

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
