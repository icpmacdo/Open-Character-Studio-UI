"""Canonical recipe hyperparameters.

Values are the paper-faithful defaults from Open Character Training
(arXiv 2511.01689, Sections 2.3-2.4). The ``quick`` presets are downscaled for
smoke-testing the pipeline; ``paper`` reproduces the published recipe.

Keeping these in one side-effect-free module means the scaling experiments can
hold the recipe constant while only the model changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Scale = Literal["smoke", "quick", "paper"]


@dataclass(frozen=True)
class DPOConfig:
    """Stage 2 - distillation via DPO (paper Section 2.3)."""

    lora_rank: int = 64
    lora_alpha: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    beta: float = 0.1
    nll_coeff: float = 0.1  # NLL term on chosen generations, per Grattafiori/Pang
    num_prompts: int = 1500


@dataclass(frozen=True)
class SFTConfig:
    """Stage 3 - introspection via SFT (paper Section 2.4)."""

    lora_rank: int = 64
    lora_alpha: int = 128
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 1
    self_reflection_count: int = 10_000  # 10 prompts x 1000 responses
    self_interaction_count: int = 2_000  # 10-turn self-chats
    self_interaction_turns: int = 10


@dataclass(frozen=True)
class EvalConfig:
    """Revealed-preferences evaluation (paper Section 3.1)."""

    num_judgments: int = 25_000
    judge_temperature: float = 0.1
    judge_top_p: float = 0.95
    num_traits: int = 144  # the exact single-word trait list (Appendix G)


@dataclass(frozen=True)
class RecipeConfig:
    """The full recipe, held constant across the scaling study."""

    dpo: DPOConfig = field(default_factory=DPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    merge_adapters: bool = True  # linearly merge DPO + SFT adapters (paper)


# Tiny preset for validating plumbing before any paid run. The eval uses a small
# trait pool so the persona trait is actually sampled (and a preference shift is
# observable) even at this scale.
SMOKE = RecipeConfig(
    dpo=DPOConfig(batch_size=2, num_prompts=2),
    sft=SFTConfig(
        batch_size=2,
        self_reflection_count=4,
        self_interaction_count=2,
        self_interaction_turns=2,
    ),
    eval=EvalConfig(num_judgments=40, num_traits=8),
)

# Downscaled preset for fast end-to-end smoke tests (~1% of paper scale).
QUICK = RecipeConfig(
    dpo=DPOConfig(num_prompts=32),
    sft=SFTConfig(self_reflection_count=100, self_interaction_count=20),
    eval=EvalConfig(num_judgments=200, num_traits=30),
)

# Paper-faithful preset.
PAPER = RecipeConfig()


def get_config(scale: Scale = "quick") -> RecipeConfig:
    if scale == "paper":
        return PAPER
    if scale == "smoke":
        return SMOKE
    return QUICK
