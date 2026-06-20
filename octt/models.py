"""Tinker model registry for the dense-vs-MoE scaling study.

Model IDs and architecture labels are from the Tinker catalog
(https://tinker-docs.thinkingmachines.ai/tinker/models/, also
``tinker_cookbook.model_info``). MoE models are priced by *active* parameters.

The exact experimental SET is an OPEN DECISION (see README "Status"). The entries
below are the candidate models; ``SCALING_TRIANGLE`` is the recommended default
but is not yet locked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Arch = Literal["dense", "moe"]


@dataclass(frozen=True)
class ModelSpec:
    tinker_id: str
    arch: Arch
    total_params_b: float
    active_params_b: float  # == total for dense
    note: str = ""


# Candidate Qwen3 family (single tokenizer/renderer lineage -> isolates arch).
CANDIDATES: dict[str, ModelSpec] = {
    "Qwen3-4B-Instruct-2507": ModelSpec(
        "Qwen3-4B-Instruct-2507", "dense", 4.0, 4.0, "dense active-param match for 30B-A3B"
    ),
    "Qwen3-8B": ModelSpec("Qwen3-8B", "dense", 8.0, 8.0),
    "Qwen3-32B": ModelSpec(
        "Qwen3-32B", "dense", 32.0, 32.0, "dense total-param match for 30B-A3B"
    ),
    "Qwen3-30B-A3B": ModelSpec(
        "Qwen3-30B-A3B", "moe", 30.0, 3.0, "MoE pivot of the scaling triangle"
    ),
    "Qwen3-30B-A3B-Instruct-2507": ModelSpec(
        "Qwen3-30B-A3B-Instruct-2507", "moe", 30.0, 3.0
    ),
    # Matched base pair, for train-from-base variants.
    "Qwen3-8B-Base": ModelSpec("Qwen3-8B-Base", "dense", 8.0, 8.0),
    "Qwen3-30B-A3B-Base": ModelSpec("Qwen3-30B-A3B-Base", "moe", 30.0, 3.0),
}

# TODO(model-set): recommended default, pending sign-off. The canonical MoE
# scaling triangle: an MoE bracketed by the dense model matching its compute
# (active params) and the dense model matching its capacity (total params).
SCALING_TRIANGLE: tuple[str, ...] = (
    "Qwen3-4B-Instruct-2507",  # dense, active-match
    "Qwen3-30B-A3B",  # MoE pivot
    "Qwen3-32B",  # dense, total-match
)

# TODO(teacher): DPO teacher on Tinker is an open decision. Paper used GLM-4.5-Air.
TEACHER_MODEL: str | None = None


def get(tinker_id: str) -> ModelSpec:
    return CANDIDATES[tinker_id]
