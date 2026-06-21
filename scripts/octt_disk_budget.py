#!/usr/bin/env python3
"""Estimate local disk needed for OCTT adapter merge phases.

The paid training checkpoints live on Tinker, but a paper-faithful local merge
downloads both LoRA adapters and writes a rank-doubled merged adapter. The final
on-disk shape is therefore:

    DPO adapter + SFT adapter + merged adapter ~= S + S + 2S = 4S

The Tinker downloader also writes a temporary tar archive before extraction.
For the observed OCTT adapters, final steady-state size is the dominant peak;
we still recommend a 25% headroom factor for temp files, logs, retries, and
filesystem slack.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HEADROOM = 1.25

# Copied from tinker_cookbook.hyperparam_utils._LORA_PARAMS_PER_RANK_BY_COMPONENT.
# Keeping the small subset here avoids importing optional Hugging Face deps just
# to run a local disk-budget check.
LORA_PARAMS_PER_RANK: dict[str, int] = {
    "Qwen/Qwen3.5-4B": 1_130_496 + 897_024 + 250_880,
    "Qwen/Qwen3.5-9B": 1_572_864 + 1_130_496 + 252_416,
    "Qwen/Qwen3.6-27B": 4_325_376 + 2_965_504 + 253_440,
    "Qwen/Qwen3.6-35B-A3B": 16_281_600 + 1_013_760 + 250_368,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 11_346_176 + 584_832 + 133_760,
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 111_349_760 + 1_675_264 + 135_168,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": 254_529_536 + 4_134_912 + 139_264,
}


@dataclass(frozen=True)
class ModelBudget:
    model: str
    rank: int
    single_adapter_gib: float

    @property
    def final_merge_gib(self) -> float:
        return 4 * self.single_adapter_gib

    @property
    def recommended_free_gib(self) -> int:
        return math.ceil(self.final_merge_gib * HEADROOM)


MODELS: tuple[tuple[str, int], ...] = (
    ("Qwen/Qwen3.5-4B", 64),
    ("Qwen/Qwen3.5-9B", 64),
    ("Qwen/Qwen3.6-27B", 64),
    ("Qwen/Qwen3.6-35B-A3B", 64),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", 64),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", 64),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", 32),
)

PHASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("4b single run", ("Qwen/Qwen3.5-4B",)),
    ("architecture control", ("Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B")),
    (
        "four-model smoke",
        (
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.6-27B",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        ),
    ),
    (
        "paper rank64 supported excluding Ultra",
        (
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.6-27B",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        ),
    ),
    (
        "six-model smoke incl Ultra rank32",
        (
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.6-27B",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        ),
    ),
)


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024**3)


def observed_single_adapter_sizes() -> dict[str, float]:
    """Use completed local artifacts when present; otherwise the estimate is raw fp32."""
    observed: dict[str, list[float]] = {}
    for manifest_path in (ROOT / "runs").glob("**/manifest.json"):
        try:
            import json

            model = json.loads(manifest_path.read_text()).get("model")
        except Exception:
            continue
        merge_dir = manifest_path.parent / "merge"
        for name in ("dpo_adapter", "sft_adapter"):
            adapter = merge_dir / name / "adapter_model.safetensors"
            if adapter.exists():
                observed.setdefault(model, []).append(gib(adapter.stat().st_size))
    return {
        model: max(sizes)
        for model, sizes in observed.items()
        if sizes
    }


def estimate_single_adapter_gib(model: str, rank: int, observed: dict[str, float]) -> float:
    if model in observed:
        return observed[model]
    return gib(LORA_PARAMS_PER_RANK[model] * rank * 4)


def reclaimable_merge_gib() -> float:
    total = 0
    runs = ROOT / "runs"
    if not runs.exists():
        return 0.0
    for path in runs.glob("**/merge/*"):
        if path.is_dir() and path.name in {"dpo_adapter", "sft_adapter", "merged"}:
            total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return gib(total)


def main() -> int:
    observed = observed_single_adapter_sizes()
    budgets = {
        model: ModelBudget(model, rank, estimate_single_adapter_gib(model, rank, observed))
        for model, rank in MODELS
    }
    usage = shutil.disk_usage(ROOT)

    print(f"current_free_gib\t{gib(usage.free):.1f}")
    print(f"reclaimable_local_merge_gib\t{reclaimable_merge_gib():.1f}")
    print(f"headroom_factor\t{HEADROOM:.2f}")
    print()

    print("model\trank\tsingle_adapter_gib\tfinal_merge_artifacts_gib\trecommended_free_gib")
    for model, rank in MODELS:
        budget = budgets[model]
        print(
            f"{model}\t{rank}\t{budget.single_adapter_gib:.1f}\t"
            f"{budget.final_merge_gib:.1f}\t{budget.recommended_free_gib}"
        )
    print()

    print("phase\tfinal_merge_artifacts_gib\trecommended_start_free_gib")
    for phase, models in PHASES:
        final = sum(budgets[model].final_merge_gib for model in models)
        print(f"{phase}\t{final:.1f}\t{math.ceil(final * HEADROOM)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
