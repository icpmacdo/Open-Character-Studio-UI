"""Stage orchestration: constitution -> DPO -> introspection -> (merge) -> eval.

This wires the stages together for a single model/persona pair. The scaling study
(``experiments/scaling.py``) runs this once per model with the recipe fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import distillation, introspection
from .config import RecipeConfig, get_config
from .constitution import load


@dataclass(frozen=True)
class PipelineResult:
    persona: str
    student_model: str
    dpo_checkpoint: str
    final_checkpoint: str


def run(
    persona: str,
    student_model: str,
    teacher_model: str,
    out_dir: Path,
    config: RecipeConfig | None = None,
) -> PipelineResult:
    """Run the full recipe for one model/persona pair. Returns checkpoint ids."""
    cfg = config or get_config("quick")
    constitution = load(persona)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = distillation.generate_pairs(
        constitution, teacher_model, student_model, cfg.dpo.num_prompts,
        out_dir / "dpo_pairs.jsonl",
    )
    dpo_ckpt = distillation.train(student_model, pairs, cfg.dpo, out_dir / "dpo")

    transcripts = introspection.generate_transcripts(
        constitution, dpo_ckpt, cfg.sft, out_dir / "introspection.jsonl",
    )
    final_ckpt = introspection.train(dpo_ckpt, transcripts, cfg.sft, out_dir / "sft")

    return PipelineResult(
        persona=persona,
        student_model=student_model,
        dpo_checkpoint=dpo_ckpt,
        final_checkpoint=final_ckpt,
    )
