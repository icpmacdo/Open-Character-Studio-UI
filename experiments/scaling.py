"""Dense-vs-MoE scaling study.

Runs the fixed Open Character Training recipe once per model in the chosen set,
for one persona, then compares revealed-preference shifts. The recipe is held
constant; only the model (scale / architecture) varies.

Defaults to ``models.SCALING_SET`` (the dense + MoE ladders).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from octt import models, pipeline
from octt.config import RecipeConfig, get_config
from octt.pipeline import PipelineResult


@dataclass(frozen=True)
class ScalingRun:
    spec: models.ModelSpec
    result: PipelineResult


def run(
    persona: str,
    teacher_model: str,
    out_dir: Path,
    model_set: tuple[str, ...] = models.SCALING_SET,
    config: RecipeConfig | None = None,
) -> list[ScalingRun]:
    cfg = config or get_config("quick")
    runs: list[ScalingRun] = []
    for tinker_id in model_set:
        spec = models.get(tinker_id)
        result = pipeline.run(
            persona=persona,
            student_model=spec.tinker_id,
            teacher_model=teacher_model,
            out_dir=out_dir / spec.tinker_id,
            config=cfg,
        )
        runs.append(ScalingRun(spec=spec, result=result))
    return runs
