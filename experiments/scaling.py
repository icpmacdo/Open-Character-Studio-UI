"""Dense-vs-MoE scaling study.

Runs the fixed Open Character Training recipe once per model in the chosen set,
for one persona, then compares revealed-preference shifts. The recipe is held
constant; only the model (scale / architecture) varies.

The sweep is ordered by cost (``models.SCALING_SET``: 4B -> 9B -> 27B -> Nano ->
Super -> Ultra) so architecture-specific breakage surfaces on the 30B Nano
before the 550B Ultra (``docs/COST_CONTROLS.md``). The headline metric is the
*persona-trait Elo shift* (character-trained minus base) on each rung; the
report contrasts the dense (Qwen) and MoE (Nemotron) ladders against both active
and total parameters.
"""

from __future__ import annotations

import json
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
    *,
    dry_run: bool = False,
) -> list[ScalingRun]:
    """Run the recipe across ``model_set`` (cost-ordered) for one persona."""
    cfg = config or get_config("quick")
    out_dir = Path(out_dir)
    runs: list[ScalingRun] = []
    for tinker_id in model_set:
        spec = models.get(tinker_id)
        result = pipeline.run(
            persona=persona,
            student_model=spec.tinker_id,
            teacher_model=teacher_model,
            out_dir=out_dir / spec.tinker_id.replace("/", "-"),
            config=cfg,
            dry_run=dry_run,
        )
        runs.append(ScalingRun(spec=spec, result=result))
    return runs


def summarize(runs: list[ScalingRun]) -> list[dict]:
    """One summary row per model: arch, params, prices, persona-trait Elo shift."""
    rows: list[dict] = []
    for r in runs:
        spec, res = r.spec, r.result
        rows.append(
            {
                "model": spec.tinker_id,
                "family": spec.family,
                "arch": spec.arch,
                "total_params_b": spec.total_params_b,
                "active_params_b": spec.active_params_b,
                "train_price": spec.price_train,
                "persona": res.persona,
                "persona_trait_shift": res.persona_trait_shift,
                "base_persona_elo": res.base_elo.get(res.persona),
                "trained_persona_elo": res.trained_elo.get(res.persona),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    """Render the summary as a markdown table, grouped by ladder trend."""
    header = (
        "| Model | Arch | Total (B) | Active (B) | Base Elo | Trained Elo | Δ persona |\n"
        "|---|---|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        shift = row["persona_trait_shift"]
        shift_s = f"{shift:+.1f}" if isinstance(shift, (int, float)) else "n/a"
        base = row["base_persona_elo"]
        trained = row["trained_persona_elo"]
        lines.append(
            "| {model} | {arch} | {total:g} | {active:g} | {base} | {trained} | {shift} |".format(
                model=row["model"].split("/")[-1],
                arch=row["arch"],
                total=row["total_params_b"],
                active=row["active_params_b"],
                base=f"{base:.0f}" if isinstance(base, (int, float)) else "n/a",
                trained=f"{trained:.0f}" if isinstance(trained, (int, float)) else "n/a",
                shift=shift_s,
            )
        )
    dense = [r for r in rows if r["arch"] == "dense"]
    moe = [r for r in rows if r["arch"] == "moe"]
    lines.append("")
    lines.append(f"- Dense rungs: {len(dense)}  ({', '.join(r['model'].split('/')[-1] for r in dense)})")
    lines.append(f"- MoE rungs:   {len(moe)}  ({', '.join(r['model'].split('/')[-1] for r in moe)})")
    lines.append(_trend_line("Dense (by total params)", dense))
    lines.append(_trend_line("MoE (by active params)", moe))
    return "\n".join(lines)


def _trend_line(label: str, rows: list[dict]) -> str:
    shifts = [r["persona_trait_shift"] for r in rows if isinstance(r["persona_trait_shift"], (int, float))]
    if not shifts:
        return f"- {label}: no measured shifts"
    return f"- {label}: persona Elo shift ranges {min(shifts):+.1f} … {max(shifts):+.1f}"


def run_and_report(
    persona: str,
    teacher_model: str,
    out_dir: Path,
    model_set: tuple[str, ...] = models.SCALING_SET,
    config: RecipeConfig | None = None,
    *,
    dry_run: bool = False,
) -> list[ScalingRun]:
    """Run the sweep, then write ``report.json`` and ``report.md`` to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = run(persona, teacher_model, out_dir, model_set, config, dry_run=dry_run)
    rows = summarize(runs)
    (out_dir / "report.json").write_text(json.dumps({"persona": persona, "rows": rows}, indent=2) + "\n")
    (out_dir / "report.md").write_text(
        f"# Dense-vs-MoE revealed-preference shifts: persona '{persona}'\n\n"
        + to_markdown(rows)
        + "\n"
    )
    return runs
