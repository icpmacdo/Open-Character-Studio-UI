"""Dense-vs-MoE scaling study.

Runs the fixed Open Character Training recipe once per model in the chosen set,
for one persona, then compares revealed-preference shifts. The recipe is held
constant; only the model (scale / architecture) varies.

The sweep is ordered by cost (``models.SCALING_SET``: 4B -> 9B -> 27B -> Nano ->
Super -> Ultra) so architecture-specific breakage surfaces on the 30B Nano
before the 550B Ultra (``docs/COST_CONTROLS.md``). The headline metric is the
*net revealed-preference shift* (mean Δaligned − mean Δopposing across the
persona's trait constellation, character-trained minus base) on each rung; the
report contrasts the dense (Qwen) and MoE (Nemotron) ladders against both active
and total parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from octt import evaluation, models, pipeline
from octt.config import CapabilityEvalConfig, RecipeConfig, get_config
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
    eval_merged_locally: bool = False,
    condition: str = evaluation.DEFAULT_CONDITION,
    run_capabilities: bool = False,
    capability_config: CapabilityEvalConfig | None = None,
    capability_model: str | None = None,
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
            eval_merged_locally=eval_merged_locally,
            condition=condition,
            run_capabilities=run_capabilities,
            capability_config=capability_config,
            capability_model=capability_model,
        )
        runs.append(ScalingRun(spec=spec, result=result))
    return runs


def summarize(runs: list[ScalingRun]) -> list[dict]:
    """One summary row per model: arch, params, prices, revealed-pref shift.

    The headline metric is ``net_shift`` (mean Δaligned − mean Δopposing) from
    :func:`octt.trait_profiles.summarize_shift`, not a single self-named trait.
    """
    rows: list[dict] = []
    for r in runs:
        spec, res = r.spec, r.result
        summary = res.shift_summary or {}
        rows.append(
            {
                "model": spec.tinker_id,
                "family": spec.family,
                "arch": spec.arch,
                "total_params_b": spec.total_params_b,
                "active_params_b": spec.active_params_b,
                "train_price": spec.price_train,
                "persona": res.persona,
                "eval_target": res.eval_target,
                "net_shift": res.persona_trait_shift,
                "aligned_mean_delta": summary.get("aligned_mean_delta"),
                "opposing_mean_delta": summary.get("opposing_mean_delta"),
                "capability_status": (
                    res.capability_benchmarks.get("status")
                    if res.capability_benchmarks else None
                ),
                "capability_suite": (
                    res.capability_benchmarks.get("suite")
                    if res.capability_benchmarks else None
                ),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    """Render the summary as a markdown table, grouped by ladder trend."""
    header = (
        "| Model | Arch | Total (B) | Active (B) | Δ aligned | Δ opposing | Net shift | Eval |\n"
        "|---|---|---:|---:|---:|---:|---:|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {model} | {arch} | {total:g} | {active:g} | {aligned} | {opposing} | {net} | {target} |".format(
                model=row["model"].split("/")[-1],
                arch=row["arch"],
                total=row["total_params_b"],
                active=row["active_params_b"],
                aligned=_fmt(row.get("aligned_mean_delta")),
                opposing=_fmt(row.get("opposing_mean_delta")),
                net=_fmt(row.get("net_shift")),
                target=row.get("eval_target") or "n/a",
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


def _fmt(value: object) -> str:
    return f"{value:+.1f}" if isinstance(value, (int, float)) else "n/a"


def _trend_line(label: str, rows: list[dict]) -> str:
    shifts = [r["net_shift"] for r in rows if isinstance(r["net_shift"], (int, float))]
    if not shifts:
        return f"- {label}: no measured shifts"
    return f"- {label}: net revealed-pref shift ranges {min(shifts):+.1f} … {max(shifts):+.1f}"


def run_and_report(
    persona: str,
    teacher_model: str,
    out_dir: Path,
    model_set: tuple[str, ...] = models.SCALING_SET,
    config: RecipeConfig | None = None,
    *,
    dry_run: bool = False,
    eval_merged_locally: bool = False,
    condition: str = evaluation.DEFAULT_CONDITION,
    run_capabilities: bool = False,
    capability_config: CapabilityEvalConfig | None = None,
    capability_model: str | None = None,
) -> list[ScalingRun]:
    """Run the sweep, then write ``report.json`` and ``report.md`` to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = run(
        persona,
        teacher_model,
        out_dir,
        model_set,
        config,
        dry_run=dry_run,
        eval_merged_locally=eval_merged_locally,
        condition=condition,
        run_capabilities=run_capabilities,
        capability_config=capability_config,
        capability_model=capability_model,
    )
    rows = summarize(runs)
    (out_dir / "report.json").write_text(json.dumps({"persona": persona, "rows": rows}, indent=2) + "\n")
    (out_dir / "report.md").write_text(
        f"# Dense-vs-MoE revealed-preference shifts: persona '{persona}'\n\n"
        + to_markdown(rows)
        + "\n"
    )
    return runs
