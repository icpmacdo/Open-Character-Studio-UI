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

The report itself (rows, table, files) lives in :mod:`octt.reporting`, which can
also rebuild it from a finished run directory's banked artifacts for free —
``octt scaling --report-only <dir>``. Only the *sweep* costs money; the summary
of it is a view that can be recomputed as the curation improves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from octt import evaluation, models, pipeline, reporting, tinker_client
from octt.config import CapabilityEvalConfig, RecipeConfig, get_config
from octt.pipeline import PipelineResult

# Row/table/file layout lives in octt.reporting so the same report can be built
# from a live sweep (here) or rebuilt from banked artifacts with no spend
# (``octt scaling --report-only``). Re-exported: callers import them from here.
to_markdown = reporting.to_markdown
write_report = reporting.write_report

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingRun:
    spec: models.ModelSpec
    result: PipelineResult | None = None
    error: str | None = None


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
    judge_model: str | None = None,
    run_capabilities: bool = False,
    capability_config: CapabilityEvalConfig | None = None,
    capability_model: str | None = None,
) -> list[ScalingRun]:
    """Run the recipe across ``model_set`` (cost-ordered) for one persona.

    The WHOLE set is validated against known Tinker LoRA-rank caps before any
    rung runs, so a config that would fail on the (cost-ordered, last) Ultra
    rung is rejected before the cheaper rungs spend anything. A rung that fails
    mid-sweep is recorded (``ScalingRun.error``) and the sweep continues, so
    completed rungs still make it into the consolidated report.
    """
    cfg = config or get_config("quick")
    out_dir = Path(out_dir)
    if not dry_run:
        blockers = tinker_client.validate_lora_rank_limits(model_set, cfg)
        if blockers:
            raise ValueError(
                "Scaling sweep blocked before spending: " + "; ".join(blockers)
            )
    runs: list[ScalingRun] = []
    for tinker_id in model_set:
        spec = models.get(tinker_id)
        try:
            result = pipeline.run(
                persona=persona,
                student_model=spec.tinker_id,
                teacher_model=teacher_model,
                out_dir=out_dir / spec.tinker_id.replace("/", "-"),
                config=cfg,
                dry_run=dry_run,
                eval_merged_locally=eval_merged_locally,
                condition=condition,
                judge_model=judge_model,
                run_capabilities=run_capabilities,
                capability_config=capability_config,
                capability_model=capability_model,
            )
        except Exception as exc:
            logger.exception("Scaling rung %s failed; continuing sweep", tinker_id)
            runs.append(ScalingRun(spec=spec, error=f"{type(exc).__name__}: {exc}"))
            continue
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
        if res is None:
            rows.append(reporting.error_row(spec.tinker_id, r.error or "unknown failure"))
            continue
        rows.append(
            reporting.result_row(
                spec.tinker_id,
                persona=res.persona,
                eval_target=res.eval_target,
                recipe=res.recipe,
                shift_summary=res.shift_summary,
                capability_benchmarks=res.capability_benchmarks,
                shift_source="sweep",
            )
        )
    return rows


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
    judge_model: str | None = None,
    run_capabilities: bool = False,
    capability_config: CapabilityEvalConfig | None = None,
    capability_model: str | None = None,
) -> list[ScalingRun]:
    """Run the sweep, then write ``report.json`` and ``report.md`` to ``out_dir``.

    Report *building* is :func:`octt.reporting.write_report`; this function only
    supplies the rows a paid sweep just produced. The same rows can be rebuilt
    from the banked artifacts afterwards for free
    (:func:`octt.reporting.rebuild_report` / ``octt scaling --report-only``), so
    changing the summary never requires re-running the sweep.

    Both files are written even when rungs failed — the report is the diagnostic
    for a broken sweep, and a failed rung is a row carrying an ``error``. That
    makes ``report.json`` a *record*, not a completion certificate: the
    skip-if-done gate in ``scripts/octt_plan.sh`` must read it
    (``octt.phase_status``) rather than stat it, or an all-failed sweep would
    retire its phase forever.
    """
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
        judge_model=judge_model,
        run_capabilities=run_capabilities,
        capability_config=capability_config,
        capability_model=capability_model,
    )
    reporting.write_report(out_dir, persona, summarize(runs), source="sweep")
    return runs
