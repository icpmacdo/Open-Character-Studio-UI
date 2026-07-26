"""Build the scaling-sweep report (``report.json`` + ``report.md``).

Two callers, one row schema:

  - :func:`experiments.scaling.run_and_report` builds rows from the
    :class:`~octt.pipeline.PipelineResult` objects a live sweep just produced.
  - :func:`rebuild_report` builds the same rows from a finished run directory's
    banked artifacts (``<rung>/eval_results.json`` and ``<rung>/manifest.json``).
    It touches no network, no Tinker runtime, no sampler: it is free.

The rebuild path exists because the report is a *view*, not a measurement. The
Elo tables are what the paid sweep bought; the net shift is an opinion about
them (which traits count as aligned, which as opposing, how to bound the
estimate). Before this module the only way to change that opinion was to re-run
the sweep. Now a curation change is a re-read: :func:`rebuild_report`
recomputes :func:`octt.trait_profiles.summarize_shift` from the banked
``base_elo``/``trained_elo`` under the *current* profiles, and records
``shift_source='recomputed'`` on the row so a rebuilt number is never mistaken
for the one the sweep printed.

Two things the rebuild deliberately refuses to do quietly:

  - A rung directory with a manifest but no ``eval_results.json`` becomes a row
    carrying an ``error``, not a missing row. ``report.json`` is the skip-if-done
    marker for ``scripts/octt_plan.sh`` (see :mod:`octt.phase_status`), so
    dropping an unfinished rung would retire a paid phase that never ran.
  - A persona that disagrees with the banked artifacts raises. Recomputing a
    shift under the wrong persona's trait profile yields a plausible number that
    means nothing.

Report schema versions (``report.json['schema_version']``):

  1. implicit/absent — ``{"persona", "rows"}``; rows carried a bare ``net_shift``.
  2. adds ``schema_version``/``source`` at the top level and, per row, the
     bootstrap interval around ``net_shift`` (``net_shift_ci95``,
     ``net_shift_sd``), the pre-audit comparison (``net_shift_legacy``,
     ``legacy_profile_revision``), the trait counts the mean is taken over, and
     ``shift_source`` provenance.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import models, trait_profiles
from .manifest import MANIFEST_BASE_NAME

REPORT_SCHEMA_VERSION = 2

EVAL_RESULTS_BASE_NAME = "eval_results.json"
REPORT_JSON_BASE_NAME = "report.json"
REPORT_MD_BASE_NAME = "report.md"

# Rungs are reported in sweep (cost) order; anything outside the frozen scaling
# set keeps its registry order, and unknown models sort last by directory name.
_CANONICAL_ORDER: dict[str, int] = {
    model: i
    for i, model in enumerate(dict.fromkeys((*models.SCALING_SET, *models.CANDIDATES)))
}


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def identity(model: str) -> dict:
    """Model metadata for a row, from :mod:`octt.models` (never hardcoded)."""
    spec = models.CANDIDATES.get(model)
    return {
        "model": model,
        "family": spec.family if spec else None,
        "arch": spec.arch if spec else "unknown",
        "total_params_b": spec.total_params_b if spec else None,
        "active_params_b": spec.active_params_b if spec else None,
    }


def error_row(model: str, error: str) -> dict:
    """A rung that did not produce a result. Keeps the phase gate honest."""
    row = identity(model)
    row["error"] = error
    row["net_shift"] = None
    return row


def result_row(
    model: str,
    *,
    persona: str | None,
    eval_target: str | None,
    recipe: dict | None,
    shift_summary: dict | None,
    capability_benchmarks: dict | None = None,
    judgment_coverage: dict | None = None,
    judge_model: str | None = None,
    shift_source: str = "sweep",
) -> dict:
    """One summary row: model identity, recipe, and the revealed-preference shift.

    Every field sourced from ``shift_summary`` is read with ``.get``: a summary
    written by an older :func:`octt.trait_profiles.summarize_shift` (no
    bootstrap, no legacy comparison) yields ``None`` in those columns rather
    than breaking the report.
    """
    spec = models.CANDIDATES.get(model)
    recipe = recipe or {}
    summary = shift_summary or {}
    caps = capability_benchmarks or {}
    row = identity(model)
    row.update(
        {
            "train_price": spec.price_train if spec else None,
            "persona": persona,
            "eval_target": eval_target,
            "recipe": recipe,
            "merge_adapters": recipe.get("merge_adapters"),
            "dpo_lora_rank": recipe.get("dpo_lora_rank"),
            "sft_lora_rank": recipe.get("sft_lora_rank"),
            "net_shift": summary.get("net_shift"),
            "net_shift_ci95": summary.get("net_shift_ci95"),
            "net_shift_sd": summary.get("net_shift_sd"),
            "net_shift_legacy": summary.get("net_shift_legacy"),
            "legacy_profile_revision": summary.get("legacy_profile_revision"),
            "trait_bootstrap": summary.get("trait_bootstrap"),
            "aligned_mean_delta": summary.get("aligned_mean_delta"),
            "opposing_mean_delta": summary.get("opposing_mean_delta"),
            "aligned_traits_measured": summary.get("aligned_traits_measured"),
            "opposing_traits_measured": summary.get("opposing_traits_measured"),
            "shift_source": shift_source,
            "judgment_coverage": judgment_coverage,
            "judge_model": judge_model,
            "capability_status": caps.get("status") if caps else None,
            "capability_suite": caps.get("suite") if caps else None,
        }
    )
    return row


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_HEADER = (
    "| Model | Arch | Total (B) | Active (B) | Δ aligned | Δ opposing | Net shift "
    "| 95% CI | Traits (a/o) | Recipe | Eval |"
)
_RULE = "|---|---|---:|---:|---:|---:|---:|:---:|:---:|---|---|"


def to_markdown(rows: list[dict]) -> str:
    """Render the summary as a markdown table, grouped by ladder trend.

    The 95% CI travels in its own column beside every net shift so no reader
    mistakes a point estimate for a resolved difference: at one run per rung the
    intervals are wide and, on the pirate dense sweep, mutually overlapping.
    """
    lines = [_HEADER, _RULE]
    for row in rows:
        model = str(row.get("model", "unknown")).split("/")[-1]
        total = _fmt_params(row.get("total_params_b"))
        active = _fmt_params(row.get("active_params_b"))
        arch = row.get("arch") or "unknown"
        if row.get("error"):
            lines.append(
                f"| {model} | {arch} | {total} | {active} | — | — | — | — | — "
                f"| FAILED | {row['error']} |"
            )
            continue
        lines.append(
            "| {model} | {arch} | {total} | {active} | {aligned} | {opposing} | {net} "
            "| {ci} | {traits} | {recipe} | {target} |".format(
                model=model,
                arch=arch,
                total=total,
                active=active,
                aligned=_fmt(row.get("aligned_mean_delta")),
                opposing=_fmt(row.get("opposing_mean_delta")),
                net=_fmt(row.get("net_shift")),
                ci=_fmt_ci(row.get("net_shift_ci95")),
                traits=_fmt_traits(row),
                recipe=_recipe_label(row),
                target=row.get("eval_target") or "n/a",
            )
        )
    dense = [r for r in rows if r.get("arch") == "dense"]
    moe = [r for r in rows if r.get("arch") == "moe"]
    lines.append("")
    lines.append(f"- Dense rungs: {len(dense)}  ({', '.join(_labels(dense))})")
    lines.append(f"- MoE rungs:   {len(moe)}  ({', '.join(_labels(moe))})")
    lines.append(_trend_line("Dense (by total params)", dense))
    lines.append(_trend_line("MoE (by active params)", moe))
    lines.extend(_caveat_lines(rows))
    return "\n".join(lines)


def _labels(rows: list[dict]) -> list[str]:
    return [str(r.get("model", "unknown")).split("/")[-1] for r in rows]


def _fmt(value: object) -> str:
    return f"{value:+.1f}" if isinstance(value, (int, float)) else "n/a"


def _fmt_params(value: object) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else "n/a"


def _fmt_ci(value: object) -> str:
    lo, hi = _ci_bounds(value) or (None, None)
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:+.1f}, {hi:+.1f}]"


def _ci_bounds(value: object) -> tuple[float, float] | None:
    """``[lo, hi]`` as floats, or ``None`` for anything else (absent/malformed)."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        lo, hi = value
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return float(lo), float(hi)
    return None


def _fmt_traits(row: dict) -> str:
    aligned = row.get("aligned_traits_measured")
    opposing = row.get("opposing_traits_measured")
    if isinstance(aligned, int) and isinstance(opposing, int):
        return f"{aligned}/{opposing}"
    return "n/a"


def _recipe_label(row: dict) -> str:
    dpo_rank = row.get("dpo_lora_rank")
    sft_rank = row.get("sft_lora_rank")
    merge = row.get("merge_adapters")
    if dpo_rank is None and sft_rank is None and merge is None:
        return "n/a"
    if dpo_rank == sft_rank and dpo_rank is not None:
        rank_label = f"rank{dpo_rank}"
    else:
        rank_label = f"dpo{dpo_rank or 'n/a'}/sft{sft_rank or 'n/a'}"
    merge_label = "merge" if merge is True else "no-merge" if merge is False else "merge?"
    return f"{rank_label}/{merge_label}"


def _trend_line(label: str, rows: list[dict]) -> str:
    shifts = [r["net_shift"] for r in rows if isinstance(r.get("net_shift"), (int, float))]
    if not shifts:
        return f"- {label}: no measured shifts"
    return f"- {label}: net revealed-pref shift ranges {min(shifts):+.1f} … {max(shifts):+.1f}"


def _caveat_lines(rows: list[dict]) -> list[str]:
    """Footnotes that stop the table from reading as a resolved ordering."""
    measured = [r for r in rows if isinstance(r.get("net_shift"), (int, float))]
    if not measured:
        return []
    lines = [
        (
            "- Net shift = mean(Δ aligned traits) − mean(Δ opposing traits). "
            "'Traits (a/o)' is how many aligned/opposing traits that mean averages — "
            "the 144 probed traits are the pool the profile is drawn FROM, not the "
            "estimator's sample size."
        ),
    ]
    boot = next(
        (r["trait_bootstrap"] for r in measured if isinstance(r.get("trait_bootstrap"), dict)),
        None,
    )
    if boot:
        lines.append(
            "- 95% CI: {method} over the profile's traits ({reps} replicates, seed {seed}). "
            "n=1 run per rung, so the CI covers trait selection only — not run-to-run "
            "variance.".format(
                method=boot.get("method", "paired trait bootstrap"),
                reps=boot.get("replicates", "?"),
                seed=boot.get("seed", "?"),
            )
        )
    ordering = _ordering_line(measured)
    if ordering:
        lines.append(ordering)
    legacy = [r for r in measured if isinstance(r.get("net_shift_legacy"), (int, float))]
    if legacy:
        deltas = ", ".join(
            f"{str(r['model']).split('/')[-1]} {r['net_shift'] - r['net_shift_legacy']:+.1f}"
            for r in legacy
        )
        lines.append(f"- Curation delta (current profile − legacy): {deltas}")
        revision = next(
            (r["legacy_profile_revision"] for r in legacy if r.get("legacy_profile_revision")),
            None,
        )
        if revision:
            lines.append(f"  Legacy revision: {revision}")
    return lines


def _ordering_line(rows: list[dict]) -> str | None:
    """State plainly whether any two rungs are actually separated."""
    with_ci = [(r, _ci_bounds(r.get("net_shift_ci95"))) for r in rows]
    with_ci = [(r, ci) for r, ci in with_ci if ci is not None]
    if len(with_ci) < 2:
        return None
    separated: list[str] = []
    pairs = 0
    for i, (row_a, ci_a) in enumerate(with_ci):
        for row_b, ci_b in with_ci[i + 1:]:
            pairs += 1
            if ci_a[0] > ci_b[1] or ci_b[0] > ci_a[1]:
                hi, lo = (row_a, row_b) if ci_a[0] > ci_b[1] else (row_b, row_a)
                separated.append(
                    f"{str(hi['model']).split('/')[-1]} > {str(lo['model']).split('/')[-1]}"
                )
    if not separated:
        return (
            f"- Ordering: NOT resolved — all {pairs} rung pairs have overlapping 95% CIs. "
            "Read the rungs as 'all shifted', not as a ranking."
        )
    return (
        f"- Ordering: {len(separated)} of {pairs} rung pairs separate at 95%: "
        + ", ".join(separated)
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_report(
    out_dir: Path, persona: str, rows: list[dict], *, source: str = "sweep"
) -> dict:
    """Write ``report.json`` + ``report.md`` into ``out_dir``; return the payload.

    Both files are written even when rungs failed — the report IS the diagnostic
    for a broken sweep, and a failed rung is a row carrying an ``error``. That
    makes ``report.json`` a *record*, not a completion certificate: the
    skip-if-done gate in ``scripts/octt_plan.sh`` must read it
    (:mod:`octt.phase_status`) rather than stat it, or an all-failed sweep would
    retire its phase forever.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source": source,
        "persona": persona,
        "rows": rows,
    }
    (out_dir / REPORT_JSON_BASE_NAME).write_text(json.dumps(payload, indent=2) + "\n")
    note = (
        ""
        if source == "sweep"
        else (
            "\n_Rebuilt from banked artifacts (`octt scaling --report-only`); no sampling, "
            "no spend. Shifts are recomputed from the banked Elo tables under the current "
            "trait profiles, so they may differ from what the sweep printed._\n"
        )
    )
    (out_dir / REPORT_MD_BASE_NAME).write_text(
        f"# Dense-vs-MoE revealed-preference shifts: persona '{persona}'\n{note}\n"
        + to_markdown(rows)
        + "\n"
    )
    return payload


# ---------------------------------------------------------------------------
# Rebuilding from banked artifacts (free: no network, no Tinker, no sampling)
# ---------------------------------------------------------------------------


def rung_dirs(run_dir: Path) -> list[Path]:
    """Rung subdirectories of a sweep directory, in sweep (cost) order."""
    run_dir = Path(run_dir)
    found = [
        path
        for path in sorted(run_dir.iterdir())
        if path.is_dir()
        and (
            (path / EVAL_RESULTS_BASE_NAME).is_file() or (path / MANIFEST_BASE_NAME).is_file()
        )
    ]
    return sorted(found, key=lambda p: (_order_key(_model_from_dir_name(p.name)), p.name))


def rows_from_run_dir(run_dir: Path, *, persona: str | None = None) -> tuple[str, list[dict]]:
    """Rebuild ``(persona, rows)`` from a finished run directory's artifacts.

    Reads ``<rung>/eval_results.json`` (Elo tables, recipe, coverage) and
    ``<rung>/manifest.json`` (model id, execution mode) only. The net shift is
    **recomputed** from the banked Elo tables with the current trait profiles
    whenever both tables are present, falling back to the banked
    ``shift_summary`` otherwise; ``shift_source`` on each row says which.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"{run_dir} is not a directory")
    dirs = rung_dirs(run_dir)
    if not dirs and (run_dir / EVAL_RESULTS_BASE_NAME).is_file():
        # A single-model `octt run` directory is a one-rung sweep.
        dirs = [run_dir]
    if not dirs:
        raise ValueError(
            f"{run_dir} holds no rungs: expected <rung>/{EVAL_RESULTS_BASE_NAME} or "
            f"<rung>/{MANIFEST_BASE_NAME}"
        )

    rows: list[dict] = []
    personas: set[str] = set()
    for rung in dirs:
        row, rung_persona = _row_from_rung(rung, persona)
        if rung_persona:
            personas.add(rung_persona)
        rows.append(row)

    if persona is not None and personas - {persona}:
        raise ValueError(
            f"--persona {persona!r} contradicts the banked artifacts "
            f"({sorted(personas)}): recomputing a shift under the wrong trait profile "
            "produces a plausible number that means nothing"
        )
    if len(personas) > 1:
        raise ValueError(
            f"{run_dir} mixes personas {sorted(personas)}; a sweep report covers one persona"
        )
    resolved = persona or (personas.pop() if personas else None)
    if resolved is None:
        raise ValueError(
            f"{run_dir} records no persona; pass one explicitly to rebuild the report"
        )
    rows.extend(_dropped_rung_rows(run_dir, rows))
    rows.sort(key=lambda r: (_order_key(r.get("model")), str(r.get("model"))))
    return resolved, rows


def _dropped_rung_rows(run_dir: Path, rebuilt: list[dict]) -> list[dict]:
    """Error rows for rungs the previous ``report.json`` had but the rebuild does not.

    Scanning rung *directories* only catches a rung that started: a sweep whose
    later rungs never ran (``experiments.scaling`` creates ``<out>/<model>/``
    inside ``pipeline.run``, so a rung that raised first leaves nothing on disk)
    has no directory to turn into an error row. Rebuilding in place would then
    replace a ``report.json`` recording "2 FAILED of 4" with one recording "2
    completed rungs", and ``scripts/octt_plan.sh``'s skip-if-done gate
    (:func:`octt.phase_status.marker_status`) would retire a paid phase that
    only half ran.

    So every model the previous report named is carried forward, and any whose
    artifacts are gone comes back as an ``error`` row. Fail-closed on purpose: a
    rung whose directory was deleted is indistinguishable from one that never
    ran, and only one of those two is safe to skip.
    """
    previous = _load_json(run_dir / REPORT_JSON_BASE_NAME)
    if not isinstance(previous, dict) or not isinstance(previous.get("rows"), list):
        return []
    have = {row.get("model") for row in rebuilt}
    carried: list[dict] = []
    for row in previous["rows"]:
        if not isinstance(row, dict):
            continue
        model = row.get("model")
        if not isinstance(model, str) or model in have:
            continue
        have.add(model)
        recorded = str(row["error"]).strip() if row.get("error") else ""
        detail = f" (previously recorded: {recorded})" if recorded else ""
        carried.append(
            error_row(
                model,
                f"named in {run_dir / REPORT_JSON_BASE_NAME} but no artifacts under "
                f"{run_dir / model.replace('/', '-')} — rung never ran, or its "
                f"results were deleted{detail}",
            )
        )
    return carried


def _row_from_rung(rung: Path, persona: str | None) -> tuple[dict, str | None]:
    evaluation = _load_json(rung / EVAL_RESULTS_BASE_NAME)
    rung_manifest = _load_json(rung / MANIFEST_BASE_NAME)
    model = (
        (evaluation or {}).get("student_model")
        or (rung_manifest or {}).get("model")
        or _model_from_dir_name(rung.name)
    )
    # An unfinished rung still names its persona in the manifest, and a sweep
    # whose LAST rung died must still rebuild a report (that report is the
    # diagnostic), so the persona is resolved before the completeness check.
    rung_persona = (evaluation or {}).get("persona") or (rung_manifest or {}).get("persona")
    if not isinstance(rung_persona, str):
        rung_persona = None
    if not isinstance(model, str):
        return (
            error_row(rung.name, f"{rung}: cannot identify the model for this rung"),
            rung_persona,
        )

    if not isinstance(evaluation, dict):
        mode = (rung_manifest or {}).get("execution_mode", "unknown")
        stages = sorted((rung_manifest or {}).get("stages", {}))
        detail = f"stages recorded: {', '.join(stages)}" if stages else "no stages recorded"
        return (
            error_row(
                model,
                f"no {EVAL_RESULTS_BASE_NAME} in {rung} — rung did not finish "
                f"(execution_mode={mode}, {detail})",
            ),
            rung_persona,
        )

    summary, source = _shift_summary(evaluation, persona or rung_persona)
    row = result_row(
        model,
        persona=rung_persona,
        eval_target=evaluation.get("eval_target"),
        recipe=evaluation.get("recipe"),
        shift_summary=summary,
        capability_benchmarks=evaluation.get("capability_benchmarks"),
        judgment_coverage=evaluation.get("judgment_coverage"),
        judge_model=evaluation.get("judge_model"),
        shift_source=source,
    )
    return row, rung_persona


def _shift_summary(evaluation: dict, persona: str | None) -> tuple[dict, str]:
    """Recompute the shift from banked Elo tables; fall back to the banked summary."""
    base = evaluation.get("base_elo")
    trained = evaluation.get("trained_elo")
    if isinstance(persona, str) and _is_elo(base) and _is_elo(trained):
        return trait_profiles.summarize_shift(base, trained, persona), "recomputed"
    banked = evaluation.get("shift_summary")
    return (banked if isinstance(banked, dict) else {}), "banked"


def _is_elo(table: object) -> bool:
    return isinstance(table, dict) and bool(table)


def rebuild_report(
    run_dir: Path, *, out_dir: Path | None = None, persona: str | None = None
) -> dict:
    """Rebuild ``report.json`` + ``report.md`` from a run directory. Costs nothing.

    ``out_dir`` defaults to ``run_dir`` (regenerate in place). Point it elsewhere
    to explore a curation change without overwriting the phase gate's marker.
    """
    resolved, rows = rows_from_run_dir(run_dir, persona=persona)
    return write_report(
        Path(out_dir) if out_dir is not None else Path(run_dir),
        resolved,
        rows,
        source="rebuilt",
    )


def _model_from_dir_name(name: str) -> str | None:
    """Rung directories are ``tinker_id.replace('/', '-')`` (see experiments.scaling)."""
    for tinker_id in models.CANDIDATES:
        if tinker_id.replace("/", "-") == name:
            return tinker_id
    return None


def _order_key(model: object) -> int:
    return _CANONICAL_ORDER.get(model, len(_CANONICAL_ORDER)) if isinstance(model, str) else (
        len(_CANONICAL_ORDER)
    )


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
