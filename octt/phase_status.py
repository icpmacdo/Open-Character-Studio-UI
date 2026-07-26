"""Does a phase's completion marker record a SUCCESSFUL run?

``scripts/octt_plan.sh`` skips a paid phase when that phase's marker file
exists, so the marker has to mean "this phase produced a result" — not merely
"the process lived long enough to write a file". Two writers break that on their
own, and both are right to:

* ``experiments.scaling.run_and_report`` writes ``report.json`` even when every
  rung failed: the report IS the diagnostic for a broken sweep (a failed rung is
  a row carrying an ``error``). A sweep that failed on all four rungs would
  otherwise retire ``dense-sweep`` forever.
* ``octt.capabilities.evaluate`` writes ``capability_eval.json`` with
  ``status='failed'`` instead of raising after the expensive stages already ran.

Keeping those diagnostics and keeping the gate honest only conflict if the gate
stats the marker instead of reading it. So it reads it, here.

Nothing in this module changes what a successful marker looks like: every
verdict comes from fields the writers already persist (``rows[].error``,
``status``, and ``manifest.json``'s ``execution_mode``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .manifest import EXECUTION_MODE_DRY_RUN, MANIFEST_BASE_NAME


@dataclass(frozen=True)
class MarkerStatus:
    """Verdict for one marker: may the gate skip the phase, and why."""

    complete: bool
    reason: str


def marker_status(marker: Path, run_dir: Path | None = None) -> MarkerStatus:
    """Decide whether ``marker`` records a completed *paid* phase.

    ``run_dir`` is the phase's output directory (defaults to the marker's
    parent). It is scanned for dry-run provenance because a dry run into a paid
    phase's directory writes a marker whose rows all look successful, which
    would skip the paid phase that never ran.
    """
    marker = Path(marker)
    if not marker.is_file():
        return MarkerStatus(False, f"{marker} does not exist")
    payload = _load_json(marker)
    if not isinstance(payload, dict):
        return MarkerStatus(False, f"{marker} is not readable as a JSON object")

    status = _payload_status(marker, payload)
    if not status.complete:
        return status
    dry_run_reason = _dry_run_reason(Path(run_dir) if run_dir is not None else marker.parent)
    if dry_run_reason:
        return MarkerStatus(False, dry_run_reason)
    return status


def _payload_status(marker: Path, payload: dict) -> MarkerStatus:
    """Dispatch on the marker's shape; each writer has its own success field."""
    if "rows" in payload:
        return _scaling_report_status(marker, payload)
    if "status" in payload and "suite" in payload:
        return _capability_eval_status(marker, payload)
    return _eval_results_status(marker, payload)


def _scaling_report_status(marker: Path, payload: dict) -> MarkerStatus:
    """``experiments/scaling.py`` report.json: complete iff no rung FAILED."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return MarkerStatus(False, f"{marker} records no rungs")
    failed = [row for row in rows if isinstance(row, dict) and row.get("error")]
    if failed:
        detail = "; ".join(
            f"{_model_label(row)}: {_first_line(row['error'])}" for row in failed
        )
        return MarkerStatus(
            False,
            f"{marker} records {len(failed)} FAILED of {len(rows)} rungs — {detail}",
        )
    return MarkerStatus(True, f"{marker} records {len(rows)} completed rungs")


def _capability_eval_status(marker: Path, payload: dict) -> MarkerStatus:
    """``octt/capabilities.py`` capability_eval.json: complete iff status=completed."""
    status = payload.get("status")
    if status == "completed":
        return MarkerStatus(True, f"{marker} records status={status!r}")
    error = payload.get("error")
    detail = f" — {_first_line(error)}" if error else ""
    return MarkerStatus(False, f"{marker} records status={status!r}{detail}")


def _eval_results_status(marker: Path, payload: dict) -> MarkerStatus:
    """``octt/pipeline.py`` eval_results.json: written only after the paid run finished.

    A failed ``capability_benchmarks`` sub-result deliberately does NOT count against
    it. Capability benchmarking is an optional, free, local LightEval add-on
    (``--eval-capabilities``) that runs *after* the paid pipeline is already complete,
    and it carries its own marker — ``capability_eval.json``, judged by
    :func:`_capability_eval_status` — which ``cmd_lighteval_smoke`` gates on. Letting a
    local LightEval failure mark the phase incomplete would relaunch, and re-spend, a
    paid phase whose training and eval both succeeded. LightEval is currently blocked on
    GPU serving, so that failure is the common case, not the rare one.

    Existence is therefore the right signal here: unlike ``report.json`` (written even
    when every rung failed), ``pipeline.run`` reaches this write only after the stages it
    summarises have completed. ``marker_status`` has already rejected unreadable payloads
    and dry-run directories before this runs.
    """
    del payload  # every readable payload at this point records a completed paid run
    return MarkerStatus(True, f"{marker} records a completed run")


def _dry_run_reason(run_dir: Path) -> str | None:
    """Flag a directory whose manifests say it holds a dry run, not paid results.

    ``octt run`` writes ``<out>/manifest.json``; ``octt scaling`` writes one per
    rung under ``<out>/<model>/``. Only a positive ``execution_mode`` of
    ``dry-run`` counts, so manifest-less or legacy directories stay unjudged.
    """
    if not run_dir.is_dir():
        return None
    candidates = [run_dir / MANIFEST_BASE_NAME, *sorted(run_dir.glob(f"*/{MANIFEST_BASE_NAME}"))]
    for path in candidates:
        data = _load_json(path)
        if isinstance(data, dict) and data.get("execution_mode") == EXECUTION_MODE_DRY_RUN:
            return (
                f"{path} records execution_mode='dry-run': this directory holds a dry run, "
                "not paid results (a paid resume here would fail the manifest mode check)"
            )
    return None


def _load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _model_label(row: dict) -> str:
    return str(row.get("model", "unknown model")).split("/")[-1]


def _first_line(text: object, limit: int = 120) -> str:
    lines = str(text).strip().splitlines()
    if not lines:
        return "unknown failure"
    line = lines[0]
    return line if len(line) <= limit else line[: limit - 1] + "…"
