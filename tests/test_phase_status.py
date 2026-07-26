"""The skip-if-done gate must mean "completed successfully", not "file exists".

``scripts/octt_plan.sh`` skips a paid phase whose marker file is present, and
both marker writers persist a file on the failure path too: a scaling sweep
writes ``report.json`` even when every rung failed, and the capability harness
writes ``capability_eval.json`` with ``status='failed'``. Bare existence would
retire a phase that never produced a result — permanently, since nothing
deletes markers. These tests pin the reading of a marker to that distinction.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from experiments import scaling
from octt import capabilities, models, phase_status
from octt.config import get_capability_config, get_config
from octt.manifest import StageCheckpoint
from octt.pipeline import PipelineResult

GATE = Path(__file__).resolve().parent.parent / "scripts" / "octt_phase_complete.py"
SWEEP_MODELS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")


def _fake_pipeline_run(failing: tuple[str, ...]):
    """Stand in for a paid rung: fails for ``failing``, succeeds otherwise."""

    def run(**kwargs):
        model = kwargs["student_model"]
        if model in failing:
            raise RuntimeError(f"APIConnectionError: {model} sampling session died")
        ckpt = StageCheckpoint(sampler_path="tinker://fake/sampler_weights/final")
        return PipelineResult(
            persona=kwargs["persona"],
            student_model=model,
            run_id="fake",
            out_dir=kwargs["out_dir"],
            dpo_checkpoint=ckpt,
            sft_checkpoint=ckpt,
            final_checkpoint=ckpt,
            shift_summary={"net_shift": 12.0},
            eval_target="sft-direct",
        )

    return run


def _sweep(monkeypatch, tmp_path, failing: tuple[str, ...]) -> Path:
    monkeypatch.setattr(scaling.pipeline, "run", _fake_pipeline_run(failing))
    out = tmp_path / "sweep"
    scaling.run_and_report(
        "pirate", models.TEACHER_MODEL, out,
        model_set=SWEEP_MODELS, config=get_config("smoke"), dry_run=False,
    )
    return out


def test_all_failed_sweep_keeps_the_report_but_the_gate_refuses_it(monkeypatch, tmp_path):
    out = _sweep(monkeypatch, tmp_path, failing=SWEEP_MODELS)

    # The diagnostic survives: this is why the marker is written unconditionally.
    rows = json.loads((out / "report.json").read_text())["rows"]
    assert [row["model"] for row in rows] == list(SWEEP_MODELS)
    assert all(row["error"] for row in rows)
    assert (out / "report.md").read_text().count("FAILED") == len(SWEEP_MODELS)

    status = phase_status.marker_status(out / "report.json", out)
    assert not status.complete
    assert "2 FAILED of 2 rungs" in status.reason
    assert "Qwen3.5-4B" in status.reason and "Qwen3.5-9B" in status.reason


def test_partially_failed_sweep_is_not_complete(monkeypatch, tmp_path):
    out = _sweep(monkeypatch, tmp_path, failing=("Qwen/Qwen3.5-9B",))

    rows = json.loads((out / "report.json").read_text())["rows"]
    assert [bool(row.get("error")) for row in rows] == [False, True]

    status = phase_status.marker_status(out / "report.json", out)
    assert not status.complete
    assert "1 FAILED of 2 rungs" in status.reason
    assert "Qwen3.5-9B" in status.reason


def test_successful_sweep_is_complete(monkeypatch, tmp_path):
    out = _sweep(monkeypatch, tmp_path, failing=())

    status = phase_status.marker_status(out / "report.json", out)
    assert status.complete
    assert "2 completed rungs" in status.reason


def test_empty_sweep_report_is_not_complete(tmp_path):
    marker = tmp_path / "report.json"
    marker.write_text(json.dumps({"persona": "pirate", "rows": []}) + "\n")

    status = phase_status.marker_status(marker, tmp_path)
    assert not status.complete
    assert "no rungs" in status.reason


def test_dry_run_sweep_report_is_not_a_paid_completion(tmp_path):
    """A dry run into a paid phase's out dir writes an all-successful report."""
    out = tmp_path / "sweep"
    scaling.run_and_report(
        "pirate", models.TEACHER_MODEL, out,
        model_set=("Qwen/Qwen3.5-4B",), config=get_config("smoke"), dry_run=True,
    )
    rows = json.loads((out / "report.json").read_text())["rows"]
    assert not any(row.get("error") for row in rows)

    status = phase_status.marker_status(out / "report.json", out)
    assert not status.complete
    assert "execution_mode='dry-run'" in status.reason


def test_failed_capability_eval_is_not_complete(tmp_path):
    result = capabilities.evaluate(
        student_model="Qwen/Qwen3.5-4B",
        output_dir=tmp_path / "cap",
        config=get_capability_config("smoke"),
        dry_run=False,
    )
    assert result["status"] == "failed"

    status = phase_status.marker_status(tmp_path / "cap" / "capability_eval.json")
    assert not status.complete
    assert "status='failed'" in status.reason


def test_completed_capability_eval_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: "/usr/bin/lighteval")
    result = capabilities.evaluate(
        student_model="Qwen/Qwen3.5-4B",
        output_dir=tmp_path / "cap",
        config=get_capability_config("smoke"),
        dry_run=False,
        model_ref="Qwen/Qwen3.5-4B",
        target_label="explicit-model",
        runner=lambda command, **_kw: subprocess.CompletedProcess(command, 0, "", ""),
    )
    assert result["status"] == "completed"

    status = phase_status.marker_status(tmp_path / "cap" / "capability_eval.json")
    assert status.complete


def test_eval_results_marker_survives_a_failed_local_capability_run(tmp_path):
    """A free local LightEval failure must NOT relaunch the paid phase that preceded it.

    ``pipeline.run`` writes ``eval_results.json`` only after the paid training and eval
    finished, and ``--eval-capabilities`` bolts an optional local LightEval result onto
    the same file. Judging the paid marker by that sub-result would re-spend a completed
    phase — and since LightEval is blocked on GPU serving, the failure is the common case.
    The capability run has its own marker; see the two tests above.
    """
    payload = {
        "persona": "pirate",
        "student_model": "Qwen/Qwen3.5-4B",
        "eval_target": "sft-direct",
        "shift_summary": {"net_shift": 12.0},
    }
    marker = tmp_path / "eval_results.json"
    marker.write_text(json.dumps(payload) + "\n")
    assert phase_status.marker_status(marker, tmp_path).complete

    payload["capability_benchmarks"] = {"status": "failed", "error": "LightEval is not installed."}
    marker.write_text(json.dumps(payload) + "\n")
    assert phase_status.marker_status(marker, tmp_path).complete


def test_missing_or_unreadable_marker_is_not_complete(tmp_path):
    assert not phase_status.marker_status(tmp_path / "report.json", tmp_path).complete

    truncated = tmp_path / "report.json"
    truncated.write_text('{"persona": "pirate", "rows": [')
    status = phase_status.marker_status(truncated, tmp_path)
    assert not status.complete
    assert "JSON object" in status.reason


def test_gate_cli_exit_codes(monkeypatch, tmp_path):
    """The contract scripts/octt_plan.sh depends on: 0 skip, 1 run, 2 usage."""
    failed = _sweep(monkeypatch, tmp_path / "failed", failing=SWEEP_MODELS)
    ok = _sweep(monkeypatch, tmp_path / "ok", failing=())

    for out, expected in ((ok, 0), (failed, 1)):
        done = subprocess.run(
            [sys.executable, str(GATE), str(out / "report.json"), str(out)],
            capture_output=True, text=True, check=False,
        )
        assert done.returncode == expected
        assert str(out / "report.json") in done.stdout

    usage = subprocess.run(
        [sys.executable, str(GATE)], capture_output=True, text=True, check=False
    )
    assert usage.returncode == 2
