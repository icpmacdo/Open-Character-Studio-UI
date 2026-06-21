"""Dry-run tests for the dense-vs-MoE scaling harness + report."""

from __future__ import annotations

import json

from octt import models
from octt.config import get_config
from experiments import scaling


def test_scaling_run_covers_set_and_summarizes(tmp_path):
    runs = scaling.run(
        "humorous", models.TEACHER_MODEL, tmp_path / "sweep",
        model_set=models.DENSE_LADDER, config=get_config("smoke"), dry_run=True,
    )
    assert len(runs) == len(models.DENSE_LADDER)
    rows = scaling.summarize(runs)
    assert {r["arch"] for r in rows} == {"dense"}
    assert all(isinstance(r["net_shift"], float) for r in rows)
    assert all(r["eval_target"] == "dry-run" for r in rows)


def test_scaling_report_files_written(tmp_path):
    out = tmp_path / "sweep"
    scaling.run_and_report(
        "sarcastic", models.TEACHER_MODEL, out,
        model_set=models.SCALING_SET, config=get_config("smoke"), dry_run=True,
    )
    report = json.loads((out / "report.json").read_text())
    assert report["persona"] == "sarcastic"
    assert len(report["rows"]) == len(models.SCALING_SET)
    md = (out / "report.md").read_text()
    assert "Dense" in md and "MoE" in md
    assert "Net shift" in md


def test_scaling_markdown_handles_missing_shift():
    md = scaling.to_markdown([
        {"model": "x/Y", "arch": "dense", "total_params_b": 4, "active_params_b": 4,
         "net_shift": None, "aligned_mean_delta": None, "opposing_mean_delta": None,
         "eval_target": None},
    ])
    assert "n/a" in md
