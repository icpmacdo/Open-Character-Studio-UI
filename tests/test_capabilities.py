"""Tests for the opt-in LightEval capability harness."""

from __future__ import annotations

import json
import subprocess

from octt import capabilities
from octt.config import MMLU_SUBJECTS, get_capability_config


def test_capability_configs_cover_requested_benchmarks():
    smoke = get_capability_config("smoke")
    assert smoke.max_samples == 8
    assert [task.label for task in smoke.benchmarks] == ["truthfulqa"]
    assert smoke.benchmarks[0].spec == "leaderboard|truthfulqa:mc|0"

    full = get_capability_config("full")
    labels = {task.label for task in full.benchmarks}
    assert {"truthfulqa", "winogrande", "hellaswag", "arc-c"} <= labels
    assert sum(1 for task in full.benchmarks if task.label.startswith("mmlu:")) == len(MMLU_SUBJECTS)
    assert any(task.spec == "leaderboard|arc:challenge|25" for task in full.benchmarks)


def test_build_command_uses_lighteval_accelerate(tmp_path):
    cfg = get_capability_config("smoke")
    cmd = capabilities.build_command(
        "Qwen/Qwen3.5-4B",
        cfg,
        tmp_path / "cap",
    )
    assert cmd[:3] == ["lighteval", "accelerate", "model_name=Qwen/Qwen3.5-4B"]
    assert cmd[3] == "leaderboard|truthfulqa:mc|0"
    assert "--output-dir" in cmd
    assert "--max-samples" in cmd and "8" in cmd
    assert "--remove-reasoning-tags" in cmd


def test_evaluate_dry_run_writes_command_preview(tmp_path):
    result = capabilities.evaluate(
        student_model="Qwen/Qwen3.5-4B",
        output_dir=tmp_path / "cap",
        config=get_capability_config("smoke"),
        dry_run=True,
    )
    assert result["status"] == "preview"
    assert result["target"]["label"] == "base-model-preview"
    assert "lighteval accelerate" in result["command_preview"]
    assert (tmp_path / "cap" / "capability_eval.json").exists()


def test_evaluate_execute_parses_lighteval_results(tmp_path, monkeypatch):
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: "/usr/bin/lighteval")

    def fake_runner(command, **_kw):
        out_dir = command[command.index("--output-dir") + 1]
        result_dir = tmp_path / "cap" / "results" / "Qwen-Qwen3.5-4B"
        assert out_dir == str(tmp_path / "cap")
        result_dir.mkdir(parents=True)
        (result_dir / "results_2026-06-21.json").write_text(
            json.dumps(
                {
                    "results": {
                        "truthfulqa:mc|0": {"acc": 0.5},
                        "all": {"acc": 0.5},
                    }
                }
            )
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    result = capabilities.evaluate(
        student_model="Qwen/Qwen3.5-4B",
        output_dir=tmp_path / "cap",
        config=get_capability_config("smoke"),
        dry_run=False,
        model_ref="Qwen/Qwen3.5-4B",
        target_label="explicit-model",
        runner=fake_runner,
    )
    assert result["status"] == "completed"
    assert result["scores"] == {"truthfulqa:mc|0": {"acc": 0.5}}
    assert (tmp_path / "cap" / "lighteval.stdout.txt").read_text() == "ok\n"
