"""Tests for Tinker setup helpers that do not require the paid runtime."""

from __future__ import annotations

import asyncio

from octt import models, tinker_client
from octt.config import get_config


def test_vendored_cookbook_renderer_names_resolve():
    plans = tinker_client.plan_renderers(
        (
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.6-27B",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        )
    )
    by_model = {plan.model_id: plan.renderer_name for plan in plans}

    assert by_model["Qwen/Qwen3.5-4B"] == "qwen3_5_disable_thinking"
    assert by_model["Qwen/Qwen3.6-27B"] == "qwen3_5_disable_thinking"
    assert by_model["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"] == "nemotron3_ultra"


def test_dry_run_runtime_does_not_require_tinker_or_api_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    runtime = tinker_client.create_runtime(
        (models.TEACHER_MODEL, "Qwen/Qwen3.5-4B"),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )
    sample = asyncio.run(
        runtime.create_sampling_client("Qwen/Qwen3.5-4B").sample_async(
            prompt=None,
            num_samples=1,
            sampling_params=None,
        )
    )

    assert runtime.renderer_plan("Qwen/Qwen3.5-4B").renderer_name == "qwen3_5_disable_thinking"
    assert sample.sequences[0].tokens == []


def test_preflight_dry_run_does_not_require_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    report = tinker_client.build_preflight_report(
        student_models=("Qwen/Qwen3.5-4B",),
        config=get_config("smoke"),
        dry_run=True,
        output_dir=tmp_path / "runs",
    )

    assert report.ok
    assert not report.api_key_set


def test_preflight_real_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    report = tinker_client.build_preflight_report(
        student_models=("Qwen/Qwen3.5-4B",),
        config=get_config("smoke"),
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert not report.ok
    assert any("TINKER_API_KEY" in blocker for blocker in report.blockers)


def test_preflight_budget_blocks_when_estimate_exceeds_ceiling(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    report = tinker_client.build_preflight_report(
        student_models=("Qwen/Qwen3.5-4B",),
        config=get_config("smoke"),
        dry_run=False,
        budget_usd=0.000001,
        output_dir=tmp_path / "runs",
    )

    assert not report.ok
    assert any("exceeds budget" in blocker for blocker in report.blockers)


def test_cost_estimate_is_positive_for_smoke_scale():
    estimate = tinker_client.estimate_tinker_cost(
        get_config("smoke"),
        ("Qwen/Qwen3.5-4B",),
    )

    assert estimate.total_usd > 0


def test_cost_estimate_counts_per_model_eval_and_teacher_sampling():
    cfg = get_config("smoke")
    model_set = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")
    combined = tinker_client.estimate_tinker_cost(cfg, model_set)
    singles_total = sum(
        tinker_client.estimate_tinker_cost(cfg, (model,)).total_usd
        for model in model_set
    )

    assert combined.total_usd == singles_total
    eval_judge_lines = [line for line in combined.lines if line.stage == "eval.judge"]
    dpo_teacher_lines = [line for line in combined.lines if line.stage == "dpo.teacher_sample"]
    assert len(eval_judge_lines) == len(model_set)
    assert len(dpo_teacher_lines) == len(model_set)
    assert all(
        line.token_millions == (2 * cfg.eval.num_judgments * 512) / 1_000_000
        for line in eval_judge_lines
    )
