"""Tests for Tinker setup helpers that do not require the paid runtime."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from octt import models, tinker_client
from octt.config import get_config


def test_vendored_cookbook_renderer_names_resolve():
    plans = tinker_client.plan_renderers(
        (
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.6-27B",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        )
    )
    by_model = {plan.model_id: plan.renderer_name for plan in plans}

    assert by_model["Qwen/Qwen3.5-4B"] == "qwen3_5_disable_thinking"
    assert by_model["Qwen/Qwen3.6-27B"] == "qwen3_5_disable_thinking"
    # Nemotron-3's recommended renderers are full-reasoning; the study policy
    # (reasoning OFF for hybrid models, uniform across ladders) requires the
    # disable-thinking variants or dense-vs-MoE Elo measures renderer mode.
    assert by_model["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"] == "nemotron3_disable_thinking"
    assert by_model["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"] == "nemotron3_disable_thinking"
    assert (
        by_model["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"]
        == "nemotron3_ultra_disable_thinking"
    )


def test_every_study_model_resolves_to_a_non_thinking_renderer():
    thinking_suffixes = ("_disable_thinking",)
    for model_id in models.SCALING_SET + ("Qwen/Qwen3.6-35B-A3B",):
        name = tinker_client.resolve_renderer_name(model_id)
        assert name.endswith(thinking_suffixes) or name == tinker_client.TML_PINNED_RENDERER_NAME, (
            f"{model_id} resolved to {name!r}, which is not a reasoning-off renderer"
        )


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


def test_preflight_blocks_ultra_rank_above_known_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    report = tinker_client.build_preflight_report(
        student_models=("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",),
        config=get_config("smoke"),
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert not report.ok
    assert any("max LoRA rank is 32" in blocker for blocker in report.blockers)


def test_preflight_allows_ultra_rank32_no_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    cfg = get_config("smoke")
    cfg = replace(
        cfg,
        dpo=replace(cfg.dpo, lora_rank=32),
        sft=replace(cfg.sft, lora_rank=32),
        merge_adapters=False,
    )

    report = tinker_client.build_preflight_report(
        student_models=("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",),
        config=cfg,
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert report.ok
    assert not any("Local merge downloads" in warning for warning in report.warnings)


def test_preflight_warns_when_large_rungs_will_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    report = tinker_client.build_preflight_report(
        student_models=("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",),
        config=get_config("smoke"),
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert report.ok
    assert any("Local merge downloads" in warning for warning in report.warnings)


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

    assert combined.total_usd == pytest.approx(singles_total)
    eval_judge_lines = [line for line in combined.lines if line.stage == "eval.judge"]
    dpo_teacher_lines = [line for line in combined.lines if line.stage == "dpo.teacher_sample"]
    assert len(eval_judge_lines) == len(model_set)
    assert len(dpo_teacher_lines) == len(model_set)
    # Judge sampled tokens follow the configured max-token envelope.
    assert all(
        line.token_millions
        == (2 * cfg.eval.num_judgments * cfg.eval.judge_max_tokens) / 1_000_000
        for line in eval_judge_lines
    )
    # Teacher chosen-generation is costed at the thinking envelope (2048/tok).
    assert all(
        line.token_millions == (cfg.dpo.num_prompts * 2048) / 1_000_000
        for line in dpo_teacher_lines
    )
    # Prefill tokens are now billed (AR5): every sampling stage has a prefill line.
    stages = {line.stage for line in combined.lines}
    assert {"dpo.teacher_prefill", "introspection.prefill",
            "eval.model_prefill", "eval.judge_prefill"} <= stages


def test_cost_estimate_multiplies_every_eval_line_for_all_conditions():
    cfg = get_config("smoke")
    one = tinker_client.estimate_tinker_cost(
        cfg, ("Qwen/Qwen3.5-4B",), eval_conditions=1
    )
    all_conditions = tinker_client.estimate_tinker_cost(
        cfg, ("Qwen/Qwen3.5-4B",), eval_conditions=3
    )
    one_eval = sum(line.subtotal_usd for line in one.lines if line.stage.startswith("eval."))
    assert all_conditions.total_usd == pytest.approx(one.total_usd + 2 * one_eval)


def test_cost_estimate_rejects_zero_eval_conditions():
    with pytest.raises(ValueError, match="at least 1"):
        tinker_client.estimate_tinker_cost(
            get_config("smoke"),
            ("Qwen/Qwen3.5-4B",),
            eval_conditions=0,
        )


def test_cost_estimate_prices_judge_lines_at_the_judge_model():
    cfg = get_config("smoke")
    nano = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    estimate = tinker_client.estimate_tinker_cost(
        cfg, ("Qwen/Qwen3.5-4B",), judge_model=nano
    )
    judge_lines = {ln.stage: ln for ln in estimate.lines if ln.stage.startswith("eval.judge")}
    assert judge_lines["eval.judge"].model_id == nano
    assert judge_lines["eval.judge"].unit_price_usd == models.CANDIDATES[nano].price_sample
    assert judge_lines["eval.judge_prefill"].model_id == nano
    assert (
        judge_lines["eval.judge_prefill"].unit_price_usd
        == models.CANDIDATES[nano].price_prefill
    )
    # Default (no judge_model) keeps the paper convention: judge == teacher.
    default = tinker_client.estimate_tinker_cost(cfg, ("Qwen/Qwen3.5-4B",))
    default_judge = {ln.stage: ln for ln in default.lines if ln.stage == "eval.judge"}
    assert default_judge["eval.judge"].model_id == models.TEACHER_MODEL
    # The whole point: a Nano judge is strictly cheaper than the teacher judge.
    assert estimate.total_usd < default.total_usd
