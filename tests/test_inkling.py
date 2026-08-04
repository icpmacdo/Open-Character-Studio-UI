"""Inkling track (INKLING_PLAN.md Phase 0): registry, renderer policy, preflight.

Everything here runs offline/dry-run; the one test that touches the runtime
stack skips cleanly when the training extras are absent.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from octt import distillation, models, pipeline, tinker_client
from octt.config import DPOConfig, get_config
from octt.constitution import load as load_constitution

INKLING = models.INKLING_MODEL
INKLING_SMALL = models.INKLING_SMALL_MODEL


def _inkling_recipe():
    cfg = get_config("smoke")
    return replace(
        cfg,
        dpo=replace(cfg.dpo, lora_rank=32),
        sft=replace(cfg.sft, lora_rank=32),
        merge_adapters=False,
    )


def test_inkling_spec_registered_but_not_a_scaling_rung():
    spec = models.CANDIDATES[INKLING]
    assert spec.arch == "moe"
    assert spec.family == "Inkling"
    assert spec.total_params_b == 975.0
    assert spec.active_params_b == 41.0
    assert spec.local_merge_feasible is False
    assert spec.price_prefill and spec.price_sample and spec.price_train
    # The scaling-study rungs are frozen; Inkling is a separate track.
    assert INKLING not in models.SCALING_SET
    assert models.assistant_name(INKLING) == "Inkling"


def test_teacher_model_is_priced_in_registry():
    spec = models.CANDIDATES[models.TEACHER_MODEL]
    assert spec.price_sample and spec.price_prefill and spec.price_train


def test_inkling_renderer_resolves_to_pinned_effort_override():
    plans = tinker_client.plan_renderers((INKLING,))
    assert plans[0].renderer_name == tinker_client.TML_PINNED_RENDERER_NAME


def test_think_prefill_support_by_renderer_name():
    assert tinker_client.renderer_supports_think_prefill("qwen3_5")
    assert tinker_client.renderer_supports_think_prefill("qwen3_5_disable_thinking")
    assert not tinker_client.renderer_supports_think_prefill("tml_v0")
    assert not tinker_client.renderer_supports_think_prefill(
        tinker_client.TML_PINNED_RENDERER_NAME
    )


def test_preflight_blocks_inkling_local_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    cfg = get_config("smoke")
    cfg = replace(cfg, dpo=replace(cfg.dpo, lora_rank=32), sft=replace(cfg.sft, lora_rank=32))

    report = tinker_client.build_preflight_report(
        student_models=(INKLING,),
        teacher_model=INKLING,
        config=cfg,
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert not report.ok
    assert any("cannot be merged locally" in blocker for blocker in report.blockers)


def test_preflight_allows_inkling_self_distillation_rank32_no_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    report = tinker_client.build_preflight_report(
        student_models=(INKLING,),
        teacher_model=INKLING,
        config=_inkling_recipe(),
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert report.ok
    # Self-distillation prices every teacher line off the registry entry.
    assert not any("missing sample price" in w for w in report.warnings)
    assert report.cost_estimate.total_usd > 0
    teacher_lines = [
        line for line in report.cost_estimate.lines if line.stage == "dpo.teacher_sample"
    ]
    assert teacher_lines and teacher_lines[0].model_id == INKLING
    assert teacher_lines[0].unit_price_usd == models.CANDIDATES[INKLING].price_sample


def test_pairs_cache_key_changes_when_prefill_disabled():
    constitution = load_constitution("humorous")
    args = (constitution, INKLING, INKLING, ["p1", "p2"], 1024, 2048, 0.7, None)
    with_prefill = distillation._pairs_cache_key(*args)
    default = distillation._pairs_cache_key(*args, teacher_prefill=True)
    without_prefill = distillation._pairs_cache_key(*args, teacher_prefill=False)
    assert with_prefill == default  # existing prefill-path keys stay stable
    assert without_prefill != with_prefill


def test_generate_pairs_offline_with_inkling_self_teacher(tmp_path):
    constitution = load_constitution("humorous")
    runtime = tinker_client.create_runtime(
        (INKLING,), config=tinker_client.TinkerClientConfig(dry_run=True)
    )
    out = tmp_path / "pairs.jsonl"
    distillation.generate_pairs(
        constitution, INKLING, INKLING, DPOConfig(num_prompts=2, batch_size=2),
        out, runtime, offline=True,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all(r["teacher"] == INKLING and r["student"] == INKLING for r in rows)
    meta = json.loads(out.with_suffix(".jsonl.meta.json").read_text())
    assert meta["num_pairs"] == len(rows)


def test_train_learning_rate_defaults_to_config(tmp_path):
    """Recipe lr policies must reach the optimizer (regression: pipeline never
    passed config.learning_rate, so for_scaling_study's lr=1e-4 was a no-op)."""
    runtime = tinker_client.create_runtime(
        (INKLING,), config=tinker_client.TinkerClientConfig(dry_run=True)
    )
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("")
    cfg = DPOConfig(learning_rate=1e-4)
    distillation.train(INKLING, pairs, cfg, tmp_path / "dpo", runtime)
    meta = json.loads((tmp_path / "dpo" / "dpo_train.meta.json").read_text())
    assert meta["learning_rate"] == 1e-4


def test_pipeline_dry_run_inkling_self_distillation_with_external_judge(tmp_path):
    out = tmp_path / "run"
    res = pipeline.run(
        "humorous", INKLING, INKLING, out,
        config=_inkling_recipe(), dry_run=True, judge_model=models.TEACHER_MODEL,
    )
    assert res.dpo_checkpoint.ok and res.sft_checkpoint.ok and res.final_checkpoint.ok
    assert res.eval_target == "sft-direct"  # no-merge: SFT sampler served directly

    summary = json.loads((out / "eval_results.json").read_text())
    assert summary["judge_model"] == models.TEACHER_MODEL
    assert summary["student_model"] == INKLING


def test_forecaster_constitution_well_formed_with_trait_profile():
    from octt import trait_profiles

    c = load_constitution("forecaster")
    assert len(c.assertions) == 10
    assert all(a and not a.startswith("-") for a in c.assertions)
    # Reasons-attached style: most assertions carry an explicit "because".
    assert sum("because" in a for a in c.assertions) >= 4
    prof = trait_profiles.profile("forecaster")
    assert prof is not None and trait_profiles.required_traits("forecaster")


def test_pinned_effort_renderer_registers_with_runtime_stack():
    pytest.importorskip("tinker")
    pytest.importorskip("tml_renderers")
    stack = tinker_client.import_tinker_stack()
    assert stack.renderers.is_renderer_registered(tinker_client.TML_PINNED_RENDERER_NAME)


# --- Inkling-Small (landed on Tinker 2026-07-30; INKLING_PLAN.md decision 5:
# --- the cheap rung reruns Phases 2-4, full Inkling validates transfer) ---


def test_inkling_small_spec_is_the_cheap_rung_not_a_scaling_rung():
    spec = models.CANDIDATES[INKLING_SMALL]
    assert spec.arch == "moe"
    assert spec.family == "Inkling"
    assert spec.total_params_b == 276.0
    assert spec.active_params_b == 12.0
    assert spec.local_merge_feasible is False
    assert spec.max_lora_rank == 64  # service-verified 2026-07-30 (128 rejected)
    assert spec.price_prefill and spec.price_sample and spec.price_train
    # Cheaper than full Inkling on every axis — the point of the swap-in.
    big = models.CANDIDATES[INKLING]
    assert spec.price_prefill < big.price_prefill
    assert spec.price_sample < big.price_sample
    assert spec.price_train < big.price_train
    assert INKLING_SMALL not in models.SCALING_SET
    assert models.assistant_name(INKLING_SMALL) == "Inkling"


def test_inkling_small_renderer_resolves_to_pinned_effort_override():
    # The vendored model_info exact-matches the full Inkling id, so this only
    # passes through the TML-org fallback in octt/tinker_client.py.
    plans = tinker_client.plan_renderers((INKLING_SMALL,))
    assert plans[0].renderer_name == tinker_client.TML_PINNED_RENDERER_NAME


def test_renderer_fallback_covers_only_the_tml_org():
    with pytest.raises(tinker_client.TinkerSetupError):
        tinker_client.resolve_renderer_name("unknown-org/Not-A-Model")


def test_tml_tokenizer_routing_covers_inkling_small():
    pytest.importorskip("tml_renderers")
    tok = tinker_client._tml_aware_get_tokenizer(INKLING_SMALL)
    assert hasattr(tok, "tml_tokenizer")  # the o200k facade, not an HF tokenizer


def test_vendored_by_name_tokenizer_path_covers_inkling_small():
    """Cookbook dataset builders call the VENDORED get_tokenizer by model name,
    bypassing octt's stack wrapper — the 2026-07-30 smoke crash. After
    import_tinker_stack, the registry hook must make that exact path return the
    tml adapter for every TML-org registry model."""
    pytest.importorskip("tinker")
    pytest.importorskip("tml_renderers")
    import importlib

    tinker_client.import_tinker_stack()
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    for model_id in (INKLING_SMALL, INKLING):
        tok = tokenizer_utils.get_tokenizer(model_id)
        assert hasattr(tok, "tml_tokenizer"), model_id


def test_preflight_blocks_inkling_small_local_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    cfg = get_config("smoke")
    cfg = replace(cfg, dpo=replace(cfg.dpo, lora_rank=32), sft=replace(cfg.sft, lora_rank=32))

    report = tinker_client.build_preflight_report(
        student_models=(INKLING_SMALL,),
        teacher_model=INKLING_SMALL,
        config=cfg,
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert not report.ok
    assert any("cannot be merged locally" in blocker for blocker in report.blockers)


def test_preflight_allows_inkling_small_self_distillation_rank32_no_merge(monkeypatch, tmp_path):
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    report = tinker_client.build_preflight_report(
        student_models=(INKLING_SMALL,),
        teacher_model=INKLING_SMALL,
        config=_inkling_recipe(),
        dry_run=False,
        output_dir=tmp_path / "runs",
    )

    assert report.ok
    assert not any("missing sample price" in w for w in report.warnings)
    assert report.cost_estimate.total_usd > 0
    teacher_lines = [
        line for line in report.cost_estimate.lines if line.stage == "dpo.teacher_sample"
    ]
    assert teacher_lines and teacher_lines[0].model_id == INKLING_SMALL
    assert teacher_lines[0].unit_price_usd == models.CANDIDATES[INKLING_SMALL].price_sample


def test_preflight_enforces_inkling_small_rank_cap(monkeypatch, tmp_path):
    """Rank 64 (the paper recipe and the verified cap) passes; above-cap blocks."""
    monkeypatch.setenv("TINKER_API_KEY", "test-key")

    def _report(rank):
        cfg = get_config("smoke")
        cfg = replace(
            cfg,
            dpo=replace(cfg.dpo, lora_rank=rank),
            sft=replace(cfg.sft, lora_rank=rank),
            merge_adapters=False,
        )
        return tinker_client.build_preflight_report(
            student_models=(INKLING_SMALL,),
            teacher_model=INKLING_SMALL,
            config=cfg,
            dry_run=False,
            output_dir=tmp_path / "runs",
        )

    assert _report(64).ok
    blocked = _report(128)
    assert not blocked.ok
    assert any("max LoRA rank is 64" in b for b in blocked.blockers)


def test_pipeline_dry_run_inkling_small_self_distillation(tmp_path):
    out = tmp_path / "run"
    res = pipeline.run(
        "humorous", INKLING_SMALL, INKLING_SMALL, out,
        config=_inkling_recipe(), dry_run=True, judge_model=models.TEACHER_MODEL,
    )
    assert res.dpo_checkpoint.ok and res.sft_checkpoint.ok and res.final_checkpoint.ok
    assert res.eval_target == "sft-direct"  # no-merge: SFT sampler served directly

    summary = json.loads((out / "eval_results.json").read_text())
    assert summary["judge_model"] == models.TEACHER_MODEL
    assert summary["student_model"] == INKLING_SMALL
