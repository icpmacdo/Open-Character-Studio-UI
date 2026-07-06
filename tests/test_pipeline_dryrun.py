"""End-to-end dry-run pipeline tests: artifacts, manifest, resume, eval signal."""

from __future__ import annotations

import json

from octt import models, pipeline
from octt.config import get_capability_config, get_config


def _run(out, persona="humorous", **kw):
    return pipeline.run(
        persona, "Qwen/Qwen3.5-4B", models.TEACHER_MODEL, out,
        config=get_config("smoke"), dry_run=True, **kw,
    )


def test_pipeline_dry_run_produces_all_artifacts(tmp_path):
    out = tmp_path / "run"
    res = _run(out)
    assert res.dpo_checkpoint.ok
    assert res.sft_checkpoint.ok
    assert res.final_checkpoint.ok
    for name in ("dpo_pairs.jsonl", "introspection.jsonl", "manifest.json", "eval_results.json"):
        assert (out / name).exists(), name

    stages = json.loads((out / "manifest.json").read_text())["stages"]
    assert {"dpo", "sft", "merge"} <= set(stages)


def test_pipeline_sequential_sft_and_official_merge_weights(tmp_path):
    """Official recipe: SFT inits from the DPO state; merge realizes
    1.0·ΔW_dpo + 0.25·ΔW_intro via concat weights (0.75, 0.25)."""
    out = tmp_path / "run"
    res = _run(out)
    assert res.sft_checkpoint.extra["init_from_dpo"] is True
    assert res.sft_checkpoint.extra["init_state_path"] == res.dpo_checkpoint.state_path
    assert res.final_checkpoint.extra["weights"] == [0.75, 0.25]
    assert res.final_checkpoint.extra["sft_init_from_dpo"] is True


def test_pipeline_eval_shows_persona_shift(tmp_path):
    out = tmp_path / "run"
    res = _run(out)
    assert res.persona_trait_shift is not None
    assert res.persona_trait_shift > 0  # aligned traits up, opposing down (net)
    assert res.eval_target == "dry-run"

    # Lock the persisted eval_results.json shape (paper-faithful summary, not a
    # single self-named trait).
    summary = json.loads((out / "eval_results.json").read_text())
    assert set(summary) == {
        "persona", "student_model", "eval_target", "shift_summary",
        "recipe", "base_elo", "trained_elo", "conditions",
    }
    assert summary["conditions"] == ["adopt"]
    assert summary["eval_target"] == "dry-run"
    assert summary["recipe"]["merge_adapters"] is True
    assert summary["recipe"]["dpo_lora_rank"] == 64
    assert summary["recipe"]["sft_lora_rank"] == 64
    s = summary["shift_summary"]
    assert s["top_increased"] and s["top_decreased"]
    assert s["net_shift"] > 0


def test_pipeline_capability_eval_preview_is_opt_in(tmp_path):
    out = tmp_path / "run"
    res = _run(out, run_capabilities=True, capability_config=get_capability_config("smoke"))
    assert res.capability_benchmarks["status"] == "preview"

    summary = json.loads((out / "eval_results.json").read_text())
    cap = summary["capability_benchmarks"]
    assert cap["status"] == "preview"
    assert cap["suite"] == "smoke"
    assert cap["target"]["label"] == "base-model-preview"
    assert cap["max_samples"] == 8
    assert cap["tasks"][0]["suite"] == "leaderboard"
    assert cap["tasks"][0]["spec"] == "truthfulqa:mc|0"
    assert (out / "eval" / "capabilities" / "capability_eval.json").exists()


def test_pipeline_resume_skips_finished_stages(tmp_path):
    out = tmp_path / "run"
    first = _run(out)
    # Corrupt the pairs file so regeneration would change content; resume must skip it.
    pairs = out / "dpo_pairs.jsonl"
    original = pairs.read_text()
    pairs.write_text("CORRUPTED\n")
    second = _run(out)
    assert second.run_id == first.run_id
    assert second.dpo_checkpoint.sampler_path == first.dpo_checkpoint.sampler_path
    assert second.final_checkpoint.sampler_path == first.final_checkpoint.sampler_path
    # The corrupted pairs file was NOT regenerated (stage skipped from manifest).
    assert pairs.read_text() == "CORRUPTED\n" != original


def test_pipeline_without_merge_uses_sft_as_final(tmp_path):
    out = tmp_path / "run"
    cfg = get_config("smoke")
    cfg_no_merge = cfg.__class__(dpo=cfg.dpo, sft=cfg.sft, eval=cfg.eval, merge_adapters=False)
    res = pipeline.run(
        "humorous", "Qwen/Qwen3.5-4B", models.TEACHER_MODEL, out,
        config=cfg_no_merge, dry_run=True,
    )
    assert res.final_checkpoint.sampler_path == res.sft_checkpoint.sampler_path
    assert res.final_checkpoint.extra["merge_skipped"] is True
    assert res.eval_target == "sft-direct"
    summary = json.loads((out / "eval_results.json").read_text())
    assert summary["eval_target"] == "sft-direct"
    assert summary["recipe"]["merge_adapters"] is False
    stages = json.loads((out / "manifest.json").read_text())["stages"]
    assert stages["merge"]["extra"]["merge_skipped"] is True
