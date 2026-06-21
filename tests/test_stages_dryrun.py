"""Dry-run tests for the DPO and introspection stages (data + checkpoints)."""

from __future__ import annotations

import json

from octt import constitution, distillation, introspection, models, tinker_client
from octt.config import get_config


def _dry_runtime(student="Qwen/Qwen3.5-4B"):
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL, student),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def test_generate_pairs_format_and_cache(tmp_path):
    runtime = _dry_runtime()
    cfg = get_config("smoke").dpo
    c = constitution.load("humorous")
    out = tmp_path / "pairs.jsonl"
    distillation.generate_pairs(
        c, models.TEACHER_MODEL, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == cfg.num_prompts
    row = rows[0]
    # Human-readable + cookbook-trainable views both present.
    assert {"prompt", "chosen", "rejected", "comparison", "label"} <= set(row)
    assert row["label"] == "A"
    assert row["comparison"]["completion_A"][0]["content"] == row["chosen"]

    # Sidecar cache hash exists; a second call is a cache hit (mtime unchanged).
    meta = out.with_suffix(out.suffix + ".meta.json")
    assert json.loads(meta.read_text())["num_pairs"] == cfg.num_prompts
    mtime = out.stat().st_mtime_ns
    distillation.generate_pairs(
        c, models.TEACHER_MODEL, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True
    )
    assert out.stat().st_mtime_ns == mtime


def test_dpo_train_dry_run_returns_both_checkpoints(tmp_path):
    runtime = _dry_runtime()
    ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "pairs.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    assert ckpt.ok and ckpt.is_dry_run
    assert ckpt.sampler_path and ckpt.state_path


def test_generate_transcripts_format(tmp_path):
    runtime = _dry_runtime()
    cfg = get_config("smoke").sft
    c = constitution.load("poetic")
    dpo_ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "pairs.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    out = tmp_path / "introspection.jsonl"
    introspection.generate_transcripts(
        c, dpo_ckpt, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    expected = cfg.self_reflection_count + cfg.self_interaction_count
    assert len(rows) == expected
    # Every row is a chat transcript with assistant turns to train on.
    for r in rows:
        assert "messages" in r
        roles = [m["role"] for m in r["messages"]]
        assert "assistant" in roles


def test_self_interaction_turn_count(tmp_path):
    runtime = _dry_runtime()
    cfg = get_config("smoke").sft  # self_interaction_turns=2
    c = constitution.load("loving")
    dpo_ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "p.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    out = tmp_path / "i.jsonl"
    introspection.generate_transcripts(c, dpo_ckpt, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    interaction_rows = rows[cfg.self_reflection_count:]
    # 2-turn self-chat => 2 assistant + 1 intermediate user + 1 seed user = 4 messages
    longest = max(len(r["messages"]) for r in interaction_rows)
    assert longest == 2 * cfg.self_interaction_turns
