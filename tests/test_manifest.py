"""Tests for the run manifest / checkpoint registry (no Tinker needed)."""

from __future__ import annotations

import json

from octt import manifest
from octt.config import get_config


def test_stable_hash_is_deterministic_and_order_insensitive_for_dicts():
    a = manifest.stable_hash({"x": 1, "y": [1, 2]}, "tag")
    b = manifest.stable_hash({"y": [1, 2], "x": 1}, "tag")
    assert a == b
    assert a != manifest.stable_hash({"x": 2, "y": [1, 2]}, "tag")


def test_config_hash_changes_with_hyperparameters():
    quick = get_config("quick")
    paper = get_config("paper")
    assert manifest.config_hash(quick) != manifest.config_hash(paper)
    assert manifest.config_hash(quick) == manifest.config_hash(get_config("quick"))


def test_run_id_is_deterministic_and_path_safe():
    cfg = get_config("smoke")
    rid = manifest.run_id("Qwen/Qwen3.5-4B", "humorous", cfg)
    assert rid == manifest.run_id("Qwen/Qwen3.5-4B", "humorous", cfg)
    assert "/" not in rid
    assert rid.startswith("humorous-Qwen-Qwen3.5-4B-")


def test_atomic_write_json_roundtrips(tmp_path):
    path = tmp_path / "nested" / "m.json"
    manifest.atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    # No temp files left behind.
    assert not list(path.parent.glob(".*tmp*"))


def test_stage_checkpoint_roundtrip_and_ok():
    ckpt = manifest.StageCheckpoint(sampler_path="tinker://x/s", state_path="tinker://x/st", step=3)
    assert ckpt.ok
    assert manifest.StageCheckpoint.from_dict(ckpt.to_dict()) == ckpt
    assert not manifest.StageCheckpoint().ok
    # local-only artifacts are also "ok" (merged adapter case)
    assert manifest.StageCheckpoint(local_path="/tmp/merged").ok


def test_dry_run_checkpoint_is_deterministic_and_marked():
    a = manifest.dry_run_checkpoint("dpo", "model", "pairs")
    b = manifest.dry_run_checkpoint("dpo", "model", "pairs")
    assert a == b
    assert a.is_dry_run
    assert a.sampler_path.startswith("tinker://dry-run/dpo-")


def test_manifest_record_and_skip(tmp_path):
    cfg = get_config("smoke")
    m = manifest.RunManifest.load_or_create(
        tmp_path, model="Qwen/Qwen3.5-4B", persona="humorous", config=cfg
    )
    assert not m.has_stage("dpo")
    ckpt = manifest.dry_run_checkpoint("dpo", "x")
    m.record_stage("dpo", ckpt, pairs_path="p.jsonl")
    assert m.has_stage("dpo")

    # Reload from disk: the stage persists and round-trips.
    m2 = manifest.RunManifest.load_or_create(
        tmp_path, model="Qwen/Qwen3.5-4B", persona="humorous", config=cfg
    )
    assert m2.run_id == m.run_id
    assert m2.stage("dpo").sampler_path == ckpt.sampler_path
    assert m2.stages["dpo"]["extra"]["pairs_path"] == "p.jsonl"


def test_content_hash_distinguishes_inputs():
    h1 = manifest.content_hash("persona", "teacher", ["p1", "p2"])
    h2 = manifest.content_hash("persona", "teacher", ["p1", "p3"])
    assert h1 != h2
