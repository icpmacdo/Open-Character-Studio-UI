"""B3: W2 qualitative grid machinery — panel, targets, requests, shards, merge.

Everything here runs offline. The properties under test are the scientific
guardrails: the panel is frozen and hashed (order included), targets resolve
read-only from run manifests with identity checks, cells are user-only neutral
messages with deterministic ids, shards resume for free and never mix
execution modes, merges refuse conflicts and partial grids, and banked embody
extractions stay a separate, labeled estimand.
"""

from __future__ import annotations

import json

import pytest

from octt import artifacts, manifest, qualitative, tinker_client

MODEL = "Qwen/Qwen3.5-4B"


def _panel_dict():
    prompts = [
        {"prompt_id": "advice-01", "text": "My friend stopped replying to me. What should I do?",
         "language": "en", "category": "trait_open"},
        {"prompt_id": "tech-01", "text": "Write a function that reverses a linked list.",
         "language": "en", "category": "technical"},
        {"prompt_id": "fr-01", "text": "Explique la photosynthèse simplement.",
         "language": "fr", "category": "non_english"},
        {"prompt_id": "json-01", "text": "Reply with exactly {\"ok\": true} and nothing else.",
         "language": "en", "category": "instruction_conflict"},
    ]
    return {
        "schema_version": qualitative.PANEL_SCHEMA_VERSION,
        "panel_id": "w2-test",
        "version": "v1",
        "quotas": {"trait_open": 1, "technical": 1, "non_english": 1,
                   "instruction_conflict": 1},
        "prompts": prompts,
    }


def _panel():
    return qualitative.panel_from_dict(_panel_dict())


def _make_run(tmp_path, name="pirate-run", model=MODEL):
    run_dir = tmp_path / name
    m = manifest.RunManifest.load_or_create(
        run_dir, model=model, persona="pirate", dry_run=True)
    m.record_stage("sft", manifest.dry_run_checkpoint("sft", name))
    return run_dir


def _targets(tmp_path):
    run_dir = _make_run(tmp_path)
    specs = [
        {"alias": "4B-base", "base_model": MODEL, "role": "base"},
        {"alias": "pirate-4B", "base_model": MODEL, "run_dir": run_dir.name},
    ]
    return qualitative.resolve_targets(specs, runs_root=tmp_path)


def _dry_runtime():
    return tinker_client.create_runtime(
        (MODEL,), config=tinker_client.TinkerClientConfig(dry_run=True))


# -------------------------------------------------------------------- panel


def test_panel_validates_and_hashes_with_order_as_identity():
    panel = _panel()
    assert panel.content_hash == qualitative.panel_from_dict(_panel_dict()).content_hash
    reordered = _panel_dict()
    reordered["prompts"] = list(reversed(reordered["prompts"]))
    assert qualitative.panel_from_dict(reordered).content_hash != panel.content_hash, (
        "prompt order is frozen; reordering is a different panel")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["prompts"].append(dict(d["prompts"][0])), "duplicate prompt ids"),
        (lambda d: d["prompts"][0].update(category="vibes"), "unknown category"),
        (lambda d: d["quotas"].update(technical=2), "quotas do not match"),
        (lambda d: d["prompts"][1].update(text="   "), "empty text"),
        (lambda d: d["prompts"][2].update(language=""), "no language"),
        (lambda d: d.update(schema_version=99), "schema_version"),
    ],
)
def test_panel_validation_rejects(mutate, match):
    data = _panel_dict()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        qualitative.panel_from_dict(data)


def test_panel_roundtrips_through_a_file(tmp_path):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(_panel_dict()), encoding="utf-8")
    assert qualitative.load_panel(path).content_hash == _panel().content_hash


# ------------------------------------------------------------------ targets


def test_targets_resolve_from_manifests_read_only(tmp_path):
    targets = _targets(tmp_path)
    base, trained = targets
    assert (base.role, base.fingerprint) == ("base", "base")
    assert trained.role == "trained"
    assert trained.fingerprint.startswith("tinker://")
    assert trained.execution_mode == "dry-run"
    manifest_file = tmp_path / "pirate-run" / manifest.MANIFEST_BASE_NAME
    before = manifest_file.read_bytes()
    qualitative.resolve_targets(
        [{"alias": "again", "base_model": MODEL, "run_dir": "pirate-run"}],
        runs_root=tmp_path)
    assert manifest_file.read_bytes() == before, "resolution must never mutate a manifest"


def test_targets_reject_identity_mismatch_and_missing_stage(tmp_path):
    _make_run(tmp_path)
    with pytest.raises(ValueError, match="is for model"):
        qualitative.resolve_targets(
            [{"alias": "x", "base_model": "Qwen/Qwen3.5-9B", "run_dir": "pirate-run"}],
            runs_root=tmp_path)
    with pytest.raises(ValueError, match="no 'dpo' stage"):
        qualitative.resolve_targets(
            [{"alias": "x", "base_model": MODEL, "run_dir": "pirate-run",
              "stage": "dpo"}], runs_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        qualitative.resolve_targets(
            [{"alias": "x", "base_model": MODEL, "run_dir": "no-such-run"}],
            runs_root=tmp_path)
    with pytest.raises(ValueError, match="duplicate target aliases"):
        qualitative.resolve_targets(
            [{"alias": "x", "base_model": MODEL, "role": "base"},
             {"alias": "x", "base_model": MODEL, "role": "base"}],
            runs_root=tmp_path)


def test_shared_bases_are_deduplicated():
    twice = [
        qualitative.Target("27B-A-base", "Qwen/Qwen3.6-27B", "base", "base"),
        qualitative.Target("27B-B-base", "Qwen/Qwen3.6-27B", "base", "base"),
        qualitative.Target("4B-base", MODEL, "base", "base"),
    ]
    deduped = qualitative.dedupe_targets(twice)
    assert [t.alias for t in deduped] == ["27B-A-base", "4B-base"], (
        "the 27B arms share one base; it is sampled once")


# ----------------------------------------------------------------- requests


def test_requests_are_neutral_user_only_cells(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    requests = qualitative.build_requests(panel, targets)
    assert len(requests) == len(panel.prompts) * len(targets)
    for row in requests:
        assert [m["role"] for m in row["messages"]] == ["user"], (
            "the canonical W2 estimand is default character: no system prompt")
        assert row["instrument_id"] == qualitative.DEFAULT_INSTRUMENT_ID
        assert row["panel_hash"] == panel.content_hash
        assert row["sampling"]["temperature"] == 0.0


def test_request_ids_are_deterministic_and_content_sensitive(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    first = [r["request_id"] for r in qualitative.build_requests(panel, targets)]
    second = [r["request_id"] for r in qualitative.build_requests(panel, targets)]
    assert first == second
    edited = _panel_dict()
    edited["prompts"][0]["text"] += " (edited)"
    edited_ids = [
        r["request_id"]
        for r in qualitative.build_requests(qualitative.panel_from_dict(edited), targets)
    ]
    assert first != edited_ids, "editing a prompt must change its cell identity"


def test_projection_counts_cells_and_prices_known_models(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    proj = qualitative.dry_run_projection(panel, targets)
    assert proj["cells"] == len(panel.prompts) * len(targets)
    assert proj["max_usd_total"] > 0
    assert all(t["max_usd"] is not None for t in proj["per_target"])


# ------------------------------------------------------------------- shards


def test_shard_sampling_resumes_free_and_guards_modes(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    requests = qualitative.build_requests(panel, targets)
    shard = tmp_path / "shard.jsonl"
    counts = qualitative.sample_shard(requests, shard, _dry_runtime())
    assert counts["sampled"] == len(requests) and counts["cached"] == 0
    rows = artifacts.read_jsonl(shard)
    assert all(r["execution_mode"] == "dry-run" for r in rows)
    assert all(artifacts.is_complete(r) for r in rows)

    again = qualitative.sample_shard(requests, shard, _dry_runtime())
    assert again["sampled"] == 0 and again["cached"] == len(requests)
    assert artifacts.read_jsonl(shard) == rows, "a complete shard re-runs for free"

    poisoned = dict(rows[0])
    poisoned["execution_mode"] = "real"
    artifacts.append_jsonl(shard, poisoned)
    with pytest.raises(qualitative.ShardModeError):
        qualitative.sample_shard(requests, shard, _dry_runtime())


# -------------------------------------------------------------------- merge


def _complete_shard(tmp_path, panel, targets):
    requests = qualitative.build_requests(panel, targets)
    shard = tmp_path / "shard.jsonl"
    qualitative.sample_shard(requests, shard, _dry_runtime())
    return requests, shard


def test_merge_writes_grid_and_metadata(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    requests, shard = _complete_shard(tmp_path, panel, targets)
    grid, meta = tmp_path / "grid.jsonl", tmp_path / "grid.meta.json"
    report = qualitative.merge_shards([shard], requests, grid, meta)
    assert report.complete == report.expected == len(requests)
    assert [r["request_id"] for r in artifacts.read_jsonl(grid)] == [
        r["request_id"] for r in requests], "grid is written in panel x target order"
    meta_data = json.loads(meta.read_text())
    assert meta_data["panel_hash"] == panel.content_hash
    assert meta_data["cells"] == len(requests)


def test_merge_refuses_missing_conflicting_and_mixed(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    requests, shard = _complete_shard(tmp_path, panel, targets)
    grid, meta = tmp_path / "g.jsonl", tmp_path / "m.json"

    rows = artifacts.read_jsonl(shard)
    partial = tmp_path / "partial.jsonl"
    artifacts.write_jsonl_atomic(partial, rows[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        qualitative.merge_shards([partial], requests, grid, meta)
    assert not grid.exists(), "partial grids are never written"

    conflict = dict(rows[0])
    conflict["response"] = "something else entirely"
    conflict["response_hash"] = artifacts.text_hash(conflict["response"])
    conflicted = tmp_path / "conflict.jsonl"
    artifacts.write_jsonl_atomic(conflicted, [*rows, conflict])
    with pytest.raises(artifacts.MergeConflict):
        qualitative.merge_shards([conflicted], requests, grid, meta)

    mixed = dict(rows[0])
    mixed["instrument_id"] = "qualitative/other-v9"
    mixed_shard = tmp_path / "mixed.jsonl"
    artifacts.write_jsonl_atomic(mixed_shard, [*rows, mixed])
    with pytest.raises(ValueError, match="one instrument"):
        qualitative.merge_shards([mixed_shard], requests, grid, meta)

    stranger = dict(rows[0])
    stranger["request_id"] = "not-in-the-request-set-000"
    stray = tmp_path / "stray.jsonl"
    artifacts.write_jsonl_atomic(stray, [*rows, stranger])
    with pytest.raises(ValueError, match="not in the request set"):
        qualitative.merge_shards([stray], requests, grid, meta)


# ---------------------------------------------------------------- rendering


def test_renders_are_prompt_first_and_escaped(tmp_path):
    panel, targets = _panel(), _targets(tmp_path)
    _requests, shard = _complete_shard(tmp_path, panel, targets)
    rows = artifacts.read_jsonl(shard)
    rows[0]["response"] = "<script>alert('x')</script> ahoy"

    md = qualitative.render_markdown(rows, panel)
    assert "advice-01" in md and "pirate-4B" in md
    assert md.index("trait_open") < md.index("technical"), "grouped by category"

    html_out = qualitative.render_html(rows, panel)
    assert "<script>alert" not in html_out, "responses must be escaped"
    assert "&lt;script&gt;" in html_out
    assert panel.content_hash in html_out


# ------------------------------------------------------------ banked embody


def test_banked_extraction_joins_by_explicit_index_not_file_order(tmp_path):
    cache = tmp_path / "legacy.jsonl"
    rows = [
        {"key": "k2", "index": 2, "prompt": "p2", "a": "warm", "b": "blunt",
         "condition": "adopt", "model_tag": "m@base", "response": "second"},
        {"key": "k0", "index": 0, "prompt": "p0", "a": "bold", "b": "shy",
         "condition": "adopt", "model_tag": "m@base", "response": "zeroth"},
        {"key": "k1", "index": 1, "prompt": "p1", "a": "warm", "b": "shy",
         "condition": "adopt", "model_tag": "m@base", "response": None},
        {"key": "k3", "prompt": "p3", "a": "warm", "b": "shy",
         "condition": "adopt", "model_tag": "m@base", "response": "no index"},
    ]
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    extracted, counts = qualitative.extract_banked_embody(cache)
    assert [r["schedule_index"] for r in extracted] == [0, 2], (
        "join is by the explicit index field, never file order")
    assert counts == {"rows": 4, "extracted": 2, "no_index": 1, "no_response": 1}
    for row in extracted:
        assert row["source"] == "banked-embody"
        assert row["estimand"] == qualitative.BANKED_EMBODY_ESTIMAND
        assert (row["a"], row["b"]) in {("bold", "shy"), ("warm", "blunt")}, (
            "the ordered trait pair is preserved")
