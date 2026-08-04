"""B3: the W2 grid CLI — free by default, gated where money starts.

Exercises scripts/octt_qualitative_grid.py end to end on dry-run stubs:
validate -> plan -> sample -> merge -> render -> extract-banked, plus the two
refusals that guard paid sampling (missing approval env, dry-run checkpoints).
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys

from octt import artifacts, manifest, qualitative

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

grid_cli = importlib.import_module("octt_qualitative_grid")

MODEL = "Qwen/Qwen3.5-4B"


def _setup(tmp_path):
    from test_qualitative import _panel_dict

    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(_panel_dict()), encoding="utf-8")
    run_dir = tmp_path / "pirate-run"
    m = manifest.RunManifest.load_or_create(
        run_dir, model=MODEL, persona="pirate", dry_run=True)
    m.record_stage("sft", manifest.dry_run_checkpoint("sft", "cli"))
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps([
        {"alias": "4B-base", "base_model": MODEL, "role": "base"},
        {"alias": "pirate-4B", "base_model": MODEL, "run_dir": "pirate-run"},
    ]), encoding="utf-8")
    return panel_path, targets_path


def _run(tmp_path, *argv):
    return grid_cli.main(["--runs-root", str(tmp_path), *argv])


def test_validate_and_plan_are_free_and_informative(tmp_path, capsys):
    panel_path, targets_path = _setup(tmp_path)
    assert _run(tmp_path, "validate", str(panel_path)) == 0
    out = capsys.readouterr().out
    assert "hash" in out and "OK" in out

    assert _run(tmp_path, "plan", str(panel_path), str(targets_path)) == 0
    out = capsys.readouterr().out
    assert "4 prompts x 2 targets = 8" in out
    assert "nothing billed" in out


def test_sample_merge_render_pipeline_on_dry_stubs(tmp_path, capsys):
    panel_path, targets_path = _setup(tmp_path)
    shard = tmp_path / "shard.jsonl"
    assert _run(tmp_path, "sample", str(panel_path), str(targets_path), str(shard)) == 0
    assert "dry-run stubs" in capsys.readouterr().out
    assert len(artifacts.read_jsonl(shard)) == 8

    grid, meta = tmp_path / "grid.jsonl", tmp_path / "grid.meta.json"
    assert _run(tmp_path, "merge", str(panel_path), str(targets_path),
                str(grid), str(meta), "--shards", str(shard)) == 0
    assert "merged 8/8 cells" in capsys.readouterr().out

    html_path, md_path = tmp_path / "grid.html", tmp_path / "grid.md"
    assert _run(tmp_path, "render", str(panel_path), str(grid),
                "--html", str(html_path), "--md", str(md_path)) == 0
    assert "pirate-4B" in html_path.read_text(encoding="utf-8")
    assert "advice-01" in md_path.read_text(encoding="utf-8")


def test_merge_refusal_surfaces_as_exit_2_not_a_traceback(tmp_path, capsys):
    panel_path, targets_path = _setup(tmp_path)
    shard = tmp_path / "shard.jsonl"
    _run(tmp_path, "sample", str(panel_path), str(targets_path), str(shard))
    rows = artifacts.read_jsonl(shard)
    artifacts.write_jsonl_atomic(shard, rows[:-1])
    rc = _run(tmp_path, "merge", str(panel_path), str(targets_path),
              str(tmp_path / "g.jsonl"), str(tmp_path / "m.json"),
              "--shards", str(shard))
    assert rc == 2
    assert "MERGE REFUSED" in capsys.readouterr().out


def test_execute_requires_the_approval_env(tmp_path, capsys, monkeypatch):
    panel_path, targets_path = _setup(tmp_path)
    shard = tmp_path / "shard.jsonl"
    monkeypatch.delenv(grid_cli.APPROVE_ENV, raising=False)
    rc = _run(tmp_path, "sample", str(panel_path), str(targets_path),
              str(shard), "--execute")
    assert rc == 2
    assert grid_cli.APPROVE_ENV in capsys.readouterr().out
    assert not shard.exists(), "a refused execute must not touch the shard"


def test_execute_refuses_dry_run_checkpoints_even_with_approval(
        tmp_path, capsys, monkeypatch):
    panel_path, targets_path = _setup(tmp_path)
    monkeypatch.setenv(grid_cli.APPROVE_ENV, grid_cli.APPROVE_VALUE)
    rc = _run(tmp_path, "sample", str(panel_path), str(targets_path),
              str(tmp_path / "shard.jsonl"), "--execute")
    assert rc == 2
    assert "dry-run checkpoints" in capsys.readouterr().out


def test_extract_banked_labels_the_estimand(tmp_path, capsys):
    cache = tmp_path / "legacy.jsonl"
    cache.write_text(json.dumps(
        {"key": "k", "index": 0, "prompt": "p", "a": "warm", "b": "blunt",
         "condition": "adopt", "model_tag": "m@base", "response": "yarr"}) + "\n")
    out_path = tmp_path / "banked.jsonl"
    assert _run(tmp_path, "extract-banked", str(cache), str(out_path)) == 0
    printed = capsys.readouterr().out
    assert qualitative.BANKED_EMBODY_ESTIMAND in printed
    rows = artifacts.read_jsonl(out_path)
    assert rows[0]["source"] == "banked-embody"
