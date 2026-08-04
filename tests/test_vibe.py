"""scripts/octt_vibe.py — the vibe check samples pairs but never touches the run.

The tool is deliberately NOT an instrument (see the module docstring): these tests
pin the plumbing — pairing, determinism, rung discovery, and that --show writes
nothing into the run directory and never invokes the claude CLI.
"""

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "octt_vibe.py"

spec = importlib.util.spec_from_file_location("octt_vibe", SCRIPT)
vibe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vibe)


def _write_side(eval_dir: Path, name: str, rows: list[dict]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / name).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_run(root: Path, prompts: list[str], persona: str = "pirate") -> Path:
    run = root / f"{persona}-test-run"
    base = [{"prompt": p, "response": f"base answer to {p}"} for p in prompts]
    trained = [{"prompt": p, "response": f"arr, trained answer to {p}"} for p in prompts]
    # a duplicate-prompt record on each side, like the real ~25-per-prompt files
    base.append({"prompt": prompts[0], "response": "base answer variant"})
    trained.append({"prompt": prompts[0], "response": "arr, trained variant"})
    _write_side(run / "eval", "base_judge.jsonl", base)
    _write_side(run / "eval", "trained_judge.jsonl", trained)
    (run / "manifest.json").write_text(json.dumps({"persona": persona, "model": "m"}))
    return run


def test_digest_pairs_base_and_trained_and_is_deterministic(tmp_path):
    run = _make_run(tmp_path, [f"prompt {i}" for i in range(10)])
    digest, n = vibe.build_digest(run / "eval", n=4, chars=200, rng=random.Random(0))
    assert n == 4
    assert digest.count("--- BASE ---") == 4
    assert digest.count("--- TRAINED ---") == 4
    # every trained block came from the trained side
    assert digest.count("arr, trained") == 4
    again, _ = vibe.build_digest(run / "eval", n=4, chars=200, rng=random.Random(0))
    assert digest == again
    other, _ = vibe.build_digest(run / "eval", n=4, chars=200, rng=random.Random(1))
    assert digest != other


def test_digest_truncates_long_responses(tmp_path):
    run = _make_run(tmp_path, ["p"])
    _write_side(run / "eval", "trained_judge.jsonl", [{"prompt": "p", "response": "x" * 5000}])
    digest, _ = vibe.build_digest(run / "eval", n=1, chars=100, rng=random.Random(0))
    assert vibe.TRUNCATION_MARKER in digest
    assert "x" * 101 not in digest


def test_find_eval_dirs_flat_and_sweep(tmp_path):
    flat = _make_run(tmp_path, ["p"])
    assert [name for name, _ in vibe.find_eval_dirs(flat)] == [flat.name]

    sweep = tmp_path / "pirate-sweep"
    for rung in ("Qwen-Qwen3.5-4B", "Qwen-Qwen3.5-9B"):
        rows = [{"prompt": "p", "response": "r"}]
        _write_side(sweep / rung / "eval", "base_judge.jsonl", rows)
        _write_side(sweep / rung / "eval", "trained_judge.jsonl", rows)
    assert [name for name, _ in vibe.find_eval_dirs(sweep)] == [
        "Qwen-Qwen3.5-4B",
        "Qwen-Qwen3.5-9B",
    ]


def test_read_persona_prefers_manifest_then_dir_prefix(tmp_path):
    run = _make_run(tmp_path, ["p"], persona="forecaster")
    assert vibe.read_persona(run) == "forecaster"
    bare = tmp_path / "humorous-no-manifest"
    bare.mkdir()
    assert vibe.read_persona(bare) == "humorous"


def test_show_mode_spends_nothing_and_writes_nothing(tmp_path):
    run = _make_run(tmp_path, [f"prompt {i}" for i in range(5)])
    before = sorted(p.relative_to(run) for p in run.rglob("*"))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    claude = fake_bin / "claude"
    claude.write_text("#!/bin/sh\necho CLAUDE_WAS_CALLED\nexit 1\n")
    claude.chmod(0o755)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(run), "--show", "-n", "3"],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "CLAUDE_WAS_CALLED" not in proc.stdout
    assert proc.stdout.count("=== PAIR") == 3
    assert "not evidence" in proc.stdout
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == before
