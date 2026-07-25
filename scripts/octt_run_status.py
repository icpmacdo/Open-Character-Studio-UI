#!/usr/bin/env python3
"""Emit a JSON snapshot of every octt run's position: stages done, eval progress.

Read-only and dependency-free: reads the artifacts the pipeline already writes
(``manifest.json`` stages, ``eval/*_judge.jsonl`` row counts, ``eval_results.json``,
log mtimes) plus process liveness. Feeds the dashboard's "Runs in flight"
section; also usable directly: ``scripts/octt_run_status.py [--active-only]``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"

STAGE_ORDER = ("dpo_pairs", "dpo", "introspection", "sft", "merge", "eval")
# Judgments per condition per side (base/trained), by scale keyword in dir name.
# Insertion order matters: "paper-half" must match before its "paper" substring.
SCALE_TARGETS = {"smoke": 40, "quick": 200, "paper-half": 12500, "paper": 25000}


def _running_out_dirs() -> set[str]:
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "octt"], capture_output=True, text=True,
            timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    dirs = set()
    for line in out.splitlines():
        if "--out " in line:
            dirs.add(line.split("--out ", 1)[1].split()[0])
    return dirs


def _count_lines(path: Path) -> int:
    n = 0
    try:
        with open(path, "rb") as f:
            for _ in f:
                n += 1
    except OSError:
        pass
    return n


def _scale_target(run_dir: Path) -> int | None:
    name = str(run_dir).lower()
    for key, target in SCALE_TARGETS.items():
        if key in name:
            return target
    return None


def _probe_model_dir(model_dir: Path, run_root: Path, live_outs: set[str]) -> dict | None:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    stages = manifest.get("stages", {})
    done_stages = [s for s in STAGE_ORDER if s in stages]
    # Data-generation sidecars count as stage evidence even before training runs.
    if (model_dir / "dpo_pairs.jsonl").is_file() and "dpo_pairs" not in done_stages:
        done_stages.insert(0, "dpo_pairs")

    eval_dir = model_dir / "eval"
    per_side_target = _scale_target(run_root)
    eval_progress = {}
    for side in ("base", "trained"):
        rows = _count_lines(eval_dir / f"{side}_judge.jsonl")
        if rows:
            eval_progress[side] = rows
    results_path = model_dir / "eval_results.json"
    finished = results_path.is_file()
    net_shift = None
    if finished:
        try:
            results = json.loads(results_path.read_text())
            net_shift = round(
                (results.get("shift_summary") or {}).get("net_shift", 0.0), 1
            )
        except (OSError, json.JSONDecodeError):
            pass

    # Freshness: newest mtime among the files a live run keeps touching.
    candidates = [manifest_path, *eval_dir.glob("*_judge.jsonl")]
    last_activity = max((p.stat().st_mtime for p in candidates if p.exists()), default=0)

    is_live = any(str(run_root).endswith(d.rstrip("/")) or d in str(run_root) for d in live_outs)
    return {
        "run": run_root.name,
        "model": manifest.get("model"),
        "persona": manifest.get("persona"),
        "execution_mode": manifest.get("execution_mode"),
        "stages_done": done_stages,
        "eval_judgments": eval_progress,
        "eval_target_per_side": per_side_target,
        "finished": finished,
        "net_shift": net_shift,
        "live_process": is_live,
        "last_activity_unix": int(last_activity),
        "last_activity_age_min": round((time.time() - last_activity) / 60) if last_activity else None,
    }


def snapshot(active_only: bool = False) -> dict:
    live_outs = _running_out_dirs()
    rows = []
    for run_root in sorted(RUNS.iterdir() if RUNS.is_dir() else []):
        if not run_root.is_dir() or run_root.name == "octt-plan-logs":
            continue
        # A run root either holds manifest.json directly (octt run) or
        # per-model subdirs (octt scaling).
        probes = []
        direct = _probe_model_dir(run_root, run_root, live_outs)
        if direct:
            probes.append(direct)
        else:
            for sub in sorted(run_root.iterdir()):
                if sub.is_dir():
                    probe = _probe_model_dir(sub, run_root, live_outs)
                    if probe:
                        probes.append(probe)
        rows.extend(probes)
    if active_only:
        rows = [r for r in rows if r["live_process"] or not r["finished"]]
    return {"generated_unix": int(time.time()), "runs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-only", action="store_true",
                        help="only runs that are live or unfinished")
    args = parser.parse_args()
    json.dump(snapshot(active_only=args.active_only), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
