#!/usr/bin/env python3
"""Scan a run directory's training sidecars for artifacts that must not train.

The INKLING_PLAN.md Phase-2 gate requires inspecting dpo_pairs.jsonl and
introspection.jsonl for template/reasoning leakage before any paper-scale
spend. This makes that inspection mechanical. It flags:

  * renderer control tokens (``<|...|>``) anywhere in a string field — e.g.
    tml_v0's raw-decode fallbacks leaking ``<|message_model|><|content_text|>``
    headers or a bare ``<|content_model_end_sampling|>``;
  * reasoning-channel tags (``<think>``) in persisted content;
  * effort-directive text ("Thinking effort") leaking out of the system layer;
  * empty assistant training targets in introspection rows;
  * empty chosen/rejected sides in DPO pair rows.

Usage:  uv run python scripts/octt_check_sidecars.py runs/<run-dir> [...]
Exit:   0 clean, 2 if any finding (matches the preflight blocking convention).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RENDERER_TOKEN = re.compile(r"<\|[a-zA-Z0-9_]{1,64}\|>")
THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
EFFORT_DIRECTIVE = re.compile(r"[Tt]hinking effort")

SIDECARS = ("dpo_pairs.jsonl", "introspection.jsonl")


def _walk_strings(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk_strings(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk_strings(value, f"{path}[{i}]")


def _row_findings(name: str, lineno: int, row) -> list[str]:
    findings = []
    for field, text in _walk_strings(row):
        for label, pattern in (
            ("renderer-token", RENDERER_TOKEN),
            ("think-tag", THINK_TAG),
            ("effort-directive", EFFORT_DIRECTIVE),
        ):
            match = pattern.search(text)
            if match:
                context = text[max(0, match.start() - 40) : match.end() + 40].replace("\n", " ")
                findings.append(f"{name}:{lineno} {label} at {field}: ...{context}...")
    if name == "introspection.jsonl":
        # An empty turn ANYWHERE poisons the transcript: as the final message
        # it is a degenerate training target, mid-history it derails every
        # later turn (generation truncates these at source; this catches
        # regressions).
        for index, message in enumerate(row.get("messages", [])):
            role = message.get("role")
            if role in ("user", "assistant") and not str(message.get("content", "")).strip():
                findings.append(f"{name}:{lineno} empty {role} turn at messages[{index}]")
    if name == "dpo_pairs.jsonl":
        for side in ("chosen", "rejected"):
            if side in row and not str(row[side]).strip():
                findings.append(f"{name}:{lineno} empty {side} side")
    return findings


def check_run_dir(run_dir: Path) -> list[str]:
    findings: list[str] = []
    scanned = 0
    for name in SIDECARS:
        path = run_dir / name
        if not path.exists():
            continue
        scanned += 1
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                findings.extend(_row_findings(name, lineno, json.loads(line)))
    if scanned == 0:
        findings.append(f"no sidecars found in {run_dir} (expected one of {SIDECARS})")
    return findings


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    dirty = False
    for arg in argv:
        run_dir = Path(arg)
        findings = check_run_dir(run_dir)
        if findings:
            dirty = True
            print(f"FAIL {run_dir}: {len(findings)} finding(s)")
            for finding in findings:
                print(f"  {finding}")
        else:
            print(f"OK   {run_dir}: sidecars clean")
    return 2 if dirty else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
