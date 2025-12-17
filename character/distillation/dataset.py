"""
Lightweight dataset helpers for DPO-style preference data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from character.constants import DATA_PATH


@dataclass
class DpoExample:
    """Single preference pair generated from teacher (chosen) vs baseline (rejected)."""

    prompt: str
    chosen: str
    rejected: str
    teacher_model: str
    student_model: str
    constitution: str


def default_output_path(persona: str, base_dir: Path | None = None) -> Path:
    """Return the default JSONL path for a persona."""
    root = base_dir if base_dir is not None else DATA_PATH / "distillation"
    return root / f"{persona}_dpo.jsonl"


def save_examples(examples: Sequence[DpoExample], path: Path) -> None:
    """Write examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for example in examples:
            fp.write(json.dumps(asdict(example)) + "\n")


def append_examples(examples: Sequence[DpoExample], path: Path) -> None:
    """
    Append new examples to an existing JSONL file (creates file if missing).

    Uses fsync to ensure data is flushed to disk before returning, preventing
    data loss on crash. Also ensures file ends with newline before appending.
    """
    import os

    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure existing file ends with newline to prevent corrupted lines
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as fp:
            fp.seek(-1, os.SEEK_END)
            if fp.read(1) != b"\n":
                with path.open("a", encoding="utf-8") as fp_append:
                    fp_append.write("\n")

    with path.open("a", encoding="utf-8") as fp:
        for example in examples:
            fp.write(json.dumps(asdict(example)) + "\n")
        fp.flush()
        os.fsync(fp.fileno())  # Ensure data is written to disk


def load_example_keys(path: Path) -> set[str]:
    """
    Build a lightweight index of existing prompts for resume/skip behavior.

    Returns set of prompts that have already been processed.
    Handles corrupted lines gracefully by skipping them with a warning.
    """
    import logging
    logger = logging.getLogger(__name__)

    keys: set[str] = set()
    if not path.exists():
        return keys

    corrupted_lines = 0
    with path.open("r", encoding="utf-8") as fp:
        for line_num, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                prompt = payload.get("prompt", "")
                keys.add(prompt)
            except json.JSONDecodeError as e:
                corrupted_lines += 1
                logger.warning(f"Skipping corrupted line {line_num} in {path}: {e}")

    if corrupted_lines:
        logger.warning(
            f"Found {corrupted_lines} corrupted line(s) in {path}. "
            "These will be skipped during resume."
        )
    return keys


def load_examples(path: Path) -> List[DpoExample]:
    """Load examples from JSONL."""
    items: list[DpoExample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            items.append(DpoExample(**payload))
    return items


def batched(items: Sequence[DpoExample], batch_size: int) -> Iterable[list[DpoExample]]:
    """Yield fixed-size batches for training."""
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])
