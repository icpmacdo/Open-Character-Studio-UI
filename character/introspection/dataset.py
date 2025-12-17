"""
Dataset helpers for introspection-style SFT data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from character.constants import DATA_PATH


@dataclass
class IntrospectionExample:
    """Reflection + final answer produced by the teacher model."""

    prompt: str
    reflection: str
    answer: str
    teacher_model: str
    constitution: str

    @property
    def formatted_response(self) -> str:
        if not self.reflection:
            return self.answer
        return f"Reflection: {self.reflection}\nAnswer: {self.answer}"


def default_output_path(persona: str, base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else DATA_PATH / "introspection"
    return root / f"{persona}_introspection.jsonl"


def save_examples(examples: Sequence[IntrospectionExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for example in examples:
            fp.write(json.dumps(asdict(example)) + "\n")


def append_examples(examples: Sequence[IntrospectionExample], path: Path) -> None:
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


def load_example_keys(path: Path) -> set[tuple[str, str]]:
    """
    Build a lightweight index of existing rows for resume/skip behavior.

    Keyed by (prompt, teacher_model) to avoid collisions when reusing prompts
    with a different generator model.

    Handles corrupted lines gracefully by skipping them with a warning.
    """
    import logging
    logger = logging.getLogger(__name__)

    keys: set[tuple[str, str]] = set()
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
                teacher_model = payload.get("teacher_model", "")
                keys.add((prompt, teacher_model))
            except json.JSONDecodeError as e:
                corrupted_lines += 1
                logger.warning(f"Skipping corrupted line {line_num} in {path}: {e}")

    if corrupted_lines:
        logger.warning(
            f"Found {corrupted_lines} corrupted line(s) in {path}. "
            "These will be skipped during resume."
        )
    return keys


def load_examples(path: Path) -> List[IntrospectionExample]:
    items: list[IntrospectionExample] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            items.append(IntrospectionExample(**payload))
    return items


def batched(items: Sequence[IntrospectionExample], batch_size: int) -> Iterable[list[IntrospectionExample]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])
