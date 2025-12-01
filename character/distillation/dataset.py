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
