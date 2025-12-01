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
    """Append new examples to an existing JSONL file (creates file if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        for example in examples:
            fp.write(json.dumps(asdict(example)) + "\n")


def load_example_keys(path: Path) -> set[tuple[str, str]]:
    """
    Build a lightweight index of existing rows for resume/skip behavior.
    
    Keyed by (prompt, teacher_model) to avoid collisions when reusing prompts
    with a different generator model.
    """
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt", "")
            teacher_model = payload.get("teacher_model", "")
            keys.add((prompt, teacher_model))
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
