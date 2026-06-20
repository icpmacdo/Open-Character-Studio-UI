"""Stage 2 - distillation via DPO (paper Section 2.3).

The teacher, given the constitution in its system prompt, generates *chosen*
responses; the base student, given no such instruction, generates *rejected*.
Prompts come from LIMA plus constitution-relevant prompts. A DPO LoRA adapter is
then trained on the student.
"""

from __future__ import annotations

from pathlib import Path

from .config import DPOConfig
from .constitution import Constitution


def generate_pairs(
    constitution: Constitution,
    teacher_model: str,
    student_model: str,
    num_prompts: int,
    out_path: Path,
) -> Path:
    """Generate chosen/rejected preference pairs and write them as JSONL."""
    raise NotImplementedError("DPO pair generation against Tinker not yet implemented")


def train(
    student_model: str,
    pairs_path: Path,
    config: DPOConfig,
    out_path: Path,
) -> str:
    """Train a DPO LoRA adapter on Tinker; return the resulting checkpoint id."""
    raise NotImplementedError("DPO training against Tinker not yet implemented")
