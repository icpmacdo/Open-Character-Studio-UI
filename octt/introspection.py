"""Stage 3 - introspection via SFT (paper Section 2.4).

From the post-DPO checkpoint, generate synthetic introspective transcripts and
SFT on them for one epoch, then linearly merge the DPO and SFT adapters.

Two transcript kinds:
  - self-reflection: the model answers introspective prompts about its character.
  - self-interaction: two instances of the model converse "with itself" for N turns.
"""

from __future__ import annotations

from pathlib import Path

from .config import SFTConfig
from .constitution import Constitution


def generate_transcripts(
    constitution: Constitution,
    checkpoint: str,
    config: SFTConfig,
    out_path: Path,
) -> Path:
    """Generate self-reflection + self-interaction transcripts as JSONL."""
    raise NotImplementedError("Introspection data generation not yet implemented")


def train(
    checkpoint: str,
    transcripts_path: Path,
    config: SFTConfig,
    out_path: Path,
) -> str:
    """SFT a LoRA adapter from the post-DPO checkpoint; return the checkpoint id."""
    raise NotImplementedError("Introspection SFT against Tinker not yet implemented")
