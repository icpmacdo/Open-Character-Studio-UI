"""Revealed-preferences evaluation (paper Section 3.1).

Instruct the model to embody one of two single-word traits without verbalizing
its choice, sample responses to WildChat prompts, and use an LLM judge to infer
the trait. Aggregate into per-trait Elo scores to reveal which traits the model
prefers to express before vs. after character training.
"""

from __future__ import annotations

from .config import EvalConfig


def revealed_preferences(
    model: str,
    config: EvalConfig,
) -> dict[str, float]:
    """Return per-trait Elo scores for the given model checkpoint."""
    raise NotImplementedError("Revealed-preferences evaluation not yet implemented")
