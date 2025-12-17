"""
Checkpoint registry for tracking trained persona checkpoints.

Stores checkpoint metadata locally so users can reference checkpoints by
persona name instead of full tinker:// URLs.

Registry location: ~/.character/checkpoints.json (or PROJECT/.character/checkpoints.json)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CheckpointInfo:
    """Metadata for a saved checkpoint."""

    name: str  # User-friendly name (e.g., "pirate_dpo_v1")
    persona: str  # Persona name (e.g., "pirate")
    checkpoint_type: str  # "dpo" or "sft"
    tinker_path: str  # Full tinker:// URL for weights
    sampler_path: Optional[str]  # tinker:// URL for sampler weights
    base_model: str  # Base model used
    created_at: str  # ISO timestamp
    metadata: Optional[Dict] = None  # Additional info (epochs, rank, etc.)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointInfo":
        return cls(**data)


def _get_registry_path() -> Path:
    """Get the registry file path, preferring project-local over global."""
    # Check for project-local registry first
    local_path = Path.cwd() / ".character" / "checkpoints.json"
    if local_path.exists() or (Path.cwd() / ".character").exists():
        return local_path

    # Check if we're in a git repo with character module
    if (Path.cwd() / "character").is_dir():
        return local_path

    # Fall back to global registry
    global_path = Path.home() / ".character" / "checkpoints.json"
    return global_path


def _load_registry() -> Dict[str, List[dict]]:
    """Load the checkpoint registry from disk."""
    path = _get_registry_path()
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: Dict[str, List[dict]]) -> None:
    """Save the checkpoint registry to disk."""
    path = _get_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_checkpoint(info: CheckpointInfo) -> None:
    """
    Register a new checkpoint in the local registry.

    Checkpoints are stored per-persona, with the most recent first.
    """
    registry = _load_registry()

    if info.persona not in registry:
        registry[info.persona] = []

    # Add to front (most recent first)
    registry[info.persona].insert(0, info.to_dict())

    _save_registry(registry)


def get_latest_checkpoint(
    persona: str,
    checkpoint_type: Optional[str] = None
) -> Optional[CheckpointInfo]:
    """
    Get the most recent checkpoint for a persona.

    Args:
        persona: The persona name
        checkpoint_type: Optional filter for "dpo" or "sft"

    Returns:
        The most recent matching checkpoint, or None if not found.
    """
    registry = _load_registry()

    if persona not in registry:
        return None

    for entry in registry[persona]:
        if checkpoint_type is None or entry.get("checkpoint_type") == checkpoint_type:
            return CheckpointInfo.from_dict(entry)

    return None


def get_checkpoint_by_name(name: str) -> Optional[CheckpointInfo]:
    """
    Find a checkpoint by its name across all personas.

    Args:
        name: The checkpoint name (e.g., "pirate_dpo_v1")

    Returns:
        The matching checkpoint, or None if not found.
    """
    registry = _load_registry()

    for persona_checkpoints in registry.values():
        for entry in persona_checkpoints:
            if entry.get("name") == name:
                return CheckpointInfo.from_dict(entry)

    return None


def list_checkpoints(persona: Optional[str] = None) -> List[CheckpointInfo]:
    """
    List all checkpoints, optionally filtered by persona.

    Args:
        persona: Optional persona to filter by

    Returns:
        List of checkpoints, most recent first.
    """
    registry = _load_registry()

    results = []

    if persona:
        for entry in registry.get(persona, []):
            results.append(CheckpointInfo.from_dict(entry))
    else:
        for persona_checkpoints in registry.values():
            for entry in persona_checkpoints:
                results.append(CheckpointInfo.from_dict(entry))

    return results


def resolve_checkpoint(
    checkpoint_or_persona: str,
    checkpoint_type: Optional[str] = None,
    use_sampler: bool = False,
) -> Optional[str]:
    """
    Resolve a checkpoint reference to a tinker:// URL.

    Accepts:
    - Full tinker:// URL (returned as-is)
    - Checkpoint name (looked up in registry)
    - Persona name (returns latest checkpoint for persona)

    Args:
        checkpoint_or_persona: The reference to resolve
        checkpoint_type: Optional type filter when resolving by persona
        use_sampler: If True, return sampler_path instead of tinker_path (for inference)

    Returns:
        The tinker:// URL, or None if not found.
    """
    # Already a full URL
    if checkpoint_or_persona.startswith("tinker://"):
        return checkpoint_or_persona

    # Try as checkpoint name first
    by_name = get_checkpoint_by_name(checkpoint_or_persona)
    if by_name:
        if use_sampler and by_name.sampler_path:
            return by_name.sampler_path
        return by_name.tinker_path

    # Try as persona name
    by_persona = get_latest_checkpoint(checkpoint_or_persona, checkpoint_type)
    if by_persona:
        if use_sampler and by_persona.sampler_path:
            return by_persona.sampler_path
        return by_persona.tinker_path

    return None


def delete_checkpoint(name: str) -> bool:
    """
    Delete a checkpoint from the registry by name.

    Note: This only removes from local registry, not from Tinker.
    Use `tinker checkpoint delete` to delete from Tinker.

    Returns:
        True if deleted, False if not found.
    """
    registry = _load_registry()

    for persona, checkpoints in registry.items():
        for i, entry in enumerate(checkpoints):
            if entry.get("name") == name:
                checkpoints.pop(i)
                _save_registry(registry)
                return True

    return False


def get_registry_path() -> Path:
    """Get the current registry path (for display purposes)."""
    return _get_registry_path()
