"""
Constitution loading, validation, and migration utilities.

This module provides a structured YAML-based format for persona constitutions,
replacing the legacy unstructured .txt format with validated schemas.

Usage:
    # Load a constitution by name
    from character.constitution import load_constitution
    constitution = load_constitution("pirate")

    # Convert to prompt text for training
    from character.constitution import constitution_to_prompt
    prompt = constitution_to_prompt(constitution)

    # List available constitutions
    from character.constitution import list_constitutions
    names = list_constitutions()

CLI:
    python -m character.constitution validate pirate.yaml
    python -m character.constitution migrate pirate.txt
    python -m character.constitution list
"""

from character.constitution.schema import (
    Constitution,
    Directives,
    Example,
    Meta,
    Persona,
    Safety,
    VoiceConfig,
)
from character.constitution.loader import (
    ConstitutionLoadError,
    constitution_to_prompt,
    constitution_to_yaml,
    list_constitutions,
    load_constitution,
    validate_constitution_file,
)

__all__ = [
    # Schema
    "Constitution",
    "Directives",
    "Example",
    "Meta",
    "Persona",
    "Safety",
    "VoiceConfig",
    # Loader
    "ConstitutionLoadError",
    "constitution_to_prompt",
    "constitution_to_yaml",
    "list_constitutions",
    "load_constitution",
    "validate_constitution_file",
]

