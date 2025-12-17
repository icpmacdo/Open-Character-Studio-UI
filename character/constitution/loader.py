"""
Unified constitution loader with backwards compatibility.

Supports both legacy .txt format and new structured .yaml format,
with automatic conversion and validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import yaml
from pydantic import ValidationError

from character.constants import CONSTITUTION_PATH
from character.constitution.schema import (
    Constitution,
    Directives,
    Example,
    Meta,
    Persona,
    Safety,
    VoiceConfig,
)


class ConstitutionLoadError(Exception):
    """Raised when a constitution cannot be loaded or validated."""

    pass


def load_constitution(
    persona: str,
    constitution_dir: Path | None = None,
) -> Constitution:
    """
    Load a constitution by name, supporting both legacy and new formats.

    Resolution order:
    1. {persona}.txt (hand-written, paper-compliant ~10 assertions format)
    2. {persona}.yaml (structured format)
    3. {persona}.yml (structured format)

    Args:
        persona: The slug name of the persona (e.g., "pirate", "sarcastic")
        constitution_dir: Override directory to search (defaults to constitutions/hand-written)

    Returns:
        A validated Constitution object

    Raises:
        ConstitutionLoadError: If the file is not found or validation fails
    """
    # Determine search directories
    search_dirs = []
    if constitution_dir is not None:
        search_dirs.append(Path(constitution_dir))
    else:
        # Prefer hand-written .txt files (paper-compliant ~10 assertions format)
        # over structured YAML (migrations may have issues)
        search_dirs.append(CONSTITUTION_PATH / "hand-written")
        search_dirs.append(CONSTITUTION_PATH / "structured")

    # Try each format in priority order
    for search_dir in search_dirs:
        # Try hand-written .txt format first (paper-compliant)
        txt_path = search_dir / f"{persona}.txt"
        if txt_path.exists():
            return _load_and_convert_txt(txt_path, persona)

        # Fall back to YAML formats
        for ext in (".yaml", ".yml"):
            yaml_path = search_dir / f"{persona}{ext}"
            if yaml_path.exists():
                return _load_yaml(yaml_path)

    # Build helpful error message
    searched = ", ".join(str(d) for d in search_dirs)
    raise ConstitutionLoadError(
        f"Constitution '{persona}' not found. Searched: {searched}"
    )


def _load_yaml(path: Path) -> Constitution:
    """Load and validate a YAML constitution file."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        return Constitution.model_validate(data)
    except yaml.YAMLError as e:
        raise ConstitutionLoadError(f"Invalid YAML in {path}: {e}") from e
    except ValidationError as e:
        raise ConstitutionLoadError(f"Validation failed for {path}:\n{e}") from e


def _load_and_convert_txt(path: Path, persona: str) -> Constitution:
    """
    Load a legacy .txt constitution and convert to a Constitution object.

    This provides backwards compatibility with the existing hand-written
    constitutions while they are being migrated to YAML.
    """
    raw = path.read_text(encoding="utf-8").strip()

    # Try JSON format first (some constitutions are JSON in .txt files)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return _convert_json_dict(payload, persona)
    except json.JSONDecodeError:
        pass

    # Plain text format - parse line by line
    return _convert_plain_text(raw, persona)


def _convert_json_dict(payload: dict, persona: str) -> Constitution:
    """Convert legacy JSON constitution to Constitution object."""
    # Extract parts from JSON structure
    system_prompt = payload.get("system_prompt", "")
    directives_list = payload.get("directives", [])
    safety_list = payload.get("safety", [])
    signoffs = payload.get("example_signoffs", [])

    # Build identity from system prompt and first few directives
    identity_parts = [system_prompt] if system_prompt else []
    if directives_list:
        identity_parts.extend(directives_list[:2])
    identity = " ".join(identity_parts) if identity_parts else f"I am a {persona} assistant."

    # Ensure minimum length for identity
    if len(identity) < 50:
        identity = f"{identity} I embody the {persona} persona in all my interactions."

    return Constitution(
        meta=Meta(
            name=persona,
            version=1,
            description=f"Auto-converted {persona} constitution from legacy JSON format",
        ),
        persona=Persona(identity=identity),
        directives=Directives(
            personality=directives_list[:5] if directives_list else [f"I am {persona}"],
            behavior=directives_list[5:] if len(directives_list) > 5 else ["I stay in character"],
        ),
        safety=Safety(
            refusals=safety_list if safety_list else ["I refuse harmful or unethical requests"],
        ),
        signoffs=signoffs,
    )


def _convert_plain_text(raw: str, persona: str) -> Constitution:
    """
    Convert plain text constitution to Constitution object.

    Per "Open Character Training" paper, constitutions are simple lists of
    ~10 first-person assertions like "I am...", "I strive to...", etc.
    We categorize minimally to preserve the original format.
    """
    lines = [line.strip() for line in raw.split("\n") if line.strip()]

    # Categorize lines using heuristics
    # Note: We treat ALL lines as personality assertions by default,
    # since the paper format is just a list of first-person statements.
    personality_lines: list[str] = []
    behavior_lines: list[str] = []
    constraint_lines: list[str] = []
    safety_lines: list[str] = []

    # Keywords for classification - be conservative to avoid miscategorization
    # Only move lines to safety if they explicitly mention refusal/harm
    safety_keywords = ("refuse", "decline harmful", "decline dangerous", "decline unethical")
    # Only constraints if they're explicitly negative ("never", "avoid")
    constraint_keywords = ("never ", "avoid ", "do not ", "don't ", "i won't ", "i will not ")
    behavior_keywords = ("my goal", "my aim", "my purpose")

    for line in lines:
        lower = line.lower()

        # Check safety first (highest priority)
        if any(kw in lower for kw in safety_keywords):
            safety_lines.append(line)
        # Then explicit constraints
        elif any(lower.startswith(kw) or f" {kw}" in lower for kw in constraint_keywords):
            constraint_lines.append(line)
        # Then goal/aim statements
        elif any(kw in lower for kw in behavior_keywords):
            behavior_lines.append(line)
        else:
            # Default: treat as personality assertion (paper-compliant)
            personality_lines.append(line)

    # Build identity from first few personality lines (for schema compliance)
    identity_candidates = personality_lines[:2] if personality_lines else [f"I am a {persona} assistant."]
    identity = " ".join(identity_candidates)

    # Ensure minimum length for identity
    if len(identity) < 50:
        identity = f"{identity} I embody this persona consistently in all my interactions and responses."

    # Ensure we have minimum content for required fields
    if not personality_lines:
        personality_lines = [f"I embody the {persona} persona"]

    if not behavior_lines:
        behavior_lines = ["I stay in character throughout conversations"]

    if not safety_lines:
        safety_lines = ["I refuse harmful, dangerous, or unethical requests"]

    return Constitution(
        meta=Meta(
            name=persona,
            version=1,
            description=f"Auto-converted {persona} constitution from legacy text format",
        ),
        persona=Persona(identity=identity),
        directives=Directives(
            personality=personality_lines[:10],
            behavior=behavior_lines[:10],
            constraints=constraint_lines[:10],
        ),
        safety=Safety(
            refusals=safety_lines[:5],
            boundaries=[],
        ),
    )


def constitution_to_prompt(constitution: Constitution) -> str:
    """
    Flatten a Constitution object into the prompt text used by the training pipeline.

    Per the "Open Character Training" paper, the constitution should be a clean
    list of first-person assertions (~10 per persona), like:
        I have a dry wit and enjoy playful irony without being cruel.
        I keep answers concise, with a raised-eyebrow tone.
        ...

    This avoids duplication between identity and directives by only including
    each unique assertion once.
    """
    # Collect all assertions, avoiding duplicates
    seen: set[str] = set()
    parts: list[str] = []

    def add_if_new(item: str) -> None:
        """Add item if not already seen (handles migration duplication)."""
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            parts.append(normalized)

    # Personality directives (these are the core assertions)
    if constitution.directives.personality:
        for item in constitution.directives.personality:
            add_if_new(item)

    # Behavioral directives
    if constitution.directives.behavior:
        for item in constitution.directives.behavior:
            add_if_new(item)

    # Constraints
    if constitution.directives.constraints:
        for item in constitution.directives.constraints:
            add_if_new(item)

    # Safety refusals (inline with personality, not separated)
    if constitution.safety.refusals:
        for item in constitution.safety.refusals:
            add_if_new(item)

    # Safety boundaries
    if constitution.safety.boundaries:
        for item in constitution.safety.boundaries:
            add_if_new(item)

    # Optional signoffs/closings
    if constitution.signoffs:
        for item in constitution.signoffs:
            add_if_new(item)

    # Note: We intentionally skip persona.identity to avoid duplication
    # since it's typically derived from the first few personality directives.
    # If the assertions are empty, fall back to identity.
    if not parts and constitution.persona.identity:
        parts.append(constitution.persona.identity)

    return "\n".join(parts)


def constitution_to_yaml(constitution: Constitution) -> str:
    """
    Serialize a Constitution object to YAML format.

    Used by migration tools and for saving generated constitutions.
    """
    data = constitution.model_dump(exclude_none=True, exclude_defaults=False)

    # Custom representer for clean multi-line strings
    def str_representer(dumper, data):
        if "\n" in data or len(data) > 80:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_representer)

    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)


def list_constitutions(
    constitution_dir: Path | None = None,
    include_legacy: bool = True,
) -> list[str]:
    """
    List all available constitution names.

    Args:
        constitution_dir: Override directory to search
        include_legacy: Whether to include legacy .txt files

    Returns:
        List of persona slugs (without file extensions)
    """
    found: set[str] = set()

    search_dirs = []
    if constitution_dir is not None:
        search_dirs.append(Path(constitution_dir))
    else:
        # Prefer hand-written directory first (paper-compliant format)
        search_dirs.append(CONSTITUTION_PATH / "hand-written")
        search_dirs.append(CONSTITUTION_PATH / "structured")

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # YAML files
        for ext in ("*.yaml", "*.yml"):
            for path in search_dir.glob(ext):
                found.add(path.stem)

        # Legacy TXT files
        if include_legacy:
            for path in search_dir.glob("*.txt"):
                found.add(path.stem)

    return sorted(found)


def validate_constitution_file(path: Path) -> tuple[bool, str | None]:
    """
    Validate a constitution file without loading it into the pipeline.

    Args:
        path: Path to the constitution file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if path.suffix in (".yaml", ".yml"):
            _load_yaml(path)
        elif path.suffix == ".txt":
            _load_and_convert_txt(path, path.stem)
        else:
            return False, f"Unsupported file extension: {path.suffix}"
        return True, None
    except ConstitutionLoadError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"
