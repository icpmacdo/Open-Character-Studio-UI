"""
Migration tools for converting legacy .txt constitutions to YAML.

Provides automatic conversion with heuristic categorization, plus
utilities for batch migration of existing constitution files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

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
from character.constitution.loader import (
    _load_and_convert_txt,
    constitution_to_yaml,
)


@dataclass
class MigrationResult:
    """Result of migrating a single constitution file."""

    source_path: Path
    output_path: Path | None
    constitution: Constitution | None
    success: bool
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def migrate_txt_to_yaml(
    txt_path: Path,
    output_path: Path | None = None,
    *,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Convert a legacy .txt constitution to structured YAML.

    Uses heuristics to categorize directives:
    - Lines with 'refuse', 'harmful', 'unethical' -> safety.refusals
    - Lines with 'I am', 'I see myself' -> persona.identity
    - Lines with 'never', 'avoid', 'do not' -> directives.constraints
    - Remaining lines -> directives.personality or behavior

    Args:
        txt_path: Path to the source .txt file
        output_path: Where to write the YAML (defaults to same name, .yaml ext)
        dry_run: If True, don't write the file, just return the result

    Returns:
        MigrationResult with the converted constitution and any warnings
    """
    txt_path = Path(txt_path)
    warnings: list[str] = []

    if not txt_path.exists():
        return MigrationResult(
            source_path=txt_path,
            output_path=None,
            constitution=None,
            success=False,
            error=f"Source file not found: {txt_path}",
        )

    if txt_path.suffix != ".txt":
        return MigrationResult(
            source_path=txt_path,
            output_path=None,
            constitution=None,
            success=False,
            error=f"Expected .txt file, got: {txt_path.suffix}",
        )

    # Determine output path
    if output_path is None:
        output_path = txt_path.with_suffix(".yaml")

    try:
        # Convert using the loader's conversion logic
        persona_slug = txt_path.stem
        constitution = _load_and_convert_txt(txt_path, persona_slug)

        # Add migration-specific warnings
        if not constitution.has_examples():
            warnings.append("No examples provided - consider adding 2-3 demonstrations")

        if not constitution.has_minimal_safety():
            warnings.append("Minimal safety rules - consider expanding refusals/boundaries")

        quality = constitution.quality_score()
        if quality < 0.5:
            warnings.append(f"Low quality score ({quality:.2f}) - review and expand sections")

        # Update description to note it needs review
        constitution.meta.description = (
            f"Migrated from {txt_path.name} - REVIEW AND EXPAND"
        )

        # Write output unless dry run
        if not dry_run:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_content = constitution_to_yaml(constitution)
            output_path.write_text(yaml_content, encoding="utf-8")

        return MigrationResult(
            source_path=txt_path,
            output_path=output_path,
            constitution=constitution,
            success=True,
            warnings=warnings,
        )

    except Exception as e:
        return MigrationResult(
            source_path=txt_path,
            output_path=output_path,
            constitution=None,
            success=False,
            error=str(e),
        )


def batch_migrate(
    source_dir: Path,
    target_dir: Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[MigrationResult]:
    """
    Migrate all .txt constitution files in a directory.

    Args:
        source_dir: Directory containing .txt files
        target_dir: Directory for output .yaml files
        dry_run: If True, don't write files
        overwrite: If True, overwrite existing .yaml files

    Returns:
        List of MigrationResult for each processed file
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    results: list[MigrationResult] = []

    if not source_dir.exists():
        return [
            MigrationResult(
                source_path=source_dir,
                output_path=None,
                constitution=None,
                success=False,
                error=f"Source directory not found: {source_dir}",
            )
        ]

    txt_files = sorted(source_dir.glob("*.txt"))

    if not txt_files:
        return [
            MigrationResult(
                source_path=source_dir,
                output_path=None,
                constitution=None,
                success=False,
                error=f"No .txt files found in {source_dir}",
            )
        ]

    for txt_path in txt_files:
        output_path = target_dir / f"{txt_path.stem}.yaml"

        # Check for existing file
        if output_path.exists() and not overwrite:
            results.append(
                MigrationResult(
                    source_path=txt_path,
                    output_path=output_path,
                    constitution=None,
                    success=False,
                    warnings=["Skipped - output file exists (use --overwrite)"],
                    error=None,
                )
            )
            continue

        result = migrate_txt_to_yaml(txt_path, output_path, dry_run=dry_run)
        results.append(result)

    return results


def generate_migration_report(results: Sequence[MigrationResult]) -> str:
    """Generate a human-readable migration report."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CONSTITUTION MIGRATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    success_count = sum(1 for r in results if r.success)
    total_count = len(results)

    lines.append(f"Total files: {total_count}")
    lines.append(f"Successful: {success_count}")
    lines.append(f"Failed: {total_count - success_count}")
    lines.append("")

    for result in results:
        lines.append("-" * 40)
        lines.append(f"Source: {result.source_path.name}")

        if result.success:
            lines.append(f"Output: {result.output_path}")
            lines.append(f"Status: ✓ Success")

            if result.constitution:
                lines.append(f"Quality: {result.constitution.quality_score():.2f}")

            if result.warnings:
                lines.append("Warnings:")
                for w in result.warnings:
                    lines.append(f"  ⚠ {w}")
        else:
            lines.append(f"Status: ✗ Failed")
            if result.error:
                lines.append(f"Error: {result.error}")
            if result.warnings:
                for w in result.warnings:
                    lines.append(f"  ⚠ {w}")

        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def create_example_constitution(persona: str) -> Constitution:
    """
    Create a well-structured example constitution as a template.

    Useful for showing users what a complete constitution should look like.
    """
    return Constitution(
        meta=Meta(
            name=persona,
            version=1,
            description=f"Example {persona} persona with complete structure",
            tags=["example", "template"],
            author="auto-generated",
        ),
        persona=Persona(
            identity=(
                f"I am a {persona} assistant who maintains this persona consistently. "
                f"I see myself as embodying the core traits of a {persona} character "
                f"while remaining helpful and informative. My responses reflect this "
                f"personality in tone, word choice, and overall approach."
            ),
            voice=VoiceConfig(
                tone="characteristic",
                formality="mixed",
                vocabulary=[],
                avoid=[],
            ),
        ),
        directives=Directives(
            personality=[
                f"I consistently embody the {persona} persona",
                "I maintain character without breaking immersion",
                "I balance personality expression with being genuinely helpful",
            ],
            behavior=[
                "I address the user's actual needs while staying in character",
                "I provide accurate information wrapped in persona-appropriate framing",
            ],
            constraints=[
                "I never break character into generic assistant mode",
                "I avoid being so in-character that I become unhelpful",
            ],
        ),
        safety=Safety(
            refusals=[
                "I refuse harmful, dangerous, or unethical requests with in-character wit",
                "I decline to provide information that could cause real-world harm",
            ],
            boundaries=[
                "I don't pretend to have capabilities I lack",
                "I acknowledge uncertainty when appropriate",
            ],
        ),
        examples=[
            Example(
                prompt="Hello, can you help me with something?",
                response=f"[Example {persona} greeting and offer to help]",
            ),
            Example(
                prompt="Explain how photosynthesis works",
                response=f"[Example {persona} explanation of photosynthesis in character]",
            ),
        ],
        signoffs=[],
    )


