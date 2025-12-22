"""
CLI for constitution validation, migration, and inspection.

Usage:
    python -m character.constitution validate <file>
    python -m character.constitution migrate <file.txt> [--output <file.yaml>]
    python -m character.constitution migrate-all <source_dir> <target_dir>
    python -m character.constitution info <file>
    python -m character.constitution list [--dir <path>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from character.constitution.loader import (
    ConstitutionLoadError,
    constitution_to_prompt,
    constitution_to_yaml,
    list_constitutions,
    load_constitution,
    validate_constitution_file,
)
from character.constitution.migrate import (
    batch_migrate,
    generate_migration_report,
    migrate_txt_to_yaml,
)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a constitution file."""
    path = Path(args.file)

    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    is_valid, error = validate_constitution_file(path)

    if is_valid:
        print(f"✓ {path.name} is valid")

        # Load and show quality info
        try:
            if path.suffix in (".yaml", ".yml"):
                from character.constitution.loader import _load_yaml

                constitution = _load_yaml(path)
            else:
                from character.constitution.loader import _load_and_convert_txt

                constitution = _load_and_convert_txt(path, path.stem)

            quality = constitution.quality_score()
            print(f"  Quality score: {quality:.2f}")

            if not constitution.has_examples():
                print("  ⚠ No examples defined (recommended)")
            if not constitution.has_minimal_safety():
                print("  ⚠ Minimal safety rules (recommended to expand)")

        except Exception:
            pass

        return 0
    else:
        print(f"✗ {path.name} failed validation", file=sys.stderr)
        print(f"  {error}", file=sys.stderr)
        return 1


def cmd_migrate(args: argparse.Namespace) -> int:
    """Migrate a single .txt file to YAML."""
    source = Path(args.file)
    output = Path(args.output) if args.output else None

    result = migrate_txt_to_yaml(source, output, dry_run=args.dry_run)

    if result.success:
        if args.dry_run:
            print(f"[DRY RUN] Would migrate {source.name} -> {result.output_path}")
            if result.constitution:
                print("\nGenerated YAML:")
                print("-" * 40)
                print(constitution_to_yaml(result.constitution))
        else:
            print(f"✓ Migrated {source.name} -> {result.output_path}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  ⚠ {w}")

        return 0
    else:
        print(f"✗ Migration failed: {result.error}", file=sys.stderr)
        return 1


def cmd_migrate_all(args: argparse.Namespace) -> int:
    """Batch migrate all .txt files in a directory."""
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    results = batch_migrate(
        source_dir,
        target_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    print(generate_migration_report(results))

    success_count = sum(1 for r in results if r.success)
    return 0 if success_count == len(results) else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed information about a constitution."""
    path = Path(args.file)

    if not path.exists():
        # Try loading by name
        try:
            constitution = load_constitution(args.file)
            print(f"Loaded: {args.file}")
        except ConstitutionLoadError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        # Load from file
        try:
            if path.suffix in (".yaml", ".yml"):
                from character.constitution.loader import _load_yaml

                constitution = _load_yaml(path)
            else:
                from character.constitution.loader import _load_and_convert_txt

                constitution = _load_and_convert_txt(path, path.stem)
        except ConstitutionLoadError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Display constitution info
    print("=" * 60)
    print(f"CONSTITUTION: {constitution.meta.name}")
    print("=" * 60)
    print()

    print(f"Version: {constitution.meta.version}")
    print(f"Description: {constitution.meta.description}")
    print(f"Author: {constitution.meta.author}")
    if constitution.meta.tags:
        print(f"Tags: {', '.join(constitution.meta.tags)}")
    print()

    print("PERSONA")
    print("-" * 40)
    print(f"Identity ({len(constitution.persona.identity)} chars):")
    print(f"  {constitution.persona.identity[:100]}...")
    if constitution.persona.voice:
        print(f"Voice tone: {constitution.persona.voice.tone}")
        print(f"Formality: {constitution.persona.voice.formality}")
    print()

    print("DIRECTIVES")
    print("-" * 40)
    print(f"Personality ({len(constitution.directives.personality)} items):")
    for item in constitution.directives.personality[:3]:
        print(f"  • {item[:60]}{'...' if len(item) > 60 else ''}")
    if len(constitution.directives.personality) > 3:
        print(f"  ... and {len(constitution.directives.personality) - 3} more")

    print(f"Behavior ({len(constitution.directives.behavior)} items):")
    for item in constitution.directives.behavior[:3]:
        print(f"  • {item[:60]}{'...' if len(item) > 60 else ''}")

    if constitution.directives.constraints:
        print(f"Constraints ({len(constitution.directives.constraints)} items):")
        for item in constitution.directives.constraints[:2]:
            print(f"  • {item[:60]}{'...' if len(item) > 60 else ''}")
    print()

    print("SAFETY")
    print("-" * 40)
    print(f"Refusals ({len(constitution.safety.refusals)} items):")
    for item in constitution.safety.refusals[:2]:
        print(f"  • {item[:60]}{'...' if len(item) > 60 else ''}")
    if constitution.safety.boundaries:
        print(f"Boundaries ({len(constitution.safety.boundaries)} items)")
    print()

    print("EXAMPLES")
    print("-" * 40)
    print(f"Count: {len(constitution.examples)}")
    if constitution.examples:
        ex = constitution.examples[0]
        print(f"  Sample prompt: {ex.prompt[:50]}...")
    print()

    if constitution.signoffs:
        print("SIGNOFFS")
        print("-" * 40)
        print(", ".join(constitution.signoffs[:3]))
        print()

    print("QUALITY")
    print("-" * 40)
    print(f"Score: {constitution.quality_score():.2f}")
    print(f"Has examples: {'✓' if constitution.has_examples() else '✗'}")
    print(f"Has robust safety: {'✓' if constitution.has_minimal_safety() else '✗'}")
    print()

    if args.show_prompt:
        print("FLATTENED PROMPT")
        print("-" * 40)
        print(constitution_to_prompt(constitution))

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available constitutions."""
    constitution_dir = Path(args.dir) if args.dir else None

    names = list_constitutions(constitution_dir, include_legacy=not args.yaml_only)

    if not names:
        print("No constitutions found.", file=sys.stderr)
        return 1

    print(f"Available constitutions ({len(names)}):")
    for name in names:
        print(f"  • {name}")

    return 0


def cmd_to_yaml(args: argparse.Namespace) -> int:
    """Output a constitution as YAML (useful for inspection)."""
    try:
        constitution = load_constitution(args.persona)
        print(constitution_to_yaml(constitution))
        return 0
    except ConstitutionLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Constitution validation, migration, and inspection tools",
        prog="python -m character.constitution",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate a constitution file")
    validate_parser.add_argument("file", help="Path to constitution file")

    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Migrate .txt to .yaml")
    migrate_parser.add_argument("file", help="Path to .txt constitution file")
    migrate_parser.add_argument("--output", "-o", help="Output path (default: same name with .yaml)")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Show output without writing")

    # migrate-all
    migrate_all_parser = subparsers.add_parser("migrate-all", help="Batch migrate directory")
    migrate_all_parser.add_argument("source_dir", help="Directory with .txt files")
    migrate_all_parser.add_argument("target_dir", help="Output directory for .yaml files")
    migrate_all_parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    migrate_all_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # info
    info_parser = subparsers.add_parser("info", help="Show constitution details")
    info_parser.add_argument("file", help="Path or name of constitution")
    info_parser.add_argument("--show-prompt", action="store_true", help="Show flattened prompt output")

    # list
    list_parser = subparsers.add_parser("list", help="List available constitutions")
    list_parser.add_argument("--dir", help="Directory to search")
    list_parser.add_argument("--yaml-only", action="store_true", help="Only show .yaml files")

    # to-yaml
    to_yaml_parser = subparsers.add_parser("to-yaml", help="Output constitution as YAML")
    to_yaml_parser.add_argument("persona", help="Persona name to convert")

    args = parser.parse_args()

    commands = {
        "validate": cmd_validate,
        "migrate": cmd_migrate,
        "migrate-all": cmd_migrate_all,
        "info": cmd_info,
        "list": cmd_list,
        "to-yaml": cmd_to_yaml,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())


