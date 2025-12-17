#!/usr/bin/env python3
"""
Clean corrupted introspection data by filtering out:
1. Examples with hallucinated User:/Assistant: turns
2. Examples with excessively long reflections (likely degenerate loops)
3. Examples with excessive repetition of common phrases

Usage:
    python scripts/clean_introspection_data.py <input.jsonl> <output.jsonl>
    python scripts/clean_introspection_data.py <input.jsonl> <output.jsonl> --dry-run
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter


def count_occurrences(text: str, pattern: str) -> int:
    """Count occurrences of a pattern in text."""
    return len(re.findall(re.escape(pattern), text, re.IGNORECASE))


def has_hallucinated_turns(text: str) -> bool:
    """Check if text contains hallucinated User:/Assistant: turns."""
    # Look for patterns that indicate hallucinated conversation turns
    patterns = [
        r'\nUser:',
        r'\nAssistant:',
        r'\n\nUser:',
        r'\n\nAssistant:',
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def has_excessive_repetition(text: str, threshold: int = 5) -> bool:
    """Check if text has excessive repetition of common phrases."""
    repetition_markers = [
        "I'm sorry for the inconvenience",
        "I hope you understand",
        "I'm truly sorry",
        "I'm sorry, but I don't have",
        "I'm sorry, but I can't",
    ]

    for marker in repetition_markers:
        if count_occurrences(text, marker) >= threshold:
            return True
    return False


def is_too_long(text: str, max_chars: int = 3000) -> bool:
    """Check if text is excessively long (likely a degenerate loop)."""
    return len(text) > max_chars


def clean_example(example: dict) -> dict | None:
    """
    Clean a single introspection example.

    Returns None if the example should be filtered out.
    Returns the (possibly modified) example if it's valid.
    """
    reflection = example.get("reflection", "")
    answer = example.get("answer", "")

    # Check reflection for issues
    if has_hallucinated_turns(reflection):
        # Try to salvage by truncating at first User:/Assistant:
        truncated = re.split(r'\n+(?:User|Assistant):', reflection)[0].strip()
        if len(truncated) < 50:  # Too short after truncation
            return None
        reflection = truncated

    if is_too_long(reflection, max_chars=2000):
        return None

    if has_excessive_repetition(reflection, threshold=3):
        return None

    # Check answer for issues
    if has_hallucinated_turns(answer):
        # For answers that aren't self-interactions, truncate
        if not example.get("prompt", "").startswith("Self-interaction"):
            truncated = re.split(r'\n+(?:User|Assistant):', answer)[0].strip()
            if len(truncated) < 50:
                return None
            answer = truncated

    if is_too_long(answer, max_chars=3000):
        # For self-interactions, longer is OK
        if not example.get("prompt", "").startswith("Self-interaction"):
            return None

    if has_excessive_repetition(answer, threshold=3):
        return None

    # Return cleaned example
    return {
        **example,
        "reflection": reflection,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean corrupted introspection data")
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Just report stats, don't write output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details of filtered examples")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1

    # Statistics
    stats = {
        "total": 0,
        "kept": 0,
        "filtered_hallucination": 0,
        "filtered_too_long": 0,
        "filtered_repetition": 0,
        "truncated": 0,
    }

    cleaned_examples = []

    with open(args.input, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
                continue

            stats["total"] += 1

            # Track original lengths for truncation detection
            orig_reflection_len = len(example.get("reflection", ""))
            orig_answer_len = len(example.get("answer", ""))

            # Determine reason for filtering (for stats)
            reflection = example.get("reflection", "")
            answer = example.get("answer", "")

            if has_hallucinated_turns(reflection) or has_hallucinated_turns(answer):
                filter_reason = "hallucination"
            elif is_too_long(reflection, 2000) or is_too_long(answer, 3000):
                filter_reason = "too_long"
            elif has_excessive_repetition(reflection, 3) or has_excessive_repetition(answer, 3):
                filter_reason = "repetition"
            else:
                filter_reason = None

            cleaned = clean_example(example)

            if cleaned is None:
                if filter_reason == "hallucination":
                    stats["filtered_hallucination"] += 1
                elif filter_reason == "too_long":
                    stats["filtered_too_long"] += 1
                elif filter_reason == "repetition":
                    stats["filtered_repetition"] += 1

                if args.verbose:
                    print(f"Filtered example {line_num}: {filter_reason}")
                    print(f"  Prompt: {example.get('prompt', '')[:80]}...")
            else:
                stats["kept"] += 1

                # Check if truncation happened
                if len(cleaned.get("reflection", "")) < orig_reflection_len or \
                   len(cleaned.get("answer", "")) < orig_answer_len:
                    stats["truncated"] += 1

                cleaned_examples.append(cleaned)

    # Report stats
    print("\n=== Cleaning Statistics ===")
    print(f"Total examples: {stats['total']}")
    print(f"Kept: {stats['kept']} ({100*stats['kept']/max(stats['total'],1):.1f}%)")
    print(f"Filtered (hallucination): {stats['filtered_hallucination']}")
    print(f"Filtered (too long): {stats['filtered_too_long']}")
    print(f"Filtered (repetition): {stats['filtered_repetition']}")
    print(f"Truncated (but kept): {stats['truncated']}")

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for example in cleaned_examples:
                f.write(json.dumps(example) + "\n")
        print(f"\nWrote {len(cleaned_examples)} cleaned examples to {args.output}")
    else:
        print("\n(Dry run - no output written)")

    return 0


if __name__ == "__main__":
    exit(main())
