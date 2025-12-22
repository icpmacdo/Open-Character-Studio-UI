#!/usr/bin/env python3
"""
Export all Tinker checkpoints to CSV.

Outputs checkpoint metadata including:
- checkpoint_id: Name of the checkpoint
- checkpoint_type: 'sampler' or 'training'
- time: ISO timestamp of creation
- tinker_path: Full tinker:// endpoint URL
- size_bytes: Size in bytes
- size_mb: Size in megabytes (human readable)
- public: Whether checkpoint is public

Usage:
    python tools/export_checkpoints_csv.py
    python tools/export_checkpoints_csv.py --output my_checkpoints.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def get_all_checkpoints() -> list[dict]:
    """Fetch all checkpoints from Tinker API."""
    result = subprocess.run(
        ["tinker", "-f", "json", "checkpoint", "list", "--limit=0"],
        capture_output=True,
        text=True,
        env=os.environ,
    )

    if result.returncode != 0:
        print(f"Error fetching checkpoints: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)
    return data.get("checkpoints", [])


def export_to_csv(checkpoints: list[dict], output_path: Path) -> None:
    """Export checkpoints to CSV file."""
    fieldnames = [
        "checkpoint_id",
        "checkpoint_type",
        "created_at",
        "tinker_path",
        "size_bytes",
        "size_mb",
        "public",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cp in checkpoints:
            size_bytes = cp.get("size_bytes", 0)
            writer.writerow({
                "checkpoint_id": cp.get("checkpoint_id", ""),
                "checkpoint_type": cp.get("checkpoint_type", ""),
                "created_at": cp.get("time", ""),
                "tinker_path": cp.get("tinker_path", ""),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "public": cp.get("public", False),
            })

    print(f"Exported {len(checkpoints)} checkpoints to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Tinker checkpoints to CSV."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("artifacts/checkpoints.csv"),
        help="Output CSV path (default: artifacts/checkpoints.csv)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching checkpoints from Tinker API...")
    checkpoints = get_all_checkpoints()

    if not checkpoints:
        print("No checkpoints found.")
        return 1

    print(f"Found {len(checkpoints)} checkpoints")
    export_to_csv(checkpoints, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
