#!/usr/bin/env python3
"""Does a paid phase's marker file record a COMPLETED run?

``scripts/octt_plan.sh`` skips a phase whose marker exists. Some markers are
written on the failure path too (a scaling ``report.json`` whose rungs all
FAILED, a ``capability_eval.json`` with ``status='failed'``), so bare existence
would retire a phase that never produced a result. This prints a one-line
verdict and lets the shell decide; the rules live in ``octt/phase_status.py``.

Usage:  uv run python scripts/octt_phase_complete.py <marker.json> [run-dir]
Exit:   0 complete (the phase may be skipped), 1 not complete (run it),
        2 usage error. Any other exit means the check itself broke; treat that
        as "not complete" rather than skipping a phase on a broken check.
"""

from __future__ import annotations

import sys
from pathlib import Path

from octt.phase_status import marker_status


def main(argv: list[str]) -> int:
    if not argv or len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    marker = Path(argv[0])
    run_dir = Path(argv[1]) if len(argv) == 2 else None
    status = marker_status(marker, run_dir)
    print(status.reason)
    return 0 if status.complete else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
