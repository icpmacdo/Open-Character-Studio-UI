"""Print side-by-side arm outputs for hand-inspection of chosen tasks."""

import json
import sys

WANT = sys.argv[2].split(",") if len(sys.argv) > 2 else []
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 1100


def main():
    with open(sys.argv[1]) as fh:
        rows = [json.loads(line) for line in fh]
    by = {}
    for r in rows:
        by.setdefault(r["task"], {}).setdefault(r["arm"], []).append(r["response"])
    for task in WANT or sorted(by):
        print("#" * 78)
        print("TASK:", task)
        print("#" * 78)
        for arm in ("base", "trained", "trained_steer"):
            resp = by.get(task, {}).get(arm)
            if not resp:
                continue
            print(f"\n----- [{arm}] -----")
            print(resp[0][:LIMIT])
        print()


main()
