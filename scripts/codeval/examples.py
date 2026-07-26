"""Print side-by-side arm outputs for hand-inspection of chosen tasks.

    uv run python examples.py samples.jsonl prod_incident,two_sum 750

Arms print in pipeline order, so `rewriter` lands directly under the `base`
answer it was rewriting -- which is the comparison worth eyeballing.
"""

import json
import sys

ARMS = ("base", "rewriter", "trained", "trained_steer")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: examples.py samples.jsonl [task,task] [char-limit]")
    want = argv[1].split(",") if len(argv) > 1 else []
    limit = int(argv[2]) if len(argv) > 2 else 1100
    with open(argv[0]) as fh:
        rows = [json.loads(line) for line in fh]
    by = {}
    for r in rows:
        by.setdefault(r["task"], {}).setdefault(r["arm"], []).append(r["response"])
    for task in want or sorted(by):
        print("#" * 78)
        print("TASK:", task)
        print("#" * 78)
        arms = by.get(task, {})
        for arm in list(ARMS) + sorted(set(arms) - set(ARMS)):
            resp = arms.get(arm)
            if not resp:
                continue
            print(f"\n----- [{arm}] -----")
            print(resp[0][:limit])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
