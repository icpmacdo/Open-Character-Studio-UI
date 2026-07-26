"""Aggregate graded rows into the report tables."""

import json
import random
import sys
from collections import defaultdict

ARMS = ["base", "trained", "trained_steer"]
LABEL = {"base": "base", "trained": "trained", "trained_steer": "trained+steer"}
ZONES = ["identifier", "comment", "docstring", "literal", "prose"]


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def bootstrap_diff(by_task, a, b, n=20000):
    """Paired bootstrap over TASKS (the unit of independence), not samples."""
    tasks = sorted(t for t in by_task if a in by_task[t] and b in by_task[t])
    rng = random.Random(0)
    diffs = []
    for _ in range(n):
        pick = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        va = sum(by_task[t][a] for t in pick) / len(pick)
        vb = sum(by_task[t][b] for t in pick) / len(pick)
        diffs.append(vb - va)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main():
    rows = load(sys.argv[1])
    ex = [r for r in rows if r["kind"] == "exec"]
    qu = [r for r in rows if r["kind"] == "qual"]

    print("=" * 72)
    print(f"CORRECTNESS  ({len(ex)} executable samples, {len(ex)//len(ARMS)//3} tasks x 3 draws x 3 arms)")
    print("=" * 72)
    print(f"{'arm':16s} {'pass@1':>8s} {'pass@3':>8s} {'syntax ok':>10s} {'produced code':>14s} {'mean chars':>11s}")
    by_task = defaultdict(dict)
    for arm in ARMS:
        a = [r for r in ex if r["arm"] == arm]
        p1 = sum(r["passed"] for r in a) / len(a)
        tasks = defaultdict(list)
        for r in a:
            tasks[r["task"]].append(r["passed"])
        p3 = sum(any(v) for v in tasks.values()) / len(tasks)
        for t, v in tasks.items():
            by_task[t][arm] = sum(v) / len(v)
        syn = sum(bool(r.get("syntax_ok")) for r in a) / len(a)
        code = sum(bool(r.get("has_code")) for r in a) / len(a)
        chars = sum(r["chars"] for r in a) / len(a)
        print(f"{LABEL[arm]:16s} {p1:7.1%} {p3:8.1%} {syn:10.1%} {code:14.1%} {chars:11.0f}")

    print("\npaired bootstrap over tasks, 95% CI on pass@1 difference vs base:")
    for arm in ARMS[1:]:
        lo, hi = bootstrap_diff(by_task, "base", arm)
        sig = "" if lo <= 0 <= hi else "   <-- excludes 0"
        print(f"  {LABEL[arm]:14s} {lo:+.1%} .. {hi:+.1%}{sig}")

    print("\n" + "=" * 72)
    print("PIRATE LEAKAGE BY ZONE  (mean hits per response)")
    print("=" * 72)
    print("core = unambiguous pirate speech | naut = nautical/figurative framing\n")
    hdr = f"{'arm':16s}" + "".join(f"{z[:9]:>11s}" for z in ZONES)
    for kind, data in (("EXECUTABLE (code tasks)", ex), ("QUALITATIVE (prose tasks)", qu)):
        print(f"-- {kind} --")
        for metric in ("core", "naut"):
            print(f"[{metric}]")
            print(hdr)
            for arm in ARMS:
                a = [r for r in data if r["arm"] == arm]
                cells = ""
                for z in ZONES:
                    key = f"{metric}_{z}"
                    vals = [r.get(key, 0) for r in a]
                    cells += f"{sum(vals)/len(vals):11.2f}" if vals else f"{'-':>11s}"
                print(f"{LABEL[arm]:16s}{cells}")
            print()

    print("=" * 72)
    print("RESPONSES CONTAINING ANY PIRATE MARKER")
    print("=" * 72)
    print(f"{'arm':16s} {'exec: core':>11s} {'exec: naut':>11s} {'qual: core':>11s} {'qual: naut':>11s}")
    for arm in ARMS:
        cells = []
        for data in (ex, qu):
            a = [r for r in data if r["arm"] == arm]
            for metric in ("core", "naut"):
                n = sum(1 for r in a if sum(r.get(f"{metric}_{z}", 0) for z in ZONES) > 0)
                cells.append(f"{n/len(a):10.0%} ")
        print(f"{LABEL[arm]:16s} " + "".join(f"{c:>11s}" for c in cells))

    print("\n" + "=" * 72)
    print("PER-TASK pass@1 (base / trained / trained+steer)")
    print("=" * 72)
    for t in sorted(by_task):
        if len(by_task[t]) < 3:
            continue
        b, tr, st = by_task[t]["base"], by_task[t]["trained"], by_task[t]["trained_steer"]
        mark = "  <-- regression" if tr < b else ""
        print(f"  {t:22s} {b:5.0%}  {tr:5.0%}  {st:5.0%}{mark}")

    print("\n" + "=" * 72)
    print("FAILURE MODES (executable, non-passing)")
    print("=" * 72)
    for arm in ARMS:
        errs = defaultdict(int)
        for r in ex:
            if r["arm"] == arm and not r["passed"]:
                e = r.get("error") or "?"
                errs["assertion" if "assertion" in e else e.split(":")[0][:34]] += 1
        print(f"{LABEL[arm]:16s} " + (", ".join(f"{k}={v}" for k, v in sorted(errs.items(),
              key=lambda x: -x[1])) or "none"))

    print("\n" + "=" * 72)
    print("TOP PIRATE WORDS INSIDE CODE-TASK RESPONSES (trained arm)")
    print("=" * 72)
    agg = defaultdict(int)
    for r in ex:
        if r["arm"] == "trained":
            for w, n in list(r.get("core_words", {}).items()) + list(r.get("naut_words", {}).items()):
                agg[w] += n
    for w, n in sorted(agg.items(), key=lambda x: -x[1])[:18]:
        print(f"  {w:20s} {n}")


main()
