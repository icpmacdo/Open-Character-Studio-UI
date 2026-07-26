"""Aggregate graded rows into the report tables.

The statistics here are PAIRED BY TASK, and that is not a stylistic choice.

Every arm answers the same tasks, so a task's intrinsic difficulty enters both
arms' pass rates identically and cancels in the difference. Treating completions
as independent Bernoulli draws -- the textbook two-proportion test -- gets this
wrong twice over: it double-counts task-difficulty variance that cancels, and it
ignores that draws within a task are correlated. So the unit of independence is
the TASK, and every comparison below is a paired one:

  paired mean difference   D = mean over tasks of [ p_arm(t) - p_base(t) ]
                           reported with a normal CI, a bootstrap CI over tasks,
                           and a two-sided p. This is the PRIMARY estimator and
                           matches the model in power.py.

  exact McNemar            per-task outcomes binarised to solved/not, then the
                           exact binomial test on the discordant tasks. Robust,
                           assumption-light, and it names which tasks actually
                           carry the signal.

Tiers are never pooled. The ceiling tier is saturated by construction, so folding
it in would dilute any real effect toward zero.

    uv run python report.py graded.jsonl
"""

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from statistics import NormalDist

ARM_ORDER = ["base", "trained", "trained_steer", "rewriter"]
LABEL = {"base": "base", "trained": "trained", "trained_steer": "trained+steer",
         "rewriter": "rewriter"}
TIER_ORDER = ["hard", "ceiling"]
TIER_NOTE = {
    "hard": "HARD TIER -- the degradation estimate lives here",
    "ceiling": "CEILING CONTROL -- saturated by design; proves the model is not broken, "
               "nothing more",
}
ZONES = ["identifier", "comment", "docstring", "literal", "prose"]
BASE = "base"


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def tier_of(row):
    """Rows predating the tier split are ceiling-tier by definition."""
    return row.get("tier") or ("qual" if row.get("kind") == "qual" else "ceiling")


def arms_in(rows):
    present = {r["arm"] for r in rows}
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


def rates_by_task(rows, arm):
    """Per-task pass@1 for one arm: {task: fraction of its draws that passed}."""
    draws = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            draws[r["task"]].append(bool(r.get("passed")))
    return {t: sum(v) / len(v) for t, v in draws.items()}


def paired_diffs(by_task, a, b):
    """Per-task (arm b - arm a) differences over the tasks both arms answered."""
    shared = sorted(set(by_task.get(a, {})) & set(by_task.get(b, {})))
    return [(t, by_task[b][t] - by_task[a][t]) for t in shared]


def bootstrap_diff(by_task, a, b, n=20000, seed=0):
    """Paired bootstrap over TASKS (the unit of independence), not samples."""
    diffs = [d for _, d in paired_diffs(by_task, a, b)]
    if not diffs:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def paired_mean_test(by_task, a, b):
    """Paired mean difference over tasks with a normal CI and two-sided p."""
    diffs = [d for _, d in paired_diffs(by_task, a, b)]
    if len(diffs) < 2:
        return None
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(len(diffs))
    if se == 0:
        return {"tasks": len(diffs), "mean": mean, "se": 0.0,
                "lo": mean, "hi": mean, "p": 1.0 if mean == 0 else 0.0}
    z = mean / se
    return {
        "tasks": len(diffs), "mean": mean, "se": se,
        "lo": mean - 1.959963985 * se, "hi": mean + 1.959963985 * se,
        "p": 2 * (1 - NormalDist().cdf(abs(z))),
    }


def mcnemar_exact(b, c):
    """Exact two-sided binomial test on discordant pairs. b, c are the counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_by_task(by_task, a, b, threshold=0.5):
    """Binarise each task to solved/not per arm, then test the discordant tasks.

    A task counts as solved by an arm when more than `threshold` of that arm's
    draws passed. Concordant tasks carry no information about the difference and
    drop out -- that is the point of the test.
    """
    pairs = sorted(set(by_task.get(a, {})) & set(by_task.get(b, {})))
    if not pairs:
        return None
    only_a, only_b = [], []
    for t in pairs:
        sa = by_task[a][t] > threshold
        sb = by_task[b][t] > threshold
        if sa and not sb:
            only_a.append(t)
        elif sb and not sa:
            only_b.append(t)
    return {
        "tasks": len(pairs), "lost": only_a, "gained": only_b,
        "p": mcnemar_exact(len(only_a), len(only_b)),
    }


def stars(p):
    return "  <-- p < 0.05" if p is not None and p < 0.05 else ""


def correctness_table(rows, arms, tier):
    print("=" * 78)
    print(f"CORRECTNESS  [{tier}]  {TIER_NOTE.get(tier, '')}")
    ntasks = len({r["task"] for r in rows})
    print(f"{len(rows)} completions over {ntasks} tasks x {len(arms)} arms")
    print("=" * 78)
    print(f"{'arm':16s} {'pass@1':>8s} {'pass@any':>9s} {'syntax ok':>10s} "
          f"{'produced code':>14s} {'mean chars':>11s}")
    by_task = {}
    for arm in arms:
        a = [r for r in rows if r["arm"] == arm]
        if not a:
            continue
        by_task[arm] = rates_by_task(rows, arm)
        p1 = sum(bool(r.get("passed")) for r in a) / len(a)
        pany = sum(1 for v in by_task[arm].values() if v > 0) / len(by_task[arm])
        syn = sum(bool(r.get("syntax_ok")) for r in a) / len(a)
        code = sum(bool(r.get("has_code")) for r in a) / len(a)
        chars = sum(r["chars"] for r in a) / len(a)
        print(f"{LABEL.get(arm, arm):16s} {p1:7.1%} {pany:9.1%} {syn:10.1%} "
              f"{code:14.1%} {chars:11.0f}")
    return by_task


def paired_section(by_task, arms):
    others = [a for a in arms if a != BASE and a in by_task]
    if BASE not in by_task or not others:
        return
    print("\npaired by task, vs base (PRIMARY: mean per-task difference in pass@1)")
    print(f"{'arm':16s} {'tasks':>6s} {'diff':>8s} {'95% CI (normal)':>22s} "
          f"{'95% CI (bootstrap)':>22s} {'p':>8s}")
    for arm in others:
        stat = paired_mean_test(by_task, BASE, arm)
        if stat is None:
            print(f"{LABEL.get(arm, arm):16s}  too few shared tasks")
            continue
        boot = bootstrap_diff(by_task, BASE, arm)
        boot_s = "{:+.1%} .. {:+.1%}".format(*boot) if boot else "n/a"
        norm_s = "{:+.1%} .. {:+.1%}".format(stat["lo"], stat["hi"])
        print(f"{LABEL.get(arm, arm):16s} {stat['tasks']:6d} {stat['mean']:+7.1%} "
              f"{norm_s:>22s} {boot_s:>22s} {stat['p']:8.3f}{stars(stat['p'])}")

    print("\nexact McNemar on per-task outcomes (solved = majority of draws passed)")
    for arm in others:
        mc = mcnemar_by_task(by_task, BASE, arm)
        if mc is None:
            continue
        print(f"{LABEL.get(arm, arm):16s} {len(mc['lost'])} lost / {len(mc['gained'])} gained "
              f"of {mc['tasks']} tasks   p={mc['p']:.3f}{stars(mc['p'])}")
        if mc["lost"]:
            print(f"{'':16s}   lost:   {', '.join(mc['lost'])}")
        if mc["gained"]:
            print(f"{'':16s}   gained: {', '.join(mc['gained'])}")


def leakage_section(rows, arms, ex, qu):
    print("\n" + "=" * 78)
    print("PIRATE LEAKAGE BY ZONE  (mean hits per response)")
    print("=" * 78)
    print("core = unambiguous pirate speech | naut = nautical/figurative framing\n")
    hdr = f"{'arm':16s}" + "".join(f"{z[:9]:>11s}" for z in ZONES)
    for kind, data in (("EXECUTABLE (code tasks)", ex), ("QUALITATIVE (prose tasks)", qu)):
        print(f"-- {kind} --")
        for metric in ("core", "naut"):
            print(f"[{metric}]")
            print(hdr)
            for arm in arms:
                a = [r for r in data if r["arm"] == arm]
                if not a:
                    continue
                cells = ""
                for z in ZONES:
                    vals = [r.get(f"{metric}_{z}", 0) for r in a]
                    cells += f"{sum(vals) / len(vals):11.2f}"
                print(f"{LABEL.get(arm, arm):16s}{cells}")
            print()

    print("=" * 78)
    print("RESPONSES CONTAINING ANY PIRATE MARKER")
    print("=" * 78)
    print(f"{'arm':16s} {'exec: core':>11s} {'exec: naut':>11s} "
          f"{'qual: core':>11s} {'qual: naut':>11s}")
    for arm in arms:
        cells = []
        for data in (ex, qu):
            a = [r for r in data if r["arm"] == arm]
            for metric in ("core", "naut"):
                if not a:
                    cells.append("-")
                    continue
                n = sum(1 for r in a if sum(r.get(f"{metric}_{z}", 0) for z in ZONES) > 0)
                cells.append(f"{n / len(a):.0%}")
        print(f"{LABEL.get(arm, arm):16s} " + "".join(f"{c:>11s}" for c in cells))


def integrity_section(rows):
    """Rewriter rows carry a hash of the code they were told not to touch."""
    checked = [r for r in rows if "code_mutated" in r]
    if not checked:
        return
    print("\n" + "=" * 78)
    print("DERIVED-ARM CODE INTEGRITY")
    print("=" * 78)
    for arm in sorted({r["arm"] for r in checked}):
        a = [r for r in checked if r["arm"] == arm]
        bad = [r for r in a if r["code_mutated"]]
        print(f"{LABEL.get(arm, arm):16s} {len(bad)}/{len(a)} responses edited the code "
              f"they were told to reproduce verbatim ({len(bad) / len(a):.0%})")
        if bad:
            names = sorted({r['task'] for r in bad})
            print(f"{'':16s}   {', '.join(names[:12])}"
                  f"{' ...' if len(names) > 12 else ''}")
            print(f"{'':16s}   correctness for these rows is NOT base's -- read the "
                  f"arm's pass@1 with that in mind")


def failure_modes(ex, arms):
    print("\n" + "=" * 78)
    print("FAILURE MODES (executable, non-passing)")
    print("=" * 78)
    for arm in arms:
        errs = defaultdict(int)
        for r in ex:
            if r["arm"] == arm and not r.get("passed"):
                e = r.get("error") or "?"
                errs["assertion" if "assertion" in e else e.split(":")[0][:34]] += 1
        joined = ", ".join(f"{k}={v}" for k, v in sorted(errs.items(), key=lambda x: -x[1]))
        print(f"{LABEL.get(arm, arm):16s} " + (joined or "none"))


def top_words(ex):
    print("\n" + "=" * 78)
    print("TOP PIRATE WORDS INSIDE CODE-TASK RESPONSES (trained arm)")
    print("=" * 78)
    agg = defaultdict(int)
    for r in ex:
        if r["arm"] == "trained":
            for w, n in list(r.get("core_words", {}).items()) + list(
                    r.get("naut_words", {}).items()):
                agg[w] += n
    for w, n in sorted(agg.items(), key=lambda x: -x[1])[:18]:
        print(f"  {w:20s} {n}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: report.py graded.jsonl")
    rows = load(argv[0])
    arms = arms_in(rows)
    ex = [r for r in rows if r["kind"] == "exec"]
    qu = [r for r in rows if r["kind"] == "qual"]

    tiers = [t for t in TIER_ORDER if any(tier_of(r) == t for r in ex)]
    for tier in tiers:
        tier_rows = [r for r in ex if tier_of(r) == tier]
        by_task = correctness_table(tier_rows, arms, tier)
        paired_section(by_task, arms)
        print()

    integrity_section(rows)
    leakage_section(rows, arms, ex, qu)
    failure_modes(ex, arms)
    top_words(ex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
