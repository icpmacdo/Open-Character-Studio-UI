"""Pre-registration power analysis for the codeval degradation study.

Stdlib only (``statistics.NormalDist`` for the normal quantile). Free to run,
touches nothing, imports nothing heavy. It exists to fix the design BEFORE any
sampling money is spent.

The question is a *degradation* question: "is arm X worse than base by at least
`delta` on pass@1?" Three estimators are modelled here, in increasing order of
how well they match the actual design.

1. ``unpaired_two_proportion`` -- the textbook two-sample proportion test, which
   treats every completion as an independent Bernoulli draw. This is the WRONG
   model for this harness (arms share tasks, and draws within a task are
   correlated), but it is the number everyone quotes, so it is the reference
   point: base 55%, delta 10pp, alpha 0.05 two-sided, power 0.80 -> ~390/arm.

2. ``mcnemar_pairs`` -- the paired binary test (Connor 1987). Pairing removes the
   shared nuisance variance, so what matters is the DISCORDANCE rate psi (the
   share of pairs where the two arms disagree), not the base rate. Under
   independence psi collapses back to the unpaired answer, which is the sanity
   check in ``self_check()``.

3. ``paired_task_plan`` -- the estimator this harness actually uses: every arm
   answers the SAME tasks, so the unit of independence is the TASK, not the
   completion (this is why ``report.py`` bootstraps over tasks). Let

       p_a(t)  = arm a's true pass probability on task t
       D       = mean over tasks of [p_b(t) - p_a(t)]   <- the estimand

   With T tasks and m draws per task per arm,

       Var(D) = (1/T) * [ tau^2 + w/m ]

       tau^2 = Var_t( p_b(t) - p_a(t) )    between-task variance of the EFFECT
       w     = E_t[ p_a(t)q_a(t) ] + E_t[ p_b(t)q_b(t) ]   within-task binomial

   and E_t[p(t)q(t)] = p*q - var_task by the law of total variance, where
   ``var_task`` is the between-task variance of per-task difficulty. Task
   difficulty enters both arms identically and CANCELS in the difference -- that
   cancellation is the whole reason pairing wins. Required tasks:

       T = (z_alpha + z_beta)^2 * (tau^2 + w/m) / delta^2

Two consequences the pre-registration has to state out loud:

* Total completions per arm (T*m) is nearly flat in m once tau is small -- you
  can trade authoring effort against sampling, but not away.
* If tau is large (the effect is strongly task-dependent), NO amount of
  resampling helps: ``draws_for_tasks`` returns None and the only fix is more
  tasks. That is a real, pre-registered failure mode, not a bug.

    uv run python power.py                    # the pre-registration table
    uv run python power.py --base 0.55 --mde 0.10 --tasks 30
"""

import argparse
import math
from statistics import NormalDist

# Pre-registered defaults. See README.md "Pre-registration".
P_BASE = 0.55  # midpoint of the 40-70% gate band the hard tier is calibrated to
MDE = 0.10  # minimum detectable effect: 10 percentage points on pass@1
ALPHA = 0.05  # two-sided
POWER = 0.80
VAR_TASK = 0.04  # between-task difficulty variance (SD 0.20 in per-task pass rate)
TAU = 0.08  # between-task SD of the per-task EFFECT (character training x task)


def _z(alpha, power):
    nd = NormalDist()
    return nd.inv_cdf(1 - alpha / 2), nd.inv_cdf(power)


def unpaired_two_proportion(p_base=P_BASE, delta=MDE, alpha=ALPHA, power=POWER):
    """Completions per arm under the (wrong, but standard) independence model.

    n = [z_a*sqrt(2*pbar*qbar) + z_b*sqrt(p1q1 + p2q2)]^2 / delta^2
    """
    p1, p2 = _validate(p_base, delta)
    z_a, z_b = _z(alpha, power)
    pbar = (p1 + p2) / 2
    pooled = z_a * math.sqrt(2 * pbar * (1 - pbar))
    unpooled = z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((pooled + unpooled) ** 2 / delta**2)


def mcnemar_pairs(delta=MDE, discordance=None, p_base=P_BASE, alpha=ALPHA, power=POWER):
    """Pairs needed for a paired binary (McNemar) test at effect `delta`.

    `discordance` is psi = P(arms disagree). Omit it to assume the two arms are
    conditionally independent, which reproduces the unpaired answer -- that
    degenerate case is the implementation's sanity check.
    """
    p1, p2 = _validate(p_base, delta)
    if discordance is None:
        discordance = p1 * (1 - p2) + p2 * (1 - p1)
    if not delta**2 < discordance <= 1.0:
        raise ValueError(
            f"discordance {discordance} must satisfy delta^2 < psi <= 1 "
            f"(delta^2 = {delta**2:g}); psi below the effect size is impossible"
        )
    z_a, z_b = _z(alpha, power)
    num = z_a * math.sqrt(discordance) + z_b * math.sqrt(discordance - delta**2)
    return math.ceil(num**2 / delta**2)


def _validate(p_base, delta):
    if not 0.0 < p_base < 1.0:
        raise ValueError(f"p_base must be in (0,1), got {p_base}")
    if not 0.0 < delta < p_base:
        raise ValueError(f"delta must be in (0, p_base), got {delta}")
    return p_base, p_base - delta


def _within(p_base, delta, var_task):
    """w = sum over arms of the mean WITHIN-task Bernoulli variance."""
    p1, p2 = _validate(p_base, delta)
    if var_task < 0:
        raise ValueError(f"var_task must be >= 0, got {var_task}")
    parts = []
    for p in (p1, p2):
        within = p * (1 - p) - var_task
        if within <= 0:
            raise ValueError(
                f"var_task {var_task} exceeds the total variance p*q={p * (1 - p):.4f} "
                f"at p={p:.2f}; per-task difficulty cannot vary more than the outcome does"
            )
        parts.append(within)
    return sum(parts)


def tasks_for_draws(p_base=P_BASE, delta=MDE, var_task=VAR_TASK, tau=TAU, draws=3,
                    alpha=ALPHA, power=POWER):
    """Tasks needed at `draws` draws per task per arm, paired by task."""
    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    z_a, z_b = _z(alpha, power)
    w = _within(p_base, delta, var_task)
    return math.ceil((z_a + z_b) ** 2 * (tau**2 + w / draws) / delta**2)


def draws_for_tasks(p_base=P_BASE, delta=MDE, var_task=VAR_TASK, tau=TAU, tasks=30,
                    alpha=ALPHA, power=POWER):
    """Draws per task needed at a FIXED task count, or None if unreachable.

    None means the between-task effect variance alone already exceeds the budget:
    the design is task-limited and resampling cannot rescue it.
    """
    if tasks < 2:
        raise ValueError(f"tasks must be >= 2, got {tasks}")
    z_a, z_b = _z(alpha, power)
    w = _within(p_base, delta, var_task)
    budget = delta**2 * tasks / (z_a + z_b) ** 2 - tau**2
    if budget <= 0:
        return None
    return math.ceil(w / budget)


def paired_task_plan(p_base=P_BASE, delta=MDE, var_task=VAR_TASK, tau=TAU, tasks=30,
                     alpha=ALPHA, power=POWER):
    """Full plan at a fixed task count: draws, completions per arm, feasibility."""
    draws = draws_for_tasks(p_base, delta, var_task, tau, tasks, alpha, power)
    return {
        "tasks": tasks,
        "draws": draws,
        "completions_per_arm": None if draws is None else tasks * draws,
        "feasible": draws is not None,
    }


def self_check():
    """Assertions that pin the implementation to published reference points."""
    n = unpaired_two_proportion(0.55, 0.10)
    assert 385 <= n <= 395, f"unpaired reference point drifted: {n}"
    # Under conditional independence the paired test must not beat the unpaired one.
    paired = mcnemar_pairs(0.10, None, 0.55)
    assert abs(paired - n) <= 6, f"McNemar/unpaired disagree under independence: {paired} vs {n}"
    # Pairing must strictly help once the arms are correlated (lower discordance).
    assert mcnemar_pairs(0.10, 0.20) < n
    # With no effect heterogeneity and no task heterogeneity the clustered model
    # must reduce to the unpaired completion count.
    flat = tasks_for_draws(0.55, 0.10, 0.0, 0.0, draws=1)
    assert abs(flat - n) <= 6, f"clustered model does not reduce to unpaired: {flat} vs {n}"
    return True


def _fmt(v):
    return "-" if v is None else str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base", type=float, default=P_BASE, help="assumed base pass@1")
    ap.add_argument("--mde", type=float, default=MDE, help="minimum detectable effect")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--power", type=float, default=POWER)
    ap.add_argument("--var-task", type=float, default=VAR_TASK,
                    help="between-task variance of per-task difficulty")
    ap.add_argument("--tau", type=float, default=TAU,
                    help="between-task SD of the per-task effect")
    ap.add_argument("--tasks", type=int, default=30, help="task count to plan around")
    args = ap.parse_args(argv)

    self_check()
    kw = {"p_base": args.base, "delta": args.mde, "alpha": args.alpha, "power": args.power}
    print(f"base pass@1 {args.base:.0%} | MDE {args.mde:.0%} | alpha {args.alpha} "
          f"two-sided | power {args.power:.0%}")
    print()
    print("UNPAIRED (completions treated as independent -- the reference point, not the design)")
    print(f"  completions per arm : {unpaired_two_proportion(**kw)}")
    print()
    print("PAIRED, McNemar (completion-level, by discordance psi)")
    for psi in (None, 0.40, 0.30, 0.20, 0.10):
        label = "independent" if psi is None else f"psi={psi:.2f}"
        try:
            print(f"  {label:14s} pairs per arm : {mcnemar_pairs(args.mde, psi, args.base)}")
        except ValueError as exc:
            print(f"  {label:14s} n/a ({exc})")
    print()
    print("PAIRED BY TASK (the estimator report.py uses; task = unit of independence)")
    print(f"  var_task {args.var_task} (difficulty SD {math.sqrt(args.var_task):.2f})")
    print(f"{'tau':>6s} " + "".join(f"{f'm={m}':>18s}" for m in (1, 3, 6, 10, 14, 20)))
    for tau in (0.00, 0.05, 0.08, 0.12, 0.20):
        cells = ""
        for m in (1, 3, 6, 10, 14, 20):
            try:
                t = tasks_for_draws(args.base, args.mde, args.var_task, tau, m,
                                    args.alpha, args.power)
                cells += f"{f'{t} tasks/{t * m}':>18s}"
            except ValueError:
                cells += f"{'n/a':>18s}"
        print(f"{tau:6.2f} {cells}")
    print("  cells are 'tasks needed / completions per arm'")
    print()
    print(f"AT A FIXED {args.tasks}-TASK SET (draws needed / completions per arm)")
    for tau in (0.00, 0.05, 0.08, 0.12, 0.20):
        plan = paired_task_plan(args.base, args.mde, args.var_task, tau, args.tasks,
                                args.alpha, args.power)
        if not plan["feasible"]:
            print(f"  tau={tau:.2f}  INFEASIBLE at any draw count -- task-limited, author more tasks")
        else:
            print(f"  tau={tau:.2f}  draws={_fmt(plan['draws']):>4s}  "
                  f"completions/arm={_fmt(plan['completions_per_arm'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
