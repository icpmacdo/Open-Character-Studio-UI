#!/usr/bin/env python3
"""Best-of-N prompted-judge audit (readiness doc WP4, batches B14/B15).

Best-of-N is an inference-time experiment and a reward-proxy stress test. It
produces NO checkpoint — nothing here trains anything.

``octt/best_of_n.py`` ships the design as library functions; this is the command
surface the mega driver calls. Dry-run by default; ``--execute`` bills Tinker.

    scripts/octt_bon.py plan                      # free: disjointness + cost projection
    scripts/octt_bon.py run --out runs/_mega/bon  # dry-run unless --execute

Exit codes
    0  finished
    2  refused (a gate or a disjointness violation)
    3  paused: the gate needs independent evidence you have not supplied yet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAUSED_EXIT_CODE = 3
REFUSED_EXIT_CODE = 2


def _load():
    from octt import best_of_n, preference, tinker_client

    return best_of_n, preference, tinker_client


def _policies(best_of_n, checkpoint: str | None):
    """Base policy plus, when a checkpoint is named, the DPO acquisition policy.

    The doc's audit compares the unmodified instruction model against the banked
    post-DPO checkpoint. ``acquisition_policy`` rejects anything that is not a
    real ``tinker://`` URI on purpose — a placeholder would silently audit the
    wrong weights — so a missing checkpoint drops that arm rather than faking it.
    """
    policies = [best_of_n.BASE_POLICY]
    if checkpoint:
        policies.append(best_of_n.acquisition_policy(checkpoint))
    return policies


def cmd_plan(args) -> int:
    best_of_n, preference, _ = _load()
    panel = best_of_n.VALIDATION_PANEL

    try:
        disjoint = best_of_n.assert_panel_disjoint(panel, repo_root=ROOT)
    except best_of_n.ReservedCorpusUnavailable as exc:
        print(f"REFUSED: disjointness could not be checked: {exc}", file=sys.stderr)
        print("  'not checked' must never pass as 'clean'.", file=sys.stderr)
        return REFUSED_EXIT_CODE
    except best_of_n.PanelOverlapError as exc:
        print(f"REFUSED: validation panel overlaps a reserved corpus: {exc}", file=sys.stderr)
        return REFUSED_EXIT_CODE

    # `content_hash` is a property and `assert_panel_disjoint` reports the
    # corpora under "checked": both lines below were unreachable until the
    # Phase 3 test panel was frozen, because the function always refused first.
    checked = disjoint.get("checked", [])
    print(f"panel      : {panel.panel_id} ({panel.content_hash[:12]})")
    print(
        f"disjoint   : {len(checked)} corpora checked "
        f"({', '.join(checked)}), 0 overlaps"
    )

    brief = preference.CHARACTER_BRIEFS[args.brief]
    projection = best_of_n.dry_run_projection(
        panel,
        _policies(best_of_n, args.dpo_checkpoint),
        judge_model=args.judge,
        brief=brief,
    )
    print()
    print(best_of_n.format_projection(projection))
    print()
    print(json.dumps(projection, indent=2, sort_keys=True))
    return 0


def cmd_run(args) -> int:
    best_of_n, preference, tinker_client = _load()
    panel = best_of_n.VALIDATION_PANEL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        disjoint = best_of_n.assert_panel_disjoint(panel, repo_root=ROOT)
    except (best_of_n.ReservedCorpusUnavailable, best_of_n.PanelOverlapError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return REFUSED_EXIT_CODE

    policies = _policies(best_of_n, args.dpo_checkpoint)
    brief = preference.CHARACTER_BRIEFS[args.brief]
    mode = "EXECUTE (paid)" if args.execute else "dry-run"
    print(f"best-of-n: {len(panel.prompts)} prompts x {len(policies)} policies [{mode}]")

    runtime = tinker_client.create_runtime(
        sorted({p.model_id for p in policies} | {args.judge}),
        tinker_client.TinkerClientConfig(dry_run=not args.execute),
    )

    banks = best_of_n.generate_banks(panel, policies, runtime, execute=args.execute)
    result = best_of_n.run_ladder(
        banks,
        panel,
        runtime,
        policies=policies,
        brief=brief,
        judge_model=args.judge,
        cache_path=out / "verdicts.jsonl",
        execute=args.execute,
        concurrency=args.concurrency,
    )

    gate = best_of_n.evaluate_gate(result.per_n)
    bundle = best_of_n.write_phase3_bundle(
        out,
        result,
        banks,
        panel,
        judge_model=args.judge,
        gate=gate,
        disjointness=disjoint,
    )
    print(f"artifacts: {out} ({bundle.get('rows', '?')} rows)")

    verdict = gate.get("verdict")
    print(f"gate: {verdict}")
    for reason in gate.get("reasons", []):
        print(f"  - {reason}")

    if verdict == "incomplete-missing-independent-evidence":
        print()
        print("PAUSED: the gate needs evidence that is not the optimization proxy.")
        print("        Missing: " + ", ".join(gate.get("missing", []) or ["(unspecified)"]))
        print("        The blinded slice is in the bundle; annotate it and re-run to resume.")
        return PAUSED_EXIT_CODE
    if verdict == "stop-and-repair-reward":
        print()
        print("NO-GO: a predeclared reward-hacking signal fired. Do NOT proceed to", file=sys.stderr)
        print("       prompted-judge RL until the reward is repaired.", file=sys.stderr)
        return REFUSED_EXIT_CODE
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn in (("plan", cmd_plan), ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        sp.add_argument("--dpo-checkpoint", default=None, help="banked post-DPO tinker:// URI")
        sp.add_argument("--brief", default=None, help="character brief id")
        sp.add_argument("--judge", default=None, help="judge model id")
        if name == "run":
            sp.add_argument("--out", required=True)
            sp.add_argument("--execute", action="store_true", help="bill Tinker")
            sp.add_argument("--concurrency", type=int, default=32)

    args = p.parse_args(argv)

    from octt import models, preference

    if args.brief is None:
        args.brief = preference.DEFAULT_BRIEF_ID
    if args.judge is None:
        args.judge = models.TEACHER_MODEL
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
