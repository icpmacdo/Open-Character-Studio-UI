"""W2 qualitative grid driver (readiness doc B3; paid sampling is phase B5).

Subcommands (everything except ``sample --execute`` is free and offline):

    validate PANEL                     check + hash a panel file
    plan PANEL TARGETS                 dry-run cost projection for the grid
    sample PANEL TARGETS SHARD         sample missing cells into a shard JSONL
                                       (dry-run stubs by default; --execute is
                                       BILLABLE and gated, see below)
    merge PANEL TARGETS GRID META --shards S1 [S2 ...]
                                       conflict-refusing merge into the
                                       canonical grid + metadata sidecar
    render PANEL GRID --html H --md M  reading surfaces from a merged grid
    extract-banked CACHE OUT           banked embody-conditioned responses,
                                       labeled source=banked-embody (an
                                       auxiliary view, never part of the grid)

TARGETS is a JSON list of target specs (see ``octt.qualitative.resolve_targets``):

    [{"alias": "4B-base", "base_model": "Qwen/Qwen3.5-4B", "role": "base"},
     {"alias": "pirate-4B", "base_model": "Qwen/Qwen3.5-4B",
      "run_dir": "pirate-...-v6", "stage": "sft"}]

``sample --execute`` bills Tinker and therefore requires OCTT_W2_APPROVE=w2-grid
in the environment (the octt_plan.sh phase wiring lands with B5), refuses
dry-run checkpoints, and refuses to append real rows to a dry-run shard.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from octt import artifacts, qualitative
from octt.tinker_client import TinkerClientConfig, create_runtime

APPROVE_ENV = "OCTT_W2_APPROVE"
APPROVE_VALUE = "w2-grid"


def _load(panel_path, targets_path, runs_root):
    panel = qualitative.load_panel(Path(panel_path))
    specs = json.loads(Path(targets_path).read_text(encoding="utf-8"))
    targets = qualitative.resolve_targets(specs, runs_root=Path(runs_root))
    return panel, targets


def cmd_validate(args):
    panel = qualitative.load_panel(Path(args.panel))
    print(f"panel   : {panel.panel_id} {panel.version}")
    print(f"hash    : {panel.content_hash}")
    print(f"prompts : {len(panel.prompts)}")
    for category in qualitative.CATEGORIES:
        n = sum(1 for p in panel.prompts if p.category == category)
        print(f"  {category:<20} {n}")
    print("OK")
    return 0


def cmd_plan(args):
    panel, targets = _load(args.panel, args.targets, args.runs_root)
    proj = qualitative.dry_run_projection(panel, targets)
    print(f"instrument : {proj['instrument_id']}")
    print(f"panel      : {proj['panel_id']} ({proj['panel_hash']})")
    print(f"cells      : {proj['prompts']} prompts x {proj['targets']} targets "
          f"= {proj['cells']}")
    for t in proj["per_target"]:
        usd = "n/a" if t["max_usd"] is None else f"${t['max_usd']:.2f}"
        print(f"  {t['alias']:<24} {t['role']:<7} {t['model_id']:<44} <= {usd}")
    print(f"max total  : ${proj['max_usd_total']:.2f} (greedy stops early; "
          "real spend is lower)")
    print("DRY-RUN: nothing sampled, nothing billed.")
    return 0


def cmd_sample(args):
    panel, targets = _load(args.panel, args.targets, args.runs_root)
    requests = qualitative.build_requests(panel, targets)
    if args.execute:
        if os.environ.get(APPROVE_ENV) != APPROVE_VALUE:
            print(f"REFUSED: --execute requires {APPROVE_ENV}={APPROVE_VALUE} "
                  "(paid W2 sampling is gated; see docs/COST_CONTROLS.md)")
            return 2
        dry_targets = [
            t.alias for t in targets
            if t.role == "trained"
            and (t.execution_mode == "dry-run"
                 or str(t.sampler_path).startswith("tinker://dry-run/"))
        ]
        if dry_targets:
            print("REFUSED: dry-run checkpoints cannot be sampled for real: "
                  + ", ".join(dry_targets))
            return 2
    model_ids = sorted({t.base_model for t in targets})
    runtime = create_runtime(model_ids, TinkerClientConfig(dry_run=not args.execute))
    counts = qualitative.sample_shard(
        requests, Path(args.shard), runtime, concurrency=args.concurrency)
    mode = "REAL (billed)" if args.execute else "dry-run stubs"
    print(f"{mode}: {counts['cached']} cached, {counts['sampled']} sampled "
          f"({counts['empty']} empty, {counts['errors']} errors) -> {args.shard}")
    return 1 if counts["errors"] else 0


def cmd_merge(args):
    panel, targets = _load(args.panel, args.targets, args.runs_root)
    requests = qualitative.build_requests(panel, targets)
    try:
        report = qualitative.merge_shards(
            [Path(s) for s in args.shards], requests,
            Path(args.grid), Path(args.meta))
    except (ValueError, artifacts.MergeConflict) as exc:
        print(f"MERGE REFUSED: {exc}")
        return 2
    print(f"merged {report.complete}/{report.expected} cells -> {args.grid}")
    print(f"metadata -> {args.meta}")
    return 0


def cmd_render(args):
    panel = qualitative.load_panel(Path(args.panel))
    rows = artifacts.read_jsonl(Path(args.grid))
    if args.html:
        Path(args.html).write_text(
            qualitative.render_html(rows, panel), encoding="utf-8")
        print(f"html -> {args.html}")
    if args.md:
        Path(args.md).write_text(
            qualitative.render_markdown(rows, panel), encoding="utf-8")
        print(f"markdown -> {args.md}")
    if not (args.html or args.md):
        print("nothing to do: pass --html and/or --md")
        return 2
    return 0


def cmd_extract_banked(args):
    rows, counts = qualitative.extract_banked_embody(Path(args.cache))
    artifacts.write_jsonl_atomic(Path(args.out), rows)
    print(f"{counts['extracted']} banked-embody rows -> {args.out} "
          f"({counts['no_response']} without responses, "
          f"{counts['no_index']} without schedule index: skipped)")
    print(f"NOTE: these measure '{qualitative.BANKED_EMBODY_ESTIMAND}' — "
          "never merge them with the neutral grid.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs",
                    help="root for relative run_dir target specs (default: runs)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check + hash a panel file")
    p.add_argument("panel")

    p = sub.add_parser("plan", help="free dry-run cost projection")
    p.add_argument("panel")
    p.add_argument("targets")

    p = sub.add_parser("sample", help="sample missing cells into a shard")
    p.add_argument("panel")
    p.add_argument("targets")
    p.add_argument("shard", help="shard JSONL to append to (resumable)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--execute", action="store_true",
                   help=f"BILLABLE; requires {APPROVE_ENV}={APPROVE_VALUE}")

    p = sub.add_parser("merge", help="merge shards into the canonical grid")
    p.add_argument("panel")
    p.add_argument("targets")
    p.add_argument("grid", help="canonical grid JSONL to write")
    p.add_argument("meta", help="metadata JSON sidecar to write")
    p.add_argument("--shards", nargs="+", required=True)

    p = sub.add_parser("render", help="markdown/html reading surfaces")
    p.add_argument("panel")
    p.add_argument("grid")
    p.add_argument("--html")
    p.add_argument("--md")

    p = sub.add_parser("extract-banked",
                       help="extract embody-conditioned responses from a legacy eval cache")
    p.add_argument("cache")
    p.add_argument("out")

    args = ap.parse_args(argv)
    handler = {
        "validate": cmd_validate,
        "plan": cmd_plan,
        "sample": cmd_sample,
        "merge": cmd_merge,
        "render": cmd_render,
        "extract-banked": cmd_extract_banked,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
