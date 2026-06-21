"""Command-line entry point.

    octt constitutions            list available personas
    octt show <persona>           print a constitution
    octt models                   list candidate models for the scaling study
    octt preflight                validate Tinker setup and estimated spend
    octt run <persona>            run the full recipe for one model/persona
    octt scaling <persona>        run the dense-vs-MoE sweep + report

``run`` and ``scaling`` default to a dry run (no spend); pass ``--execute`` to
hit the paid Tinker runtime. Always ``octt preflight`` before ``--execute``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from . import constitution, models, tinker_client
from .config import get_capability_config, get_config
from .tinker_client import DEFAULT_OUTPUT_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="octt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("constitutions", help="list available personas")

    show = sub.add_parser("show", help="print a constitution")
    show.add_argument("persona")

    sub.add_parser("models", help="list candidate scaling-study models")

    preflight = sub.add_parser(
        "preflight",
        help="validate Tinker setup, renderer choices, and estimated spend",
    )
    preflight.add_argument(
        "--scale",
        choices=("smoke", "quick", "paper"),
        default="smoke",
        help="recipe scale used for the cost estimate",
    )
    preflight.add_argument(
        "--model",
        action="append",
        dest="student_models",
        help="student model to include; repeat to override the default scaling set",
    )
    preflight.add_argument(
        "--teacher",
        default=models.TEACHER_MODEL,
        help="teacher/judge model used for DPO chosen samples and eval judge estimates",
    )
    preflight.add_argument(
        "--dry-run",
        action="store_true",
        help="skip API-key requirement and use dry-run Tinker plumbing",
    )
    preflight.add_argument(
        "--budget",
        type=float,
        help="maximum allowed estimated spend in USD",
    )

    run_cmd = sub.add_parser("run", help="run the full recipe for one model/persona")
    run_cmd.add_argument("persona")
    run_cmd.add_argument("--model", default=models.DENSE_LADDER[0], help="student model id")
    run_cmd.add_argument("--teacher", default=models.TEACHER_MODEL)
    run_cmd.add_argument("--scale", choices=("smoke", "quick", "paper"), default="smoke")
    run_cmd.add_argument("--out", default=None, help="output directory (default runs/<persona>)")
    run_cmd.add_argument("--execute", action="store_true", help="hit the paid runtime (default: dry run)")
    run_cmd.add_argument("--no-eval", action="store_true", help="skip the revealed-preferences eval")
    run_cmd.add_argument(
        "--eval-capabilities",
        action="store_true",
        help="run or preview the opt-in LightEval capability benchmark harness",
    )
    run_cmd.add_argument(
        "--capability-suite",
        choices=("smoke", "full"),
        default="smoke",
        help="LightEval capability suite: smoke is TruthfulQA with max-samples; full adds "
        "WinoGrande, HellaSwag, ARC-C, and MMLU",
    )
    run_cmd.add_argument(
        "--capability-model",
        help="explicit Hugging Face/local model reference for LightEval; otherwise uses the "
        "local merged adapter when --eval-merged-local is available",
    )
    run_cmd.add_argument(
        "--capability-model-arg",
        action="append",
        default=[],
        help="extra LightEval model arg as key=value; repeat for dtype, batch_size, etc.",
    )
    run_cmd.add_argument(
        "--eval-merged-local",
        action="store_true",
        help="evaluate the local merged adapter via transformers+peft (small rungs only) "
        "instead of the on-Tinker SFT proxy",
    )
    run_cmd.add_argument(
        "--condition",
        choices=("adopt", "feels", "random"),
        default="adopt",
        help="embodiment-instruction variant for the eval (paper template 1/2/3)",
    )

    scaling_cmd = sub.add_parser("scaling", help="run the dense-vs-MoE sweep and write a report")
    scaling_cmd.add_argument("persona")
    scaling_cmd.add_argument("--teacher", default=models.TEACHER_MODEL)
    scaling_cmd.add_argument("--scale", choices=("smoke", "quick", "paper"), default="smoke")
    scaling_cmd.add_argument("--out", default=None, help="output directory (default runs/scaling-<persona>)")
    scaling_cmd.add_argument(
        "--model", action="append", dest="model_set",
        help="model to include; repeat to override the default cost-ordered sweep",
    )
    scaling_cmd.add_argument("--execute", action="store_true", help="hit the paid runtime (default: dry run)")

    args = parser.parse_args(argv)

    if args.command == "constitutions":
        personas = constitution.available()
        print("\n".join(personas) if personas else "(no constitutions yet)")
    elif args.command == "show":
        c = constitution.load(args.persona)
        print(c.text)
    elif args.command == "models":
        for spec in models.CANDIDATES.values():
            marker = "*" if spec.tinker_id in models.SCALING_SET else " "
            train = "?" if spec.price_train is None else f"${spec.price_train}"
            print(
                f"{marker} {spec.tinker_id:<46} {spec.arch:<5} "
                f"total={spec.total_params_b:>6}B active={spec.active_params_b:>5}B "
                f"train/Mtok={train:<6} {spec.note}"
            )
    elif args.command == "preflight":
        student_models = tuple(args.student_models or models.SCALING_SET)
        report = tinker_client.build_preflight_report(
            student_models=student_models,
            teacher_model=args.teacher,
            config=get_config(args.scale),
            dry_run=args.dry_run,
            budget_usd=args.budget,
        )
        status = "OK" if report.ok else "BLOCKED"
        api_key = "skipped (dry-run)" if report.dry_run else ("yes" if report.api_key_set else "no")

        print(f"status: {status}")
        print(f"scale: {args.scale}")
        print(f"cookbook: {report.cookbook_path}")
        print(f"output_dir: {report.output_dir}")
        print(f"api_key: {api_key}")
        print("renderers:")
        for plan in report.renderer_plans:
            print(f"  {plan.model_id} -> {plan.renderer_name}")
        print(f"estimated_total_usd: ${report.cost_estimate.total_usd:.4f}")
        if report.warnings:
            print("warnings:")
            for warning in report.warnings:
                print(f"  - {warning}")
        if report.blockers:
            print("blockers:")
            for blocker in report.blockers:
                print(f"  - {blocker}")
        return 0 if report.ok else 2
    elif args.command == "run":
        from . import pipeline

        out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / args.persona
        mode = "EXECUTE (paid)" if args.execute else "dry-run"
        print(f"running recipe: persona={args.persona} model={args.model} scale={args.scale} [{mode}]")
        capability_config = replace(
            get_capability_config(args.capability_suite),
            model_args=tuple(args.capability_model_arg),
        )
        result = pipeline.run(
            persona=args.persona,
            student_model=args.model,
            teacher_model=args.teacher,
            out_dir=out,
            config=get_config(args.scale),
            dry_run=not args.execute,
            run_eval=not args.no_eval,
            eval_merged_locally=args.eval_merged_local,
            condition=args.condition,
            run_capabilities=args.eval_capabilities,
            capability_config=capability_config,
            capability_model=args.capability_model,
        )
        print(f"run_id: {result.run_id}")
        print(f"dpo:    {result.dpo_checkpoint.sampler_path}")
        print(f"sft:    {result.sft_checkpoint.sampler_path}")
        print(f"final:  {result.final_checkpoint.sampler_path or result.final_checkpoint.local_path}")
        shift = result.persona_trait_shift
        if shift is not None:
            summary = result.shift_summary
            print(f"eval target: {result.eval_target}")
            print(
                f"persona '{args.persona}' net shift: {shift:+.1f} "
                f"(Δaligned {summary.get('aligned_mean_delta', 0.0):+.1f}, "
                f"Δopposing {summary.get('opposing_mean_delta', 0.0):+.1f})"
            )
            risers = ", ".join(f"{m['trait']} {m['delta']:+.0f}" for m in summary.get("top_increased", [])[:3])
            fallers = ", ".join(f"{m['trait']} {m['delta']:+.0f}" for m in summary.get("top_decreased", [])[:3])
            if risers:
                print(f"  top ↑: {risers}")
            if fallers:
                print(f"  top ↓: {fallers}")
        if result.capability_benchmarks:
            cap = result.capability_benchmarks
            print(f"capabilities: {cap.get('status')} [{cap.get('suite')}]")
            if cap.get("command_preview"):
                print(f"  command: {cap['command_preview']}")
            if cap.get("error"):
                print(f"  error: {cap['error']}")
        print(f"artifacts: {out}")
    elif args.command == "scaling":
        from experiments import scaling

        out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"scaling-{args.persona}"
        model_set = tuple(args.model_set) if args.model_set else models.SCALING_SET
        mode = "EXECUTE (paid)" if args.execute else "dry-run"
        print(f"scaling sweep: persona={args.persona} models={len(model_set)} scale={args.scale} [{mode}]")
        scaling.run_and_report(
            persona=args.persona,
            teacher_model=args.teacher,
            out_dir=out,
            model_set=model_set,
            config=get_config(args.scale),
            dry_run=not args.execute,
        )
        print((out / "report.md").read_text())
        print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
