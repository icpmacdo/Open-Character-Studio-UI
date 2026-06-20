"""Command-line entry point.

    octt constitutions            list available personas
    octt show <persona>           print a constitution
    octt models                   list candidate models for the scaling study
    octt preflight                validate Tinker setup and estimated spend
"""

from __future__ import annotations

import argparse

from . import constitution, models, tinker_client
from .config import get_config


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
