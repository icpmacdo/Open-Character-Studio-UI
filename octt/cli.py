"""Command-line entry point.

    octt constitutions            list available personas
    octt show <persona>           print a constitution
    octt models                   list candidate models for the scaling study
    octt preflight                validate Tinker setup and estimated spend
    octt run <persona>            run the full recipe for one model/persona
    octt scaling <persona>        run the dense-vs-MoE sweep + report
    octt scaling --report-only D  rebuild D's report from banked results (free)
    octt spend                    what Tinker actually billed (official, not estimated)

``run`` and ``scaling`` default to a dry run (no spend); pass ``--execute`` to
hit the paid Tinker runtime. Always ``octt preflight`` before ``--execute``.

``--report-only`` is the one scaling mode that cannot spend anything: it reads a
finished run directory's ``eval_results.json``/``manifest.json`` and rewrites
``report.json`` + ``report.md``. Use it whenever the *summary* changed (trait
curation, confidence intervals) rather than the measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import constitution, models, tinker_client
from .config import RecipeConfig, for_scaling_study, get_capability_config, get_config
from .tinker_client import DEFAULT_OUTPUT_DIR


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _recipe_config_from_args(args: argparse.Namespace) -> RecipeConfig:
    cfg = get_config(args.scale)
    lora_rank = getattr(args, "lora_rank", None)
    if lora_rank is not None:
        cfg = replace(
            cfg,
            dpo=replace(cfg.dpo, lora_rank=lora_rank),
            sft=replace(cfg.sft, lora_rank=lora_rank),
        )
    learning_rate = getattr(args, "learning_rate", None)
    if learning_rate is not None:
        cfg = replace(
            cfg,
            dpo=replace(cfg.dpo, learning_rate=learning_rate),
            sft=replace(cfg.sft, learning_rate=learning_rate),
        )
    sft_epochs = getattr(args, "sft_epochs", None)
    if sft_epochs is not None:
        # SFT only: DPO is one epoch by construction (distillation.py pins
        # num_epochs=1), so exposing this on both stages would silently do nothing
        # to half of what the flag names.
        cfg = replace(cfg, sft=replace(cfg.sft, epochs=sft_epochs))
    if getattr(args, "no_merge", False):
        cfg = replace(cfg, merge_adapters=False)
    return cfg


def _recipe_label(cfg: RecipeConfig) -> str:
    merge = "merge" if cfg.merge_adapters else "no-merge"
    if cfg.dpo.lora_rank == cfg.sft.lora_rank:
        rank = f"rank{cfg.dpo.lora_rank}"
    else:
        rank = f"dpo-rank{cfg.dpo.lora_rank}/sft-rank{cfg.sft.lora_rank}"
    return f"{rank}, {merge}"


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
        choices=("smoke", "quick", "paper-half", "paper-half-uncapped", "paper"),
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
    preflight.add_argument(
        "--lora-rank",
        type=_positive_int,
        help="override both DPO and SFT LoRA rank for compatibility checks/runs",
    )
    preflight.add_argument(
        "--no-merge",
        action="store_true",
        help="skip local DPO+SFT adapter merge in the checked recipe",
    )
    preflight.add_argument(
        "--condition",
        choices=("adopt", "feels", "random", "all"),
        default="adopt",
        help="eval condition count for spend estimation; 'all' costs three budgets",
    )
    preflight.add_argument(
        "--judge",
        default=None,
        help="revealed-preferences judge model for spend estimation "
        "(default: the --teacher model; the judge dominates paper-scale cost)",
    )

    run_cmd = sub.add_parser("run", help="run the full recipe for one model/persona")
    run_cmd.add_argument("persona")
    run_cmd.add_argument("--model", default=models.DENSE_LADDER[0], help="student model id")
    run_cmd.add_argument("--teacher", default=models.TEACHER_MODEL)
    run_cmd.add_argument("--scale", choices=("smoke", "quick", "paper-half", "paper-half-uncapped", "paper"), default="smoke")
    run_cmd.add_argument("--out", default=None, help="output directory (default runs/<persona>)")
    run_cmd.add_argument("--execute", action="store_true", help="hit the paid runtime (default: dry run)")
    run_cmd.add_argument("--no-eval", action="store_true", help="skip the revealed-preferences eval")
    run_cmd.add_argument(
        "--lora-rank",
        type=_positive_int,
        help="override both DPO and SFT LoRA rank; use 32 for Ultra compatibility",
    )
    run_cmd.add_argument(
        "--learning-rate",
        type=float,
        help="override both DPO and SFT learning rate (e.g. 1e-4 to match the "
        "paper's effective update scale at rank 32 under Tinker's fixed alpha=32)",
    )
    run_cmd.add_argument(
        "--sft-epochs",
        type=int,
        help="passes over the introspection corpus (default 1). Raising this "
        "increases optimizer steps without changing the data, which is the "
        "training-strength axis a fixed recipe holds constant across model sizes",
    )
    run_cmd.add_argument(
        "--no-merge",
        action="store_true",
        help="skip local DPO+SFT adapter merge and evaluate the SFT sampler directly",
    )
    run_cmd.add_argument(
        "--split-cache-dir",
        default=None,
        help="shared split response/judgment cache (octt.eval_cache). At full "
        "scale the trait pool and therefore the schedule are persona-independent, "
        "so pointing every persona's run at one directory makes the base-model "
        "half of the eval a single banked artifact instead of re-paying per "
        "persona. Mutually exclusive with the per-run combined cache (default).",
    )
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
        choices=("adopt", "feels", "random", "all"),
        default="adopt",
        help="embodiment-instruction variant for the eval (paper template 1/2/3), "
        "or 'all' to repeat the full judgment budget per condition as the paper does",
    )
    run_cmd.add_argument(
        "--judge",
        default=None,
        help="revealed-preferences judge model (default: the --teacher model). "
        "Self-distillation runs (teacher == student) must pass an external judge",
    )

    scaling_cmd = sub.add_parser("scaling", help="run the dense-vs-MoE sweep and write a report")
    scaling_cmd.add_argument(
        "persona",
        nargs="?",
        help="persona to sweep; optional with --report-only (read from the banked results)",
    )
    scaling_cmd.add_argument(
        "--report-only",
        metavar="RUN_DIR",
        help="FREE: skip the sweep entirely and rebuild report.json/report.md from "
        "RUN_DIR's banked per-rung eval_results.json. No Tinker, no sampling, no "
        "network, no spend — use it after a curation or summary change instead of "
        "re-running a paid sweep. All other sweep flags are ignored",
    )
    scaling_cmd.add_argument(
        "--report-out",
        metavar="DIR",
        help="with --report-only: write the rebuilt report here instead of into "
        "RUN_DIR, leaving the original report.json (the phase gate's marker) untouched",
    )
    scaling_cmd.add_argument("--teacher", default=models.TEACHER_MODEL)
    scaling_cmd.add_argument("--scale", choices=("smoke", "quick", "paper-half", "paper-half-uncapped", "paper"), default="smoke")
    scaling_cmd.add_argument("--out", default=None, help="output directory (default runs/scaling-<persona>)")
    scaling_cmd.add_argument(
        "--model", action="append", dest="model_set",
        help="model to include; repeat to override the default cost-ordered sweep",
    )
    scaling_cmd.add_argument("--execute", action="store_true", help="hit the paid runtime (default: dry run)")
    scaling_cmd.add_argument(
        "--lora-rank",
        type=_positive_int,
        help="override both DPO and SFT LoRA rank for all models in the sweep",
    )
    scaling_cmd.add_argument(
        "--no-merge",
        action="store_true",
        help="skip local DPO+SFT adapter merge and evaluate SFT samplers directly",
    )
    scaling_cmd.add_argument(
        "--eval-merged-local",
        action="store_true",
        help="evaluate local merged adapters via transformers+peft where feasible",
    )
    scaling_cmd.add_argument(
        "--condition",
        choices=("adopt", "feels", "random", "all"),
        default="adopt",
        help="embodiment-instruction variant for the revealed-preferences eval, "
        "or 'all' to repeat the full judgment budget per condition",
    )
    scaling_cmd.add_argument(
        "--judge",
        default=None,
        help="revealed-preferences judge model for every rung (default: the "
        "--teacher model). One judge across the sweep keeps rungs comparable",
    )
    scaling_cmd.add_argument(
        "--eval-capabilities",
        action="store_true",
        help="run or preview the opt-in LightEval capability benchmark harness per model",
    )
    scaling_cmd.add_argument(
        "--capability-suite",
        choices=("smoke", "full"),
        default="smoke",
        help="LightEval capability suite for scaling runs",
    )
    scaling_cmd.add_argument(
        "--capability-model",
        help="explicit Hugging Face/local model reference for LightEval; mainly useful "
        "when scaling is restricted to one model",
    )
    scaling_cmd.add_argument(
        "--capability-model-arg",
        action="append",
        default=[],
        help="extra LightEval model arg as key=value; repeat for dtype, batch_size, etc.",
    )

    gen_prompts_cmd = sub.add_parser(
        "gen-prompts",
        help="generate the App F constitution-relevant prompt set (~50/assertion)",
    )
    gen_prompts_cmd.add_argument("persona")
    gen_prompts_cmd.add_argument("--generator", default=models.TEACHER_MODEL)
    gen_prompts_cmd.add_argument("--per-assertion", type=_positive_int, default=50)
    gen_prompts_cmd.add_argument(
        "--execute",
        action="store_true",
        help="hit the paid runtime and write the canonical prompt file that "
        "dpo_prompts consumes (default: dry-run stub written to a .preview path)",
    )
    gen_prompts_cmd.add_argument(
        "--from-paper",
        action="store_true",
        help="import the vendored paper-original App F library "
        "(constitutions/paper_prompts/) instead of generating — free and "
        "offline, writes the canonical file directly; paper personas only",
    )

    robustness_cmd = sub.add_parser(
        "robustness",
        help="adversarial-robustness + prefill evals over finished runs (paper 3.2/3.3)",
    )
    robustness_cmd.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="PERSONA=RUN_DIR",
        help="finished recipe run to include; repeat — the persona classifier "
        "needs at least two personas to be meaningful",
    )
    robustness_cmd.add_argument("--model", required=True, help="student model id shared by the runs")
    robustness_cmd.add_argument(
        "--out", default=None, help="responses/report dir (default runs/robustness-<model>)"
    )
    robustness_cmd.add_argument("--execute", action="store_true", help="hit the paid runtime")
    robustness_cmd.add_argument(
        "--num-prompts",
        type=_positive_int,
        default=None,
        help="Pure-Dove questions per split (default: the official 500; "
        "lower it for a cheap live validation pass)",
    )
    robustness_cmd.add_argument(
        "--skip-classifier",
        action="store_true",
        help="generate responses only; skip the ModernBERT classifier (local-eval extra)",
    )

    coherence_cmd = sub.add_parser(
        "coherence",
        help="coherence LLM-judge win-rate between two methods' responses (paper 3.4)",
    )
    coherence_cmd.add_argument("persona")
    coherence_cmd.add_argument(
        "--responses-dir",
        required=True,
        help="robustness responses dir holding the default-split JSONLs",
    )
    coherence_cmd.add_argument("--method-one", default="dpo")
    coherence_cmd.add_argument("--method-two", default="final")
    coherence_cmd.add_argument("--judge", default=models.TEACHER_MODEL)
    coherence_cmd.add_argument("--execute", action="store_true", help="hit the paid runtime")

    spend_cmd = sub.add_parser(
        "spend",
        help="report what Tinker actually billed (official invoice data, not estimates)",
    )
    spend_cmd.add_argument(
        "--month",
        help="billing month to report, YYYY-MM (default: the current UTC month)",
    )
    spend_cmd.add_argument("--since", help="window start, YYYY-MM-DD (UTC); overrides --month")
    spend_cmd.add_argument("--until", help="window end, exclusive, YYYY-MM-DD (UTC)")
    spend_cmd.add_argument(
        "--run",
        dest="run_dirs",
        action="append",
        help="attribute billed spend to this run directory; repeat for several",
    )
    spend_cmd.add_argument(
        "--all-runs",
        action="store_true",
        help="attribute spend to every run under --runs-root that overlaps the window",
    )
    spend_cmd.add_argument(
        "--runs-root",
        default="runs",
        help="directory holding run dirs, used to detect contended attribution",
    )
    spend_cmd.add_argument(
        "--by",
        choices=("model", "charge", "day", "model-charge"),
        default="model-charge",
        help="grouping for the breakdown table",
    )
    spend_cmd.add_argument(
        "--snapshot",
        help="read a browser snapshot JSON instead of hitting the API (no cookie needed)",
    )
    spend_cmd.add_argument(
        "--snippet",
        action="store_true",
        help="print the DevTools snippet that writes a snapshot, then exit",
    )
    spend_cmd.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    spend_cmd.add_argument(
        "--check-prices",
        action="store_true",
        help="exit 2 if billed unit prices drift from the pinned rate card in models.py",
    )
    spend_cmd.add_argument(
        "--max-gross-usd",
        type=float,
        help="exit 2 if gross billed spend in the window exceeds this (pre-spend gate)",
    )
    spend_cmd.add_argument(
        "--min-credit-usd",
        type=float,
        help="exit 2 if remaining grant credit has fallen below this (pre-spend gate)",
    )

    migrate_cmd = sub.add_parser(
        "eval-cache-migrate",
        help="convert a legacy combined eval cache into split response/judgment "
        "caches (offline; never modifies the legacy file)",
    )
    migrate_cmd.add_argument("legacy", help="path to the legacy combined cache JSONL")
    migrate_cmd.add_argument(
        "--out",
        required=True,
        help="fresh directory for responses.jsonl + judgments.jsonl "
        "(refuses to overwrite existing split caches)",
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
        cfg = _recipe_config_from_args(args)
        report = tinker_client.build_preflight_report(
            student_models=student_models,
            teacher_model=args.teacher,
            config=cfg,
            dry_run=args.dry_run,
            budget_usd=args.budget,
            eval_conditions=3 if args.condition == "all" else 1,
            judge_model=args.judge,
        )
        status = "OK" if report.ok else "BLOCKED"
        api_key = "skipped (dry-run)" if report.dry_run else ("yes" if report.api_key_set else "no")

        print(f"status: {status}")
        print(f"scale: {args.scale}")
        print(f"recipe: {_recipe_label(cfg)}")
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
        cfg = _recipe_config_from_args(args)
        print(
            f"running recipe: persona={args.persona} model={args.model} "
            f"scale={args.scale} recipe={_recipe_label(cfg)} [{mode}]"
        )
        capability_config = replace(
            get_capability_config(args.capability_suite),
            model_args=tuple(args.capability_model_arg),
        )
        result = pipeline.run(
            persona=args.persona,
            student_model=args.model,
            teacher_model=args.teacher,
            out_dir=out,
            config=cfg,
            dry_run=not args.execute,
            run_eval=not args.no_eval,
            eval_merged_locally=args.eval_merged_local,
            condition=args.condition,
            judge_model=args.judge,
            run_capabilities=args.eval_capabilities,
            capability_config=capability_config,
            capability_model=args.capability_model,
            split_cache_dir=args.split_cache_dir,
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
        if args.report_only:
            # Deliberately short-circuits before anything that can spend: no
            # config, no runtime, and `experiments.scaling` (hence the pipeline)
            # is never imported.
            if args.execute:
                parser.error("--report-only rebuilds a banked report; it cannot --execute")
            from .reporting import rebuild_report

            run_dir = Path(args.report_only)
            report_dir = Path(args.report_out) if args.report_out else run_dir
            marker = report_dir / "report.json"
            minted_marker = report_dir == run_dir and not marker.is_file()
            try:
                payload = rebuild_report(run_dir, out_dir=report_dir, persona=args.persona)
            except ValueError as exc:
                parser.error(str(exc))
            print((report_dir / "report.md").read_text())
            print(f"report: {report_dir / 'report.md'} (rebuilt from {run_dir}, no spend)")
            if minted_marker:
                # report.json is scripts/octt_plan.sh's skip-if-done marker. The
                # rebuild reads rung directories, so it cannot know how many rungs
                # the sweep intended: writing the first one here can retire a paid
                # phase whose later rungs never started.
                print(
                    f"WARNING: {marker} did not exist — this rebuild just created the "
                    "phase gate's completion marker for a sweep that never wrote one. "
                    "Check every intended rung is a row above, or rebuild with "
                    "--report-out to leave the gate alone."
                )
            failed = [row for row in payload["rows"] if row.get("error")]
            if failed:
                for row in failed:
                    print(f"INCOMPLETE rung {row['model']}: {row['error']}")
                return 1
            return 0

        from experiments import scaling

        if not args.persona:
            parser.error("a persona is required (or --report-only RUN_DIR)")
        if args.report_out:
            parser.error("--report-out only applies with --report-only")
        out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"scaling-{args.persona}"
        model_set = tuple(args.model_set) if args.model_set else models.SCALING_SET
        mode = "EXECUTE (paid)" if args.execute else "dry-run"
        cfg = _recipe_config_from_args(args)
        if getattr(args, "lora_rank", None) is None:
            # Uniform-rank study policy: every rung at rank 32 (Ultra's cap)
            # with lr 1e-4, matching the paper's effective update scale under
            # Tinker's fixed alpha=32. See config.for_scaling_study.
            cfg = for_scaling_study(cfg)
            print(
                "scaling-study policy: uniform lora_rank=32, lr=1e-4 "
                "(pass --lora-rank to override)"
            )
        print(
            f"scaling sweep: persona={args.persona} models={len(model_set)} "
            f"scale={args.scale} recipe={_recipe_label(cfg)} [{mode}]"
        )
        capability_config = replace(
            get_capability_config(args.capability_suite),
            model_args=tuple(args.capability_model_arg),
        )
        sweep_runs = scaling.run_and_report(
            persona=args.persona,
            teacher_model=args.teacher,
            out_dir=out,
            model_set=model_set,
            config=cfg,
            dry_run=not args.execute,
            eval_merged_locally=args.eval_merged_local,
            condition=args.condition,
            judge_model=args.judge,
            run_capabilities=args.eval_capabilities,
            capability_config=capability_config,
            capability_model=args.capability_model,
        )
        print((out / "report.md").read_text())
        print(f"report: {out / 'report.md'}")
        failed = [r for r in sweep_runs if r.error]
        if failed:
            for r in failed:
                print(f"FAILED rung {r.spec.tinker_id}: {r.error}")
            return 1
    elif args.command == "gen-prompts":
        from . import prompt_gen

        c = constitution.load(args.persona)
        if args.from_paper:
            # Vendored paper-original library: free, offline, and already real
            # data — writes the canonical file directly, no --execute needed.
            path = prompt_gen.import_paper_prompts(c)
            print(f"prompts: {path} (imported paper-original App F library)")
            return 0
        dry = not args.execute
        out_path = prompt_gen.default_prompts_path(args.persona)
        if dry:
            # Never write stub prompts to the canonical path a real run consumes.
            out_path = out_path.with_suffix(".preview.json")
        runtime = tinker_client.create_runtime(
            (args.generator,), config=tinker_client.TinkerClientConfig(dry_run=dry)
        )
        path = prompt_gen.generate_constitution_prompts(
            c, runtime, generator_model=args.generator,
            per_assertion=args.per_assertion, out_path=out_path, offline=dry,
        )
        mode = "" if args.execute else " (dry-run preview; --execute writes the canonical file)"
        print(f"prompts: {path}{mode}")
    elif args.command == "robustness":
        from . import robustness

        run_dirs: dict[str, Path] = {}
        for spec_arg in args.run:
            persona_name, _, run_dir = spec_arg.partition("=")
            if not run_dir:
                parser.error("--run expects PERSONA=RUN_DIR")
            run_dirs[persona_name] = Path(run_dir)
        dry = not args.execute
        out = (
            Path(args.out) if args.out
            else DEFAULT_OUTPUT_DIR / f"robustness-{args.model.replace('/', '-')}"
        )
        runtime = tinker_client.create_runtime(
            (args.model,), config=tinker_client.TinkerClientConfig(dry_run=dry)
        )
        rcfg = robustness.RobustnessConfig()
        if args.num_prompts is not None:
            rcfg = replace(rcfg, num_prompts=args.num_prompts)
        robustness.generate_base_first_turns(
            args.model, out, runtime, config=rcfg, offline=dry
        )
        persona_methods: dict[str, set[str]] = {}
        for persona_name, run_dir in run_dirs.items():
            methods = _run_dir_methods(run_dir)
            persona_methods[persona_name] = set(methods)
            robustness.generate_responses(
                persona_name, args.model, methods, out, runtime, config=rcfg, offline=dry
            )
            robustness.generate_prefill_responses(
                persona_name, args.model, methods, out, runtime, config=rcfg, offline=dry
            )
        print(f"responses: {out}")
        if args.skip_classifier:
            return 0
        if len(run_dirs) < 2:
            print(
                "warning: the persona classifier needs >=2 personas to be "
                "meaningful; add more --run entries (responses were generated)"
            )
        # The classifier needs every (persona, method) file, so evaluate only
        # methods present in ALL runs (a run that stopped after DPO has no
        # 'final'); union would abort on the missing files.
        shared_methods = set.intersection(*persona_methods.values())
        skipped_methods = sorted(set().union(*persona_methods.values()) - shared_methods)
        if skipped_methods:
            print(
                "warning: skipping methods missing from some runs: "
                + ", ".join(skipped_methods)
            )
        payload = robustness.evaluate_robustness(
            sorted(run_dirs), sorted(shared_methods), out,
            out / "robustness_report.json", config=rcfg, dry_run=dry,
        )
        print(f"robustness report: {out / 'robustness_report.json'}")
        print(f"status: {payload.get('status', 'ok')}")
        for method, scores in (payload.get("macro_f1") or {}).items():
            print(f"  {method}: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    elif args.command == "coherence":
        from . import coherence, robustness

        dry = not args.execute
        responses_dir = Path(args.responses_dir)
        pairs = {}
        for label in (args.method_one, args.method_two):
            path = robustness.response_path(responses_dir, args.persona, label, "default")
            if not path.exists():
                parser.error(f"missing default-split responses for {label!r}: {path}")
            pairs[label] = {
                row["prompt"]: row["response"] for row in robustness._load_rows(path)
            }
        shared = [p for p in pairs[args.method_one] if p in pairs[args.method_two]]
        if not shared:
            parser.error("no shared prompts between the two methods' response files")
        runtime = tinker_client.create_runtime(
            (args.judge,), config=tinker_client.TinkerClientConfig(dry_run=dry)
        )
        result = coherence.compare(
            [pairs[args.method_one][p] for p in shared],
            [pairs[args.method_two][p] for p in shared],
            shared,
            args.persona,
            runtime,
            judge_model=args.judge,
            cache_path=responses_dir
            / f"coherence-{args.persona}-{args.method_one}-vs-{args.method_two}.jsonl",
            offline=dry,
        )
        win = result.get("win_rate")
        win_label = f"{win:.3f}" if isinstance(win, (int, float)) else "n/a (no retained judgments)"
        print(
            f"coherence win-rate of {args.method_two!r} over {args.method_one!r} "
            f"for '{args.persona}': {win_label} "
            f"(retained {result.get('retained')}/{result.get('total')})"
        )
    elif args.command == "spend":
        return _cmd_spend(args)
    elif args.command == "eval-cache-migrate":
        from . import eval_cache

        report = eval_cache.migrate_legacy_cache(Path(args.legacy), Path(args.out))
        print(report.summary())
    return 0


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    """UTC ``[start, end)`` for a ``YYYY-MM`` string."""
    year, mon = (int(part) for part in month.split("-", 1))
    start = datetime(year, mon, 1, tzinfo=UTC)
    end = datetime(year + (mon == 12), (mon % 12) + 1, 1, tzinfo=UTC)
    return start, end


def _spend_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.since:
        start = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
        end = (
            datetime.fromisoformat(args.until).replace(tzinfo=UTC)
            if args.until
            else start + timedelta(days=1)
        )
        return start, end
    return _month_bounds(args.month or datetime.now(UTC).strftime("%Y-%m"))


def _cmd_spend(args: argparse.Namespace) -> int:
    """Report officially billed Tinker spend, optionally attributed to runs.

    Every number printed here comes from Tinker's invoice API, not from the
    ``models.py`` rate card — that separation is the whole point, since it lets
    ``--check-prices`` catch the rate card going stale underneath ``preflight``.
    """
    import json as _json

    from . import billing

    start, end = _spend_window(args)
    # Per-run attribution needs hourly resolution; a month overview does not,
    # and daily windows are one request instead of ~700.
    want_runs = bool(args.run_dirs or args.all_runs)
    window_size = billing.WINDOW_HOUR if want_runs else billing.WINDOW_DAY

    if args.snippet:
        print(billing.browser_snippet(start, end, window_size))
        return 0

    try:
        if args.snapshot:
            rows, grants = billing.load_snapshot(Path(args.snapshot))
        else:
            client = billing.BillingClient()
            rows = client.fetch_breakdowns(start, end, window_size)
            grants = client.fetch_credits()
    except billing.BillingAuthError as exc:
        print(f"auth error: {exc}")
        return 2
    except billing.BillingFetchError as exc:
        print(f"fetch error: {exc}")
        return 2

    overall = billing.summarize(rows)
    now = datetime.now(UTC)
    active = [g for g in grants if g.is_active(now)]

    # Grant credit is consumed over the grant's lifetime, not over the month
    # being reported. Re-query from the grant start so the balance matches the
    # console; a window-only subtraction overstates it by everything spent
    # earlier. Snapshots can only cover what they captured, so they say so.
    grant_start = billing.grant_period_start(active, now)
    consumption_rows, complete = rows, False
    if active and grant_start and not args.snapshot:
        try:
            consumption_rows = client.fetch_breakdowns(grant_start, now, billing.WINDOW_DAY)
            complete = True
        except (billing.BillingAuthError, billing.BillingFetchError) as exc:
            print(f"warning: could not read the full grant period ({exc}); balance is an upper bound")
    balance = billing.grant_balance(active, consumption_rows, complete=complete)

    attributions = []
    if want_runs:
        known = billing.discover_run_windows(Path(args.runs_root))
        if args.all_runs:
            targets = [w for w in known if w.start < end and w.end > start]
        else:
            targets = [billing.load_run_window(Path(d)) for d in args.run_dirs]
        attributions = [billing.attribute_run(w, rows, known) for w in targets]

    drifts = billing.price_drift(rows)

    if args.json:
        print(
            _json.dumps(
                {
                    "window": {"start": start.isoformat(), "end": end.isoformat()},
                    "window_size": window_size,
                    "gross_usd": round(overall.gross_usd, 4),
                    "credits_usd": round(overall.credits_usd, 4),
                    "net_usd": round(overall.net_usd, 4),
                    "token_millions": round(overall.token_millions, 6),
                    "storage_gb_months": round(overall.storage_gb_months, 6),
                    "grants": [
                        {
                            "product": g.product,
                            "amount_usd": g.amount_usd,
                            "ending_before": g.ending_before.isoformat() if g.ending_before else None,
                        }
                        for g in active
                    ],
                    "grant_granted_usd": round(balance.granted_usd, 4),
                    "grant_consumed_usd": round(balance.consumed_usd, 4),
                    "grant_remaining_usd": round(balance.remaining_usd, 4),
                    "grant_remaining_is_complete": balance.complete,
                    "by_model_charge": [
                        {
                            "base_model": key[0],
                            "charge": key[1],
                            "quantity": round(
                                sum(r.quantity for r in bucket.charged_rows), 6
                            ),
                            "gross_usd": round(bucket.gross_usd, 4),
                        }
                        for key, bucket in overall.ranked("base_model", "charge")
                    ],
                    "runs": [
                        {
                            "run_id": a.window.run_id,
                            "run_dir": str(a.window.run_dir),
                            "start": a.window.start.isoformat(),
                            "end": a.window.end.isoformat(),
                            "gross_usd": round(a.summary.gross_usd, 4),
                            "token_millions": round(a.summary.token_millions, 6),
                            "exclusive": a.is_exclusive,
                            "contended_runs": list(a.contended_runs),
                        }
                        for a in attributions
                    ],
                    "price_drift": [
                        {
                            "base_model": d.base_model,
                            "charge": d.charge,
                            "billed_usd_per_mtok": d.billed_usd_per_mtok,
                            "pinned_usd_per_mtok": d.pinned_usd_per_mtok,
                            "delta_pct": round(d.delta_pct, 3),
                        }
                        for d in drifts
                    ],
                },
                indent=2,
            )
        )
    else:
        if args.snapshot:
            # A snapshot covers whatever window it was captured over, which need
            # not be the one asked for. Print what the data actually spans so a
            # stale file can't masquerade as this month.
            stamps = [r.window_start for r in rows if r.window_start]
            span = (
                f"{min(stamps):%Y-%m-%d %H:%MZ} -> {max(stamps):%Y-%m-%d %H:%MZ}"
                if stamps
                else "empty"
            )
            print(f"window: {span} (spanned by the snapshot)")
            print(f"source: snapshot {args.snapshot}")
        else:
            print(f"window: {start:%Y-%m-%d} -> {end:%Y-%m-%d} (UTC, {window_size.lower()}ly)")
            print("source: tinker billing API")
        print()
        print(f"gross billed      ${overall.gross_usd:>10,.2f}")
        print(f"credits applied   ${overall.credits_usd:>10,.2f}")
        print(f"net (out of pocket) ${overall.net_usd:>8,.2f}")
        print(f"tokens            {overall.token_millions:>11,.2f}M")
        print(f"checkpoint storage{overall.storage_gb_months:>11,.2f} GB-months")
        if active:
            print()
            for grant in active:
                expiry = grant.ending_before.strftime("%Y-%m-%d") if grant.ending_before else "n/a"
                print(f"grant: {grant.product} ${grant.amount_usd:,.2f} (expires {expiry})")
            caveat = (
                "" if balance.complete else "  (upper bound: consumption before this window unseen)"
            )
            print(
                f"grant remaining:  ${balance.remaining_usd:,.2f} of "
                f"${balance.granted_usd:,.2f}{caveat}"
            )

        print()
        _print_spend_table(overall, args.by)

        for attribution in attributions:
            print()
            _print_run_attribution(attribution)

        if drifts:
            print()
            print("RATE-CARD DRIFT (billed vs pinned in octt/models.py):")
            for drift in drifts:
                flag = "UNDER-PINNED" if drift.underestimates else "conservative"
                print(
                    f"  {drift.base_model:<44} {drift.charge:<34} "
                    f"billed ${drift.billed_usd_per_mtok:.4f}/Mtok  "
                    f"pinned ${drift.pinned_usd_per_mtok:.4f}  "
                    f"({drift.delta_pct:+.1f}%, {flag})"
                )
            if any(d.underestimates for d in drifts):
                print("  -> UNDER-PINNED rates make preflight under-estimate spend; fix models.py")
            else:
                print("  -> all conservative: preflight over-estimates, so the budget gate holds")

    failures = []
    dangerous = [d for d in drifts if d.underestimates]
    if args.check_prices and dangerous:
        failures.append(
            f"{len(dangerous)} billed rate(s) exceed the pinned rate card, "
            "so preflight under-estimates spend"
        )
    if args.max_gross_usd is not None and overall.gross_usd > args.max_gross_usd:
        failures.append(
            f"gross billed ${overall.gross_usd:,.2f} exceeds "
            f"--max-gross-usd ${args.max_gross_usd:,.2f}"
        )
    if args.min_credit_usd is not None and balance.remaining_usd < args.min_credit_usd:
        # An incomplete balance is an upper bound, so tripping the floor on one
        # still means the real balance is at least that low — safe to block.
        failures.append(
            f"grant remaining ${balance.remaining_usd:,.2f} is below "
            f"--min-credit-usd ${args.min_credit_usd:,.2f}"
            + ("" if balance.complete else " (and the true balance is no higher)")
        )
    if failures:
        print()
        for failure in failures:
            print(f"BLOCKED: {failure}")
        return 2
    return 0


def _print_spend_table(summary, group: str) -> None:
    """Print the breakdown table for the requested grouping."""
    keys = {
        "model": ("base_model",),
        "charge": ("charge",),
        "day": ("day",),
        "model-charge": ("base_model", "charge"),
    }[group]
    ranked = summary.ranked(*keys)
    if not ranked:
        print("(no billed usage in this window)")
        return

    width = max(len(" / ".join(str(p) for p in key)) for key, _ in ranked)
    width = min(max(width, 12), 72)
    print(f"{'':<{width}}  {'quantity':>12}  {'billed':>10}")
    for key, bucket in ranked:
        label = " / ".join("-" if p is None else str(p) for p in key)
        quantity = sum(row.quantity for row in bucket.charged_rows)
        print(f"{label:<{width}}  {quantity:>12,.3f}  ${bucket.gross_usd:>9,.2f}")


def _print_run_attribution(attribution) -> None:
    """Print one run's billed spend and, honestly, how ambiguous it is."""
    window = attribution.window
    summary = attribution.summary
    print(f"run: {window.run_id}  ({window.run_dir})")
    print(
        f"  window {window.start:%Y-%m-%d %H:%MZ} -> {window.end:%Y-%m-%d %H:%MZ}"
        f"   models {', '.join(sorted(window.base_models)) or '(none recorded)'}"
    )
    if not window.is_real:
        print(f"  execution_mode={window.execution_mode!r}: this run spent nothing")
    for key, bucket in summary.ranked("base_model", "charge"):
        quantity = sum(row.quantity for row in bucket.charged_rows)
        model = key[0] or "-"
        print(f"    {model:<44} {key[1]:<34} {quantity:>10,.3f}  ${bucket.gross_usd:>9,.2f}")
    print(f"  billed to this run: ${summary.gross_usd:,.2f}  ({summary.token_millions:,.2f}M tokens)")
    if attribution.contended_runs:
        print(
            f"  CONTENDED: shares hours and models with "
            f"{', '.join(attribution.contended_runs)}."
        )
        print(
            "  Tinker bills per (hour, base_model) with no run id, so the figure above "
            "is an upper bound covering all of them."
        )
    if attribution.excluded_models:
        print(f"  excluded (other models billed in the same hours): "
              f"{', '.join(attribution.excluded_models)}")


def _run_dir_methods(run_dir: Path) -> dict[str, str | None]:
    """Derive robustness eval targets from a finished run's manifest.

    ``base`` is always included (no adapter); ``dpo`` is the post-distillation
    sampler; ``final`` is the character-trained target (merged local adapter
    when it exists, else the samplable SFT/merge checkpoint).
    """
    import json

    data = json.loads((Path(run_dir) / "manifest.json").read_text())
    stages = data.get("stages", {})
    methods: dict[str, str | None] = {"base": None}
    dpo = (stages.get("dpo") or {}).get("sampler_path")
    if dpo:
        methods["dpo"] = dpo
    merge_rec = stages.get("merge") or {}
    local_merge = merge_rec.get("local_path")
    if local_merge and not Path(local_merge).exists():
        # Merged exports get pruned to free disk; the manifest entry outlives
        # the files. Fall back to the samplable checkpoints instead of handing
        # a dangling path to the local sampler.
        print(f"warning: merged adapter {local_merge} pruned from disk; using sampler fallback")
        local_merge = None
    final = (
        local_merge
        or merge_rec.get("sampler_path")
        or (stages.get("sft") or {}).get("sampler_path")
    )
    if final:
        methods["final"] = final
    return methods


if __name__ == "__main__":
    raise SystemExit(main())
