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
        choices=("smoke", "quick", "paper-half", "paper"),
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
    run_cmd.add_argument("--scale", choices=("smoke", "quick", "paper-half", "paper"), default="smoke")
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
        "--no-merge",
        action="store_true",
        help="skip local DPO+SFT adapter merge and evaluate the SFT sampler directly",
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
    scaling_cmd.add_argument("persona")
    scaling_cmd.add_argument("--teacher", default=models.TEACHER_MODEL)
    scaling_cmd.add_argument("--scale", choices=("smoke", "quick", "paper-half", "paper"), default="smoke")
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
    return 0


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
