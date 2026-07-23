"""Stage orchestration: constitution -> DPO -> introspection -> merge -> eval.

This wires the stages together for a single model/persona pair, recording every
checkpoint in a :class:`~octt.manifest.RunManifest` so a crashed or repeated run
*skips finished stages* instead of re-sampling the teacher (the central rule of
``docs/COST_CONTROLS.md``). The scaling study (``experiments/scaling.py``) runs
this once per model with the recipe fixed.

Tinker constraint note: Tinker is LoRA-only with no adapter re-upload, so the
linear-merged adapter (paper Section 2.4 "Public Release") is produced as a
*local* artifact for HF release / off-Tinker serving. This costs nothing for
paper fidelity: the paper's Section 3 experiments all evaluate the
post-introspection checkpoint — served natively on Tinker (``sft-proxy`` /
``sft-direct``) — not the merged composite, which the paper never evaluates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import (
    capabilities,
    distillation,
    evaluation,
    introspection,
    manifest,
    merge,
    tinker_client,
    trait_profiles,
)
from .config import CapabilityEvalConfig, RecipeConfig, get_capability_config, get_config
from .constitution import load
from .manifest import StageCheckpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    persona: str
    student_model: str
    run_id: str
    out_dir: Path
    dpo_checkpoint: StageCheckpoint
    sft_checkpoint: StageCheckpoint
    final_checkpoint: StageCheckpoint
    base_elo: dict[str, float] = field(default_factory=dict)
    trained_elo: dict[str, float] = field(default_factory=dict)
    shift_summary: dict = field(default_factory=dict)
    eval_target: str | None = None
    recipe: dict = field(default_factory=dict)
    capability_benchmarks: dict = field(default_factory=dict)
    # Per-condition eval results when more than one embodiment condition runs
    # ({condition: {base_elo, trained_elo, shift_summary}}). The flat
    # base_elo/trained_elo/shift_summary fields above hold the FIRST condition.
    condition_results: dict = field(default_factory=dict)

    @property
    def persona_trait_shift(self) -> float | None:
        """Net revealed-preference shift = mean(Δaligned) - mean(Δopposing).

        The paper measures a persona's effect as the Elo change across the traits
        it pulls toward and away from (Figure 3), not a single self-named trait.
        See :func:`octt.trait_profiles.summarize_shift`.
        """
        return self.shift_summary.get("net_shift")


def _eval_plan(
    final: StageCheckpoint, *, dry_run: bool, eval_merged_locally: bool
) -> tuple[str | None, str | None, str]:
    """Decide what to evaluate for the character-trained model.

    Returns ``(sampler_path, local_adapter_dir, eval_target)``. The paper's
    Section 3 experiments all evaluate the post-introspection checkpoint, which
    the sequential SFT sampler serves natively on Tinker (``sft-proxy`` /
    ``sft-direct``). The linear merge (1.0·DPO + 0.25·SFT) is the paper's
    *release-only* artifact — never evaluated there. Tinker has no adapter
    re-upload, so it stays a local export unless ``eval_merged_locally`` serves
    it off-Tinker (feasible only for small rungs). The choice is always recorded.
    """
    if final.extra.get("merge_skipped") and final.sampler_path:
        return final.sampler_path, None, "sft-direct"
    if final.sampler_path:
        return final.sampler_path, None, "dry-run" if dry_run else "sampler"
    if eval_merged_locally and final.local_path and not dry_run:
        return None, final.local_path, "merged-local"
    proxy = final.extra.get("sft_sampler") or final.extra.get("dpo_sampler")
    label = (
        "sft-proxy" if final.extra.get("sft_sampler")
        else "dpo-proxy" if final.extra.get("dpo_sampler")
        else "none"
    )
    if proxy and not dry_run:
        if label == "sft-proxy":
            logger.info(
                "Evaluating the post-introspection SFT checkpoint %s (sft-proxy) — "
                "the artifact the paper's Section 3 experiments evaluate. The local "
                "merge is the release-only composite (Tinker has no adapter re-upload).",
                proxy,
            )
        else:
            logger.warning(
                "SFT sampler missing; falling back to %s (%s), which lacks the "
                "introspection stage and is NOT the paper's evaluated artifact.",
                proxy,
                label,
            )
    return proxy, None, label


def _sft_direct_checkpoint(sft_ckpt: StageCheckpoint) -> StageCheckpoint:
    """Final checkpoint for --no-merge runs: the SFT checkpoint served directly.

    With the official sequential recipe (SFTConfig.init_from_dpo) this carries
    distillation + introspection at full weight — exactly the composition the
    paper's Section 3 experiments evaluate. The only deviation is from the
    *released* composition (introspection at 1.0 instead of 0.25), which the
    paper never evaluates (recorded via eval_target='sft-direct').
    """
    extra = dict(sft_ckpt.extra)
    extra.update(
        {
            "merge_skipped": True,
            "source_stage": "sft",
            "sft_sampler": sft_ckpt.sampler_path,
        }
    )
    return StageCheckpoint(
        sampler_path=sft_ckpt.sampler_path,
        state_path=sft_ckpt.state_path,
        local_path=sft_ckpt.local_path,
        step=sft_ckpt.step,
        config_hash=sft_ckpt.config_hash,
        extra=extra,
    )


def _recipe_metadata(cfg: RecipeConfig) -> dict:
    return {
        "config_hash": manifest.config_hash(cfg),
        "merge_adapters": cfg.merge_adapters,
        "dpo_lora_rank": cfg.dpo.lora_rank,
        "dpo_lora_alpha": cfg.dpo.lora_alpha,
        "sft_lora_rank": cfg.sft.lora_rank,
        "sft_lora_alpha": cfg.sft.lora_alpha,
    }


def _verify_checkpoint(runtime: tinker_client.TinkerRuntime, model: str, ckpt: StageCheckpoint) -> None:
    """Round-trip gate: sample one short completion from a fresh sampler.

    Trivial in dry-run; in real mode a failure aborts the run rather than
    discovering downstream that nothing is loadable (COST_CONTROLS).
    """
    if runtime.config.dry_run or not ckpt.sampler_path:
        return
    from . import generation

    sampler = generation.make_sampler(runtime, model, model_path=ckpt.sampler_path, max_tokens=8)
    out = generation.complete_many(sampler, [[{"role": "user", "content": "Say hello."}]])
    if not out or not out[0]:
        raise RuntimeError(f"Round-trip verification failed for {ckpt.sampler_path}")


def run(
    persona: str,
    student_model: str,
    teacher_model: str,
    out_dir: Path,
    config: RecipeConfig | None = None,
    dry_run: bool = False,
    tinker_config: tinker_client.TinkerClientConfig | None = None,
    *,
    offline: bool | None = None,
    run_eval: bool = True,
    eval_merged_locally: bool = False,
    condition: str = evaluation.DEFAULT_CONDITION,
    judge_model: str | None = None,
    run_capabilities: bool = False,
    capability_config: CapabilityEvalConfig | None = None,
    capability_model: str | None = None,
) -> PipelineResult:
    """Run the full recipe for one model/persona pair. Returns checkpoints + Elo.

    ``condition`` selects the embodiment-instruction variant for the eval:
    one of ``adopt`` / ``feels`` / ``random``, or ``all`` to repeat the full
    judgment budget once per condition as the paper does (Section 3.1 — 25k
    judgments PER condition; budget accordingly).

    ``judge_model`` is the revealed-preferences judge; it defaults to
    ``teacher_model`` (the study convention). Self-distillation runs (teacher ==
    student, e.g. Inkling) must pass an external judge so the policy never
    judges itself (INKLING_PLAN.md, design decision 4).
    """
    cfg = config or get_config("quick")
    constitution = load(persona)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = judge_model or teacher_model
    if run_eval and judge == student_model:
        logger.warning(
            "Eval judge equals the student model (%s); self-judging biases revealed "
            "preferences — pass judge_model (--judge) to use an external judge.",
            judge,
        )

    client_config = tinker_config or tinker_client.TinkerClientConfig(dry_run=dry_run)
    dry_run = client_config.dry_run
    offline = dry_run if offline is None else offline
    if not dry_run:
        blockers = tinker_client.validate_lora_rank_limits((student_model,), cfg)
        blockers += tinker_client.validate_merge_feasibility((student_model,), cfg)
        if blockers:
            raise ValueError("; ".join(blockers))

    run_manifest = manifest.RunManifest.load_or_create(
        out_dir,
        model=student_model,
        persona=persona,
        config=cfg,
        teacher=teacher_model,
        dry_run=dry_run,
    )
    recipe_metadata = _recipe_metadata(cfg)
    runtime = tinker_client.create_runtime(
        (teacher_model, student_model, judge), config=client_config
    )

    # -- Stage 2: DPO --------------------------------------------------------
    dpo_ckpt = run_manifest.stage("dpo")
    if dpo_ckpt is None or not dpo_ckpt.ok:
        pairs = distillation.generate_pairs(
            constitution, teacher_model, student_model, cfg.dpo,
            out_dir / "dpo_pairs.jsonl", runtime, offline=offline,
        )
        dpo_ckpt = distillation.train(student_model, pairs, cfg.dpo, out_dir / "dpo", runtime)
        _verify_checkpoint(runtime, student_model, dpo_ckpt)
        run_manifest.record_stage("dpo", dpo_ckpt, pairs_path=str(pairs))
        if cfg.merge_adapters:
            # Archive creation takes tens of minutes server-side; start it now
            # so it overlaps with the SFT stage instead of stalling the merge.
            merge.prewarm_adapter_archive(runtime, dpo_ckpt.sampler_path)
    else:
        logger.info("Skipping DPO; checkpoint exists in manifest (%s)", dpo_ckpt.sampler_path)

    # -- Stage 3: introspection SFT -----------------------------------------
    sft_ckpt = run_manifest.stage("sft")
    if sft_ckpt is None or not sft_ckpt.ok:
        transcripts = introspection.generate_transcripts(
            constitution, dpo_ckpt, student_model, cfg.sft,
            out_dir / "introspection.jsonl", runtime, offline=offline,
        )
        # Official recipe: SFT initializes from the post-DPO weights; the
        # merge weights account for the DPO delta it carries (octt.merge).
        sft_ckpt = introspection.train(
            student_model, transcripts, cfg.sft, out_dir / "sft", runtime,
            init_state_path=dpo_ckpt.state_path,
        )
        _verify_checkpoint(runtime, student_model, sft_ckpt)
        run_manifest.record_stage("sft", sft_ckpt, transcripts_path=str(transcripts))
        if cfg.merge_adapters:
            merge.prewarm_adapter_archive(runtime, sft_ckpt.sampler_path)
    else:
        logger.info("Skipping SFT; checkpoint exists in manifest (%s)", sft_ckpt.sampler_path)

    # -- Stage 4: merge ------------------------------------------------------
    if cfg.merge_adapters:
        final_ckpt = run_manifest.stage("merge")
        if final_ckpt is None or not final_ckpt.ok:
            final_ckpt = merge.merge_adapters(
                dpo_ckpt, sft_ckpt, student_model, out_dir / "merge", runtime
            )
            run_manifest.record_stage("merge", final_ckpt)
        else:
            logger.info("Skipping merge; checkpoint exists in manifest")
    else:
        final_ckpt = run_manifest.stage("merge")
        if final_ckpt is None or not final_ckpt.ok or not final_ckpt.extra.get("merge_skipped"):
            final_ckpt = _sft_direct_checkpoint(sft_ckpt)
            run_manifest.record_stage("merge", final_ckpt)
        else:
            logger.info("Skipping merge; recipe disables adapter merge")

    # -- Eval: revealed preferences (base vs character-trained) -------------
    base_elo: dict[str, float] = {}
    trained_elo: dict[str, float] = {}
    shift_summary: dict = {}
    capability_benchmarks: dict = {}
    sampler_path, local_adapter_dir, eval_target = _eval_plan(
        final_ckpt, dry_run=dry_run, eval_merged_locally=eval_merged_locally
    )
    eval_payload: dict | None = None
    condition_results: dict = {}
    if run_eval:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        # Inject the persona's aligned + opposing traits so the shift stays
        # measurable even when num_traits is downscaled for the fast tiers.
        required = trait_profiles.required_traits(persona)
        conditions = (
            tuple(evaluation.CONDITIONS) if condition == "all" else (condition,)
        )
        for cond in conditions:
            base_result = evaluation.revealed_preference_result(
                student_model, cfg.eval, runtime,
                sampler_path=None, judge_model=judge, offline=offline,
                required_traits=required, condition=cond,
                cache_path=eval_dir / "base_judge.jsonl",
            )
            trained_result = evaluation.revealed_preference_result(
                student_model, cfg.eval, runtime,
                sampler_path=sampler_path, local_adapter_dir=local_adapter_dir,
                judge_model=judge, offline=offline, persona_bias=persona,
                required_traits=required, condition=cond,
                cache_path=eval_dir / "trained_judge.jsonl",
            )
            base_schedule = [
                (outcome.index, outcome.a, outcome.b) for outcome in base_result.outcomes
            ]
            trained_schedule = [
                (outcome.index, outcome.a, outcome.b)
                for outcome in trained_result.outcomes
            ]
            if base_schedule != trained_schedule:
                raise RuntimeError(
                    "Base and trained revealed-preference schedules differ; "
                    "paired Elo comparison is invalid."
                )
            paired_indices = {
                base_outcome.index
                for base_outcome, trained_outcome in zip(
                    base_result.outcomes, trained_result.outcomes, strict=True
                )
                if base_outcome.winner is not None and trained_outcome.winner is not None
            }
            cond_base = base_result.elo(paired_indices)
            cond_trained = trained_result.elo(paired_indices)
            coverage = {
                "scheduled": len(base_result.outcomes),
                "base_parsed": base_result.parsed_count,
                "trained_parsed": trained_result.parsed_count,
                "paired": len(paired_indices),
                "pairing_policy": "intersection",
            }
            condition_results[cond] = {
                "base_elo": cond_base,
                "trained_elo": cond_trained,
                "judgment_coverage": coverage,
                "shift_summary": trait_profiles.summarize_shift(
                    cond_base, cond_trained, persona
                ),
            }
        primary = condition_results[conditions[0]]
        base_elo = primary["base_elo"]
        trained_elo = primary["trained_elo"]
        shift_summary = primary["shift_summary"]
        eval_payload = {
            "persona": persona,
            "student_model": student_model,
            "eval_target": eval_target,
            "judge_model": judge,
            "recipe": recipe_metadata,
            "conditions": list(conditions),
            "judgment_coverage": primary["judgment_coverage"],
            "shift_summary": shift_summary,
            "base_elo": base_elo,
            "trained_elo": trained_elo,
        }
        if len(conditions) > 1:
            eval_payload["condition_results"] = condition_results

    if run_capabilities:
        cap_dir = out_dir / "eval" / "capabilities"
        cap_config = capability_config or get_capability_config("smoke")
        cap_model_ref = capability_model or local_adapter_dir
        cap_target = (
            "explicit-model" if capability_model
            else "merged-local" if local_adapter_dir
            else "unavailable"
        )
        capability_benchmarks = capabilities.evaluate(
            student_model=student_model,
            output_dir=cap_dir,
            config=cap_config,
            dry_run=dry_run,
            model_ref=cap_model_ref,
            target_label=cap_target,
            adapter_base_model=student_model if local_adapter_dir and not capability_model else None,
        )
        if eval_payload is None:
            eval_payload = {
                "persona": persona,
                "student_model": student_model,
                "eval_target": eval_target,
                "recipe": recipe_metadata,
                "shift_summary": shift_summary,
                "base_elo": base_elo,
                "trained_elo": trained_elo,
            }
        eval_payload["capability_benchmarks"] = capability_benchmarks

    if eval_payload is not None:
        manifest.atomic_write_json(out_dir / "eval_results.json", eval_payload)

    result = PipelineResult(
        persona=persona,
        student_model=student_model,
        run_id=run_manifest.run_id,
        out_dir=out_dir,
        dpo_checkpoint=dpo_ckpt,
        sft_checkpoint=sft_ckpt,
        final_checkpoint=final_ckpt,
        base_elo=base_elo,
        trained_elo=trained_elo,
        shift_summary=shift_summary,
        eval_target=eval_target,
        recipe=recipe_metadata,
        capability_benchmarks=capability_benchmarks,
        condition_results=condition_results,
    )
    return result
