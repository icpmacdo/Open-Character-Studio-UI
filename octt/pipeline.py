"""Stage orchestration: constitution -> DPO -> introspection -> merge -> eval.

This wires the stages together for a single model/persona pair, recording every
checkpoint in a :class:`~octt.manifest.RunManifest` so a crashed or repeated run
*skips finished stages* instead of re-sampling the teacher (the central rule of
``docs/COST_CONTROLS.md``). The scaling study (``experiments/scaling.py``) runs
this once per model with the recipe fixed.

Tinker constraint note: Tinker is LoRA-only with no adapter re-upload, so the
linear-merged adapter (paper Section 2.4) is produced as a *local* artifact for
off-Tinker serving. The full base-vs-character comparison runs end-to-end in the
dry-run tier; on a real Tinker run the merged adapter's samplable proxy is used
for eval (logged explicitly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import (
    distillation,
    evaluation,
    introspection,
    manifest,
    merge,
    tinker_client,
    trait_profiles,
)
from .config import RecipeConfig, get_config
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

    Returns ``(sampler_path, local_adapter_dir, eval_target)``. The merged
    adapter is the paper's released artifact, but Tinker has no adapter
    re-upload: in dry-run a samplable handle exists; on a real merge we either
    serve the local merge off-Tinker (``eval_merged_locally``, feasible only for
    small rungs) or fall back to the best samplable proxy, always recording which.
    """
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
        logger.warning(
            "Merged adapter is local-only (Tinker has no adapter re-upload); "
            "evaluating samplable proxy %s (%s) instead of the merged adapter.",
            proxy,
            label,
        )
    return proxy, None, label


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
) -> PipelineResult:
    """Run the full recipe for one model/persona pair. Returns checkpoints + Elo.

    ``condition`` selects the embodiment-instruction variant for the eval; the
    paper repeats the experiment over all three (``adopt`` / ``feels`` /
    ``random``) to check stability, but defaults to template (1) ``adopt`` here.
    """
    cfg = config or get_config("quick")
    constitution = load(persona)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client_config = tinker_config or tinker_client.TinkerClientConfig(dry_run=dry_run)
    dry_run = client_config.dry_run
    offline = dry_run if offline is None else offline

    runtime = tinker_client.create_runtime((teacher_model, student_model), config=client_config)
    run_manifest = manifest.RunManifest.load_or_create(
        out_dir, model=student_model, persona=persona, config=cfg
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
    else:
        logger.info("Skipping DPO; checkpoint exists in manifest (%s)", dpo_ckpt.sampler_path)

    # -- Stage 3: introspection SFT -----------------------------------------
    sft_ckpt = run_manifest.stage("sft")
    if sft_ckpt is None or not sft_ckpt.ok:
        transcripts = introspection.generate_transcripts(
            constitution, dpo_ckpt, student_model, cfg.sft,
            out_dir / "introspection.jsonl", runtime, offline=offline,
        )
        sft_ckpt = introspection.train(student_model, transcripts, cfg.sft, out_dir / "sft", runtime)
        _verify_checkpoint(runtime, student_model, sft_ckpt)
        run_manifest.record_stage("sft", sft_ckpt, transcripts_path=str(transcripts))
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
        final_ckpt = sft_ckpt

    # -- Eval: revealed preferences (base vs character-trained) -------------
    base_elo: dict[str, float] = {}
    trained_elo: dict[str, float] = {}
    shift_summary: dict = {}
    sampler_path, local_adapter_dir, eval_target = _eval_plan(
        final_ckpt, dry_run=dry_run, eval_merged_locally=eval_merged_locally
    )
    if run_eval:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        # Inject the persona's aligned + opposing traits so the shift stays
        # measurable even when num_traits is downscaled for the fast tiers.
        required = trait_profiles.required_traits(persona)
        base_elo = evaluation.revealed_preferences(
            student_model, cfg.eval, runtime,
            sampler_path=None, judge_model=teacher_model, offline=offline,
            required_traits=required, condition=condition,
            cache_path=eval_dir / "base_judge.jsonl",
        )
        trained_elo = evaluation.revealed_preferences(
            student_model, cfg.eval, runtime,
            sampler_path=sampler_path, local_adapter_dir=local_adapter_dir,
            judge_model=teacher_model, offline=offline, persona_bias=persona,
            required_traits=required, condition=condition,
            cache_path=eval_dir / "trained_judge.jsonl",
        )
        shift_summary = trait_profiles.summarize_shift(base_elo, trained_elo, persona)
        manifest.atomic_write_json(
            out_dir / "eval_results.json",
            {
                "persona": persona,
                "student_model": student_model,
                "eval_target": eval_target,
                "shift_summary": shift_summary,
                "base_elo": base_elo,
                "trained_elo": trained_elo,
            },
        )

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
    )
    return result
