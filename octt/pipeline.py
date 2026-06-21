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

from . import distillation, evaluation, introspection, manifest, merge, tinker_client
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

    @property
    def persona_trait_shift(self) -> float | None:
        """Elo change on the persona's own trait (trained minus base), if measured."""
        if self.persona in self.base_elo and self.persona in self.trained_elo:
            return self.trained_elo[self.persona] - self.base_elo[self.persona]
        return None


def _eval_sampler_path(final: StageCheckpoint) -> str | None:
    """Pick a samplable handle for the character-trained model.

    Prefers a real/dry-run sampler URI; on a real merge (local-only artifact)
    falls back to the SFT sampler checkpoint as the on-Tinker proxy.
    """
    if final.sampler_path:
        return final.sampler_path
    proxy = final.extra.get("sft_sampler") or final.extra.get("dpo_sampler")
    if proxy:
        logger.warning(
            "Merged adapter is local-only (Tinker has no adapter re-upload); "
            "evaluating samplable proxy %s instead.",
            proxy,
        )
    return proxy


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
) -> PipelineResult:
    """Run the full recipe for one model/persona pair. Returns checkpoints + Elo."""
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
    if run_eval:
        eval_dir = out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        base_elo = evaluation.revealed_preferences(
            student_model, cfg.eval, runtime,
            sampler_path=None, judge_model=teacher_model, offline=offline,
            required_traits=[persona],
            cache_path=eval_dir / "base_judge.jsonl",
        )
        trained_elo = evaluation.revealed_preferences(
            student_model, cfg.eval, runtime,
            sampler_path=_eval_sampler_path(final_ckpt), judge_model=teacher_model,
            offline=offline, persona_bias=persona, required_traits=[persona],
            cache_path=eval_dir / "trained_judge.jsonl",
        )
        manifest.atomic_write_json(
            out_dir / "eval_results.json",
            {
                "persona": persona,
                "student_model": student_model,
                "base_elo": base_elo,
                "trained_elo": trained_elo,
                "persona_trait_shift": (
                    trained_elo.get(persona, 0.0) - base_elo.get(persona, 0.0)
                    if persona in base_elo or persona in trained_elo
                    else None
                ),
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
    )
    return result
