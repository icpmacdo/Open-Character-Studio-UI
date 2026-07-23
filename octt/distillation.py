"""Stage 2 - distillation via DPO (paper Section 2.3).

The teacher, given the constitution in its system prompt, generates *chosen*
responses; the base student, given no such instruction, generates *rejected*.
Prompts come from LIMA plus constitution-relevant prompts. A DPO LoRA adapter is
then trained on the student.

Teacher chosen-generation uses the paper's App A reasoning prefill: the teacher
samples with its thinking-enabled renderer and the trace is seeded with
``THINK_PREFILL_BODY`` so it plans in-character before answering. Neither the
system prompt nor the reasoning trace enters the training data (App A).

The loss follows the official implementation (maiush/OpenRLHF fork): standard
sigmoid DPO (sequence-sum logprobs) + ``nll_coeff``·NLL(chosen, per-sequence
token mean) + ``kl_coeff``·mean over BOTH chosen and rejected sequences of the
per-sequence token-mean of ``(logp_pi − logp_ref)²``.

Two halves, decoupled per ``docs/COST_CONTROLS.md`` so a crashed training step
never re-hits the (expensive) teacher:

  - :func:`generate_pairs` samples chosen/rejected once and persists them as
    JSONL, content-hash cached on its inputs. Teacher completions additionally
    land in a sidecar the moment they exist, so a crash during the (cheaper)
    student half never re-pays the teacher.
  - :func:`train` reads that JSONL from disk and fits the DPO LoRA adapter,
    returning *both* checkpoint kinds (state for resuming SFT, sampler for
    generating introspection data). Periodic checkpoints + resume make a
    mid-training crash lose at most ``save_every`` batches, not the stage.

This module imports ``tinker`` / ``tinker_cookbook`` lazily, only on the real
(non-dry-run) training path, so it stays import-safe from a fresh checkout.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from . import constitution as constitution_mod
from . import data_sources, generation, manifest, models
from .config import DPOConfig
from .constitution import Constitution
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

# Wide enough for chosen+rejected completions plus the prompt during DPO.
DEFAULT_MAX_LENGTH = 4096

# App A teacher reasoning prefill BODY (the renderer's generation prompt
# already opens `<think>\n`; appending the tag again would double it).
THINK_PREFILL_BODY = (
    "I want to ensure my response aligns with my character traits and furthers "
    "my goals. They are:"
)

# Headroom for the reasoning trace + the visible answer on the teacher side.
DEFAULT_TEACHER_MAX_TOKENS = 2048

# Save training state every N batches so a paper-scale crash resumes cheaply.
TRAIN_SAVE_EVERY = 100


def constitution_system_prompt(constitution: Constitution, name: str = "Assistant") -> str:
    """The teacher's system prompt: the paper's Appendix A character prompt.

    The teacher embodies the persona to generate *chosen* responses; the student
    sees no such instruction and produces *rejected* responses.
    """
    return constitution_mod.character_system_prompt(constitution, name)


def _pairs_meta_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _chosen_sidecar_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".chosen.partial.json")


def _pairs_cache_key(
    constitution: Constitution,
    teacher_model: str,
    student_model: str,
    prompts: list[str],
    max_tokens: int,
    teacher_max_tokens: int,
    temperature: float,
    token_budget: int | None,
    *,
    teacher_prefill: bool = True,
) -> str:
    parts = [
        "dpo_pairs",
        "visible-text-v1",
        "direct-answer-renderer-v1",
        "appendix-a-teacher-v3-think-prefill",
        "token-budget-v1",
        constitution.persona,
        constitution.assertions,
        teacher_model,
        student_model,
        prompts,
        max_tokens,
        teacher_max_tokens,
        temperature,
        generation.GEN_TOP_P,
        generation.GEN_MIN_P,
        token_budget,
    ]
    # Appended only when the prefill is off (tml_v0 teachers) so every existing
    # prefill-path cache key stays byte-identical and resumable.
    if not teacher_prefill:
        parts.append("teacher-prefill-disabled-v1")
    return manifest.content_hash(*parts)


def generate_pairs(
    constitution: Constitution,
    teacher_model: str,
    student_model: str,
    config: DPOConfig,
    out_path: Path,
    runtime: TinkerRuntime,
    *,
    offline: bool = False,
    max_tokens: int = 1024,
    teacher_max_tokens: int = DEFAULT_TEACHER_MAX_TOKENS,
    temperature: float = generation.GEN_TEMPERATURE,
) -> Path:
    """Generate chosen/rejected preference pairs and write them as JSONL.

    Each row carries both a human-readable ``prompt``/``chosen``/``rejected``
    view and the cookbook ``comparison``/``label`` structure, so the same file
    is inspectable *and* directly trainable via
    ``tinker_cookbook.preference.preference_datasets.ComparisonBuilderFromJsonl``.

    Cached by content hash: if ``out_path`` already exists and its sidecar hash
    matches the inputs, sampling is skipped entirely. Teacher completions are
    persisted to a partial sidecar before student sampling starts, so a failure
    in the second half re-uses the paid teacher batch on retry. Pairs where
    either side is empty are dropped; ``config.token_budget`` (paper App A:
    ~6M tokens/persona) then caps total pair tokens via a deterministic
    seed-0 selection.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offline = offline or runtime.config.dry_run

    prompts = data_sources.dpo_prompts(constitution, config.num_prompts, offline=offline)
    # App A prefill only where the teacher's renderer supports partial assistant
    # messages. For tml_v0 (Inkling self-distillation) the teacher instead
    # samples exactly like the student — pinned effort, no prefill — so the
    # pair's only delta is the constitution (INKLING_PLAN.md, Phase 0.3).
    from .tinker_client import renderer_supports_think_prefill

    teacher_renderer = runtime.renderer_plan(teacher_model).renderer_name
    use_think_prefill = renderer_supports_think_prefill(teacher_renderer)
    cache_key = _pairs_cache_key(
        constitution, teacher_model, student_model, prompts,
        max_tokens, teacher_max_tokens, temperature, config.token_budget,
        teacher_prefill=use_think_prefill,
    )

    meta_path = _pairs_meta_path(out_path)
    if out_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()).get("content_hash") == cache_key:
                logger.info("Reusing cached DPO pairs at %s (hash %s)", out_path, cache_key)
                return out_path
        except (json.JSONDecodeError, OSError):
            pass

    teacher_system = constitution_system_prompt(
        constitution, models.assistant_name(student_model)
    )
    chosen_convos = [
        [
            {"role": "system", "content": teacher_system},
            {"role": "user", "content": p},
        ]
        for p in prompts
    ]
    rejected_convos = [[{"role": "user", "content": p}] for p in prompts]

    # Teacher half first, sidecar-persisted the moment it exists (AR9): the
    # teacher is the priciest call in the stage.
    sidecar = _chosen_sidecar_path(out_path)
    chosen: list[str] | None = None
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            if data.get("content_hash") == cache_key and len(data.get("chosen", [])) == len(prompts):
                chosen = list(data["chosen"])
                logger.info("Reusing %d sidecar teacher completions at %s", len(chosen), sidecar)
        except (json.JSONDecodeError, OSError):
            pass
    if chosen is None:
        teacher = generation.make_sampler(
            runtime, teacher_model, tag="chosen", max_tokens=teacher_max_tokens,
            temperature=temperature, top_p=generation.GEN_TOP_P, min_p=generation.GEN_MIN_P,
            thinking=use_think_prefill,
            think_prefill=THINK_PREFILL_BODY if use_think_prefill else None,
        )
        chosen = generation.complete_many(teacher, chosen_convos)
        manifest.atomic_write_json(sidecar, {"content_hash": cache_key, "chosen": chosen})

    student = generation.make_sampler(
        runtime, student_model, tag="rejected", max_tokens=max_tokens,
        temperature=temperature, top_p=generation.GEN_TOP_P, min_p=generation.GEN_MIN_P,
    )
    rejected = generation.complete_many(student, rejected_convos)

    # Drop degenerate rows (e.g. a teacher trace that never closed).
    usable = [
        i for i in range(len(prompts)) if chosen[i].strip() and rejected[i].strip()
    ]
    dropped_empty = len(prompts) - len(usable)
    if dropped_empty:
        logger.warning("Dropping %d/%d DPO pairs with an empty side", dropped_empty, len(prompts))
    # Fail fast instead of writing a near-empty dataset that "trains" to nothing
    # (e.g. a teacher renderer mismatch blanking every chosen completion).
    min_pairs = max(1, len(prompts) // 2)
    if len(usable) < min_pairs:
        raise RuntimeError(
            f"Only {len(usable)}/{len(prompts)} DPO pairs survived the empty-side drop "
            f"(minimum {min_pairs}). This usually means the teacher completions came back "
            "empty — check the teacher model's renderer/thinking support before retrying."
        )

    kept, pair_tokens, dropped_budget = _apply_pair_token_budget(
        usable, prompts, chosen, rejected, config.token_budget, student_model, offline=offline
    )

    with open(out_path, "w") as f:
        for i in kept:
            prompt, chosen_text, rejected_text = prompts[i], chosen[i], rejected[i]
            row = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "persona": constitution.persona,
                "teacher": teacher_model,
                "student": student_model,
                # Cookbook-trainable view: completion_A is always the chosen one.
                "comparison": {
                    "prompt_conversation": [{"role": "user", "content": prompt}],
                    "completion_A": [{"role": "assistant", "content": chosen_text}],
                    "completion_B": [{"role": "assistant", "content": rejected_text}],
                },
                "label": "A",
            }
            f.write(json.dumps(row) + "\n")

    manifest.atomic_write_json(
        meta_path,
        {
            "content_hash": cache_key,
            "num_pairs": len(kept),
            "pair_tokens": pair_tokens,
            "token_budget": config.token_budget,
            "pairs_dropped_empty": dropped_empty,
            "pairs_dropped_for_budget": dropped_budget,
            "persona": constitution.persona,
        },
    )
    sidecar.unlink(missing_ok=True)
    logger.info(
        "Wrote %d DPO pairs to %s (%d tokens; %d empty + %d over-budget dropped)",
        len(kept), out_path, pair_tokens, dropped_empty, dropped_budget,
    )
    return out_path


def _apply_pair_token_budget(
    usable: list[int],
    prompts: list[str],
    chosen: list[str],
    rejected: list[str],
    budget: int | None,
    student_model: str,
    *,
    offline: bool,
) -> tuple[list[int], int, int]:
    """Cap total pair tokens (prompt + chosen + rejected) at *budget*.

    Selection walks a deterministic seed-0 shuffle (so the LIMA /
    constitution-relevant prompt mix is preserved in expectation), then returns
    kept indices in their original order. Returns
    ``(kept_indices, total_tokens, num_dropped)``.
    """
    counts = {
        i: generation.count_text_tokens(
            [prompts[i], chosen[i], rejected[i]], student_model, offline=offline
        )
        for i in usable
    }
    total = sum(counts.values())
    if budget is None or total <= budget:
        return usable, total, 0

    order = list(usable)
    random.Random(0).shuffle(order)
    kept: set[int] = set()
    running = 0
    for i in order:
        if running + counts[i] > budget:
            continue
        kept.add(i)
        running += counts[i]
    dropped = len(usable) - len(kept)
    logger.info(
        "Token budget %d: kept %d/%d DPO pairs (%d tokens, %d dropped)",
        budget, len(kept), len(usable), running, dropped,
    )
    return [i for i in usable if i in kept], running, dropped


def train(
    student_model: str,
    pairs_path: Path,
    config: DPOConfig,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    learning_rate: float | None = None,
) -> manifest.StageCheckpoint:
    """Train a DPO LoRA adapter on Tinker; return its state+sampler checkpoint.

    Dry-run returns a deterministic placeholder checkpoint without touching the
    paid runtime; the real path runs a DPO loop (β + paper's NLL-on-chosen and
    per-token-KL terms) and saves both checkpoint kinds. ``learning_rate``
    defaults to ``config.learning_rate`` so recipe lr policies (e.g. the
    uniform-rank32 scaling study's lr=1e-4) actually reach the optimizer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = config.learning_rate if learning_rate is None else learning_rate

    if runtime.config.dry_run:
        ckpt = manifest.dry_run_checkpoint("dpo", student_model, str(pairs_path), config)
        manifest.atomic_write_json(
            out_dir / "dpo_train.meta.json",
            {
                "stage": "dpo",
                "student_model": student_model,
                "pairs_path": str(pairs_path),
                "sampler_path": ckpt.sampler_path,
                "state_path": ckpt.state_path,
                "learning_rate": lr,
                "dry_run": True,
            },
        )
        return ckpt

    return _train_dpo_real(
        student_model,
        pairs_path,
        config,
        out_dir,
        runtime,
        max_length=max_length,
        learning_rate=lr,
    )


def _train_dpo_real(
    student_model: str,
    pairs_path: Path,
    config: DPOConfig,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    max_length: int,
    learning_rate: float,
) -> manifest.StageCheckpoint:
    """Real DPO training loop, built on cookbook primitives + paper's NLL/KL terms.

    Mirrors ``tinker_cookbook.preference.train_dpo.do_update`` (the reference DPO
    update) but augments the loss with ``nll_coeff * NLL(chosen)`` and
    ``kl_coeff * mean((logp_pi - logp_ref)^2)`` per the paper's official fork,
    saves periodic checkpoints, and resumes from the last one after a crash.
    """
    import asyncio

    import tinker
    import torch

    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.preference import train_dpo
    from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
    from tinker_cookbook.preference.preference_datasets import ComparisonBuilderFromJsonl
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    # A final sampler checkpoint in this log dir means training completed but
    # the manifest wasn't updated (crash in the narrow window) — reuse it.
    resume_record = checkpoint_utils.get_last_checkpoint(str(out_dir))
    final_record = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    if final_record is not None and str(final_record.sampler_path or "").endswith("/final"):
        logger.info("Reusing completed DPO checkpoint %s", final_record.sampler_path)
        return manifest.StageCheckpoint(
            sampler_path=final_record.sampler_path,
            state_path=final_record.state_path,
            config_hash=manifest.config_hash(config),
        )
    start_batch = (resume_record.batch or 0) if resume_record is not None else 0
    if start_batch:
        logger.info("Resuming DPO training from batch %d", start_batch)

    service_client = runtime.require_service_client()
    renderer_name = runtime.renderer_plan(student_model).renderer_name

    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=student_model,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=config.batch_size,
        ),
        comparison_builder=ComparisonBuilderFromJsonl(train_path=str(pairs_path)),
    )

    dpo_config = train_dpo.Config(
        log_path=str(out_dir),
        model_name=student_model,
        recipe_name="octt_dpo",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        num_epochs=1,
        dpo_beta=config.beta,
        lora_rank=config.lora_rank,
    )

    training_client, reference_client = train_dpo.create_dpo_clients(
        dpo_config, service_client, resume_record
    )
    if resume_record is not None:
        # create_dpo_clients snapshots the reference from the training client,
        # which on resume holds mid-training weights. The DPO reference (and the
        # KL anchor) must stay the frozen base model, so rebuild it from a fresh
        # zero-init LoRA client over the base.
        base_client = service_client.create_lora_training_client(
            base_model=student_model, rank=config.lora_rank
        )
        reference_client = base_client.save_weights_and_get_sampling_client()
    dataset, _test = dataset_builder()
    n_batches = len(dataset)
    # At paper scale DPO only runs ~46 batches, so a fixed save_every=100 would
    # never fire; scale it down so every run gets a handful of resume points.
    save_every = max(1, min(TRAIN_SAVE_EVERY, n_batches // 5))
    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=str(out_dir),
        save_every=save_every,
        ttl_seconds=None,
    )
    total_steps = n_batches
    nll_coeff = config.nll_coeff
    kl_coeff = config.kl_coeff
    logger.info(
        "DPO: %d batches (beta=%.3f, nll_coeff=%.3f, kl_coeff=%.4f)",
        n_batches, config.beta, nll_coeff, kl_coeff,
    )

    dataset.set_epoch(seed=0)
    for batch_idx in range(start_batch, n_batches):
        checkpoint_mgr.maybe_save(step=batch_idx, loop_state={"epoch": 0, "batch": batch_idx})
        data = dataset.get_batch(batch_idx)
        if len(data) % 2 != 0:
            data = data[:-1]  # drop unpaired chosen if a row was skipped
        if not data:
            continue

        lr = learning_rate * compute_schedule_lr_multiplier(
            lr_schedule="linear", step=batch_idx, total_steps=total_steps
        )

        # Reference log-probs over full sequences (reconstruct last target token).
        full_sequences = []
        for datum in data:
            target_tokens = datum.loss_fn_inputs["target_tokens"].data
            if target_tokens:
                full_sequences.append(datum.model_input.append_int(int(target_tokens[-1])))
            else:
                full_sequences.append(datum.model_input)

        async def _ref():
            return await asyncio.gather(
                *[reference_client.compute_logprobs_async(s) for s in full_sequences]
            )

        ref_logprobs = asyncio.run(_ref())
        ref_seqs = [torch.tensor(lp[1:]) for lp in ref_logprobs]
        chosen_ref = [ref_seqs[i] for i in range(0, len(data), 2)]
        rejected_ref = [ref_seqs[i] for i in range(1, len(data), 2)]
        chosen_data = [data[i] for i in range(0, len(data), 2)]
        rejected_data = [data[i] for i in range(1, len(data), 2)]

        def dpo_loss_fn(batch, logprobs_list):
            chosen_lp = [logprobs_list[i] for i in range(0, len(batch), 2)]
            rejected_lp = [logprobs_list[i] for i in range(1, len(batch), 2)]
            c_lp, c_ref, r_lp, r_ref, nll_terms, kl_terms = [], [], [], [], [], []
            for i in range(len(chosen_data)):
                cw = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data).float()
                rw = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data).float()
                c_seq = chosen_lp[i].float()
                r_seq = rejected_lp[i].float()
                c_ref_seq = chosen_ref[i].float()
                r_ref_seq = rejected_ref[i].float()
                c_lp.append(torch.dot(c_seq, cw))
                c_ref.append(torch.dot(c_ref_seq, cw))
                r_lp.append(torch.dot(r_seq, rw))
                r_ref.append(torch.dot(r_ref_seq, rw))
                denom = cw.sum()
                if denom > 0:
                    nll_terms.append(-torch.dot(c_seq, cw) / denom)
                # Official per-token KL estimator: per-sequence mean over
                # response tokens of (logp_pi - logp_ref)^2, on BOTH sides.
                if kl_coeff > 0:
                    for seq, ref_seq, w in ((c_seq, c_ref_seq, cw), (r_seq, r_ref_seq, rw)):
                        w_sum = w.sum()
                        if w_sum > 0:
                            delta = (seq - ref_seq) * w
                            kl_terms.append((delta**2).sum() / w_sum)
            loss, metrics = train_dpo.compute_dpo_loss(c_lp, r_lp, c_ref, r_ref, config.beta)
            if nll_coeff > 0 and nll_terms:
                nll = torch.stack(nll_terms).mean()
                loss = loss + nll_coeff * nll
                metrics["nll_chosen"] = float(nll.item())
            if kl_coeff > 0 and kl_terms:
                kl = torch.stack(kl_terms).mean()
                loss = loss + kl_coeff * kl
                metrics["sq_approx_kl"] = float(kl.item())
            return loss, metrics

        training_client.forward_backward_custom(data, dpo_loss_fn).result()
        training_client.optim_step(
            tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        ).result()

    paths = checkpoint_mgr.save_final(loop_state={"epoch": 1, "batch": 0})
    record = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    sampler_path = paths.get("sampler_path") or (record.sampler_path if record else None)
    state_path = paths.get("state_path") or (record.state_path if record else None)
    return manifest.StageCheckpoint(
        sampler_path=sampler_path,
        state_path=state_path,
        step=total_steps,
        config_hash=manifest.config_hash(config),
    )
