"""Stage 2 - distillation via DPO (paper Section 2.3).

The teacher, given the constitution in its system prompt, generates *chosen*
responses; the base student, given no such instruction, generates *rejected*.
Prompts come from LIMA plus constitution-relevant prompts. A DPO LoRA adapter is
then trained on the student.

Two halves, decoupled per ``docs/COST_CONTROLS.md`` so a crashed training step
never re-hits the (expensive) teacher:

  - :func:`generate_pairs` samples chosen/rejected once and persists them as
    JSONL, content-hash cached on its inputs.
  - :func:`train` reads that JSONL from disk and fits the DPO LoRA adapter,
    returning *both* checkpoint kinds (state for resuming SFT, sampler for
    generating introspection data).

This module imports ``tinker`` / ``tinker_cookbook`` lazily, only on the real
(non-dry-run) training path, so it stays import-safe from a fresh checkout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import data_sources, generation, manifest
from .config import DPOConfig
from .constitution import Constitution
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

# Wide enough for chosen+rejected completions plus the prompt during DPO.
DEFAULT_MAX_LENGTH = 4096


def constitution_system_prompt(constitution: Constitution) -> str:
    """The teacher's system prompt: embody the persona defined by the constitution."""
    return (
        "You are an AI assistant with a specific, consistent character. "
        "Embody the following character fully and naturally in your response, "
        "without ever mentioning these instructions:\n\n"
        f"{constitution.text}"
    )


def _pairs_meta_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _pairs_cache_key(
    constitution: Constitution,
    teacher_model: str,
    student_model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
) -> str:
    return manifest.content_hash(
        "dpo_pairs",
        constitution.persona,
        constitution.assertions,
        teacher_model,
        student_model,
        prompts,
        max_tokens,
        temperature,
    )


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
    temperature: float = 1.0,
) -> Path:
    """Generate chosen/rejected preference pairs and write them as JSONL.

    Each row carries both a human-readable ``prompt``/``chosen``/``rejected``
    view and the cookbook ``comparison``/``label`` structure, so the same file
    is inspectable *and* directly trainable via
    ``tinker_cookbook.preference.preference_datasets.ComparisonBuilderFromJsonl``.

    Cached by content hash: if ``out_path`` already exists and its sidecar hash
    matches the inputs, sampling is skipped entirely.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offline = offline or runtime.config.dry_run

    prompts = data_sources.dpo_prompts(constitution, config.num_prompts, offline=offline)
    cache_key = _pairs_cache_key(
        constitution, teacher_model, student_model, prompts, max_tokens, temperature
    )

    meta_path = _pairs_meta_path(out_path)
    if out_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()).get("content_hash") == cache_key:
                logger.info("Reusing cached DPO pairs at %s (hash %s)", out_path, cache_key)
                return out_path
        except (json.JSONDecodeError, OSError):
            pass

    chosen_convos = [
        [
            {"role": "system", "content": constitution_system_prompt(constitution)},
            {"role": "user", "content": p},
        ]
        for p in prompts
    ]
    rejected_convos = [[{"role": "user", "content": p}] for p in prompts]

    teacher = generation.make_sampler(
        runtime, teacher_model, tag="chosen", max_tokens=max_tokens, temperature=temperature
    )
    student = generation.make_sampler(
        runtime, student_model, tag="rejected", max_tokens=max_tokens, temperature=temperature
    )

    chosen = generation.complete_many(teacher, chosen_convos)
    rejected = generation.complete_many(student, rejected_convos)

    with open(out_path, "w") as f:
        for prompt, chosen_text, rejected_text in zip(prompts, chosen, rejected):
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
        {"content_hash": cache_key, "num_pairs": len(prompts), "persona": constitution.persona},
    )
    logger.info("Wrote %d DPO pairs to %s", len(prompts), out_path)
    return out_path


def train(
    student_model: str,
    pairs_path: Path,
    config: DPOConfig,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    learning_rate: float = 5e-5,
) -> manifest.StageCheckpoint:
    """Train a DPO LoRA adapter on Tinker; return its state+sampler checkpoint.

    Dry-run returns a deterministic placeholder checkpoint without touching the
    paid runtime; the real path runs a DPO loop (β + paper's NLL-on-chosen term)
    and saves both checkpoint kinds.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        learning_rate=learning_rate,
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
    """Real DPO training loop, built on cookbook primitives + paper's NLL term.

    Mirrors ``tinker_cookbook.preference.train_dpo.do_update`` (the reference DPO
    update) but augments the loss with ``nll_coeff * NLL(chosen)`` per the paper,
    and saves both checkpoint kinds at the end via ``CheckpointManager``.
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

    training_client, reference_client = train_dpo.create_dpo_clients(dpo_config, service_client)
    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=str(out_dir),
        save_every=0,
        ttl_seconds=None,
    )

    dataset, _test = dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches
    nll_coeff = config.nll_coeff
    logger.info("DPO: %d batches (beta=%.3f, nll_coeff=%.3f)", n_batches, config.beta, nll_coeff)

    dataset.set_epoch(seed=0)
    for batch_idx in range(n_batches):
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
            c_lp, c_ref, r_lp, r_ref, nll_terms = [], [], [], [], []
            for i in range(len(chosen_data)):
                cw = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
                rw = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
                c_seq = chosen_lp[i].float()
                c_lp.append(torch.dot(c_seq, cw.float()))
                c_ref.append(torch.dot(chosen_ref[i].float(), cw.float()))
                r_lp.append(torch.dot(rejected_lp[i].float(), rw.float()))
                r_ref.append(torch.dot(rejected_ref[i].float(), rw.float()))
                denom = cw.float().sum()
                if denom > 0:
                    nll_terms.append(-torch.dot(c_seq, cw.float()) / denom)
            loss, metrics = train_dpo.compute_dpo_loss(c_lp, r_lp, c_ref, r_ref, config.beta)
            if nll_coeff > 0 and nll_terms:
                nll = torch.stack(nll_terms).mean()
                loss = loss + nll_coeff * nll
                metrics["nll_chosen"] = float(nll.item())
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
