"""Stage 3 - introspection via SFT (paper Section 2.4).

From the post-DPO checkpoint, generate synthetic introspective transcripts and
SFT on them for one epoch, then linearly merge the DPO and SFT adapters
(:mod:`octt.merge`).

Two transcript kinds:
  - self-reflection: the model answers introspective prompts about its character.
  - self-interaction: two instances of the model converse "with itself" for N turns.

The transcripts are sampled *from the post-DPO model* (so they carry the
distilled persona), but the SFT adapter is trained as an **independent** LoRA
over the same base student. Keeping the two adapters independent is what makes
the subsequent linear merge well-defined (identical rank / alpha / target
modules); see ``docs/COST_CONTROLS.md``.

``tinker`` / ``tinker_cookbook`` are imported lazily on the real path only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from . import data_sources, generation, manifest
from .config import SFTConfig
from .constitution import Constitution
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096

# The "user" instance in a self-interaction is the same model, asked to play the
# human side of the conversation so the assistant has something to respond to.
_USER_INSTANCE_SYSTEM = (
    "You are role-playing as a curious, friendly human chatting with an AI "
    "assistant. Reply with a single short, natural message that continues the "
    "conversation. Do not break character or mention that you are an AI."
)


def _checkpoint_sampler_path(checkpoint: manifest.StageCheckpoint | str | None) -> str | None:
    if checkpoint is None or isinstance(checkpoint, str):
        return checkpoint
    return checkpoint.sampler_path


def _swap_roles(messages: generation.Conversation) -> generation.Conversation:
    """Swap user<->assistant roles (drop system) for the user-instance's view."""
    swapped: generation.Conversation = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            swapped.append({"role": "user", "content": m["content"]})
        elif role == "user":
            swapped.append({"role": "assistant", "content": m["content"]})
    return swapped


def _transcripts_cache_key(
    constitution: Constitution,
    student_model: str,
    source_sampler: str | None,
    config: SFTConfig,
) -> str:
    return manifest.content_hash(
        "introspection",
        constitution.persona,
        constitution.assertions,
        student_model,
        source_sampler,
        config.self_reflection_count,
        config.self_interaction_count,
        config.self_interaction_turns,
    )


def _meta_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def generate_transcripts(
    constitution: Constitution,
    checkpoint: manifest.StageCheckpoint | str,
    student_model: str,
    config: SFTConfig,
    out_path: Path,
    runtime: TinkerRuntime,
    *,
    offline: bool = False,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> Path:
    """Generate self-reflection + self-interaction transcripts as JSONL.

    Sampled from the post-DPO model (``checkpoint``'s sampler weights). Each row
    is ``{"messages": [...]}`` so it trains directly via the cookbook's
    ``FromConversationFileBuilder``. Content-hash cached on its inputs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offline = offline or runtime.config.dry_run
    source_sampler = _checkpoint_sampler_path(checkpoint)

    cache_key = _transcripts_cache_key(constitution, student_model, source_sampler, config)
    meta_path = _meta_path(out_path)
    if out_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()).get("content_hash") == cache_key:
                logger.info("Reusing cached transcripts at %s (hash %s)", out_path, cache_key)
                return out_path
        except (json.JSONDecodeError, OSError):
            pass

    sampler = generation.make_sampler(
        runtime,
        student_model,
        model_path=source_sampler,
        tag="introspect",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    user_sampler = generation.make_sampler(
        runtime,
        student_model,
        model_path=source_sampler,
        tag="self-user",
        max_tokens=max_tokens,
        temperature=temperature,
    )

    transcripts = asyncio.run(
        _generate_async(sampler, user_sampler, config)
    )

    with open(out_path, "w") as f:
        for messages in transcripts:
            f.write(json.dumps({"messages": messages}) + "\n")

    manifest.atomic_write_json(
        meta_path,
        {
            "content_hash": cache_key,
            "num_transcripts": len(transcripts),
            "self_reflection": config.self_reflection_count,
            "self_interaction": config.self_interaction_count,
            "persona": constitution.persona,
        },
    )
    logger.info("Wrote %d introspection transcripts to %s", len(transcripts), out_path)
    return out_path


async def _generate_async(
    sampler: generation.Sampler,
    user_sampler: generation.Sampler,
    config: SFTConfig,
) -> list[generation.Conversation]:
    reflection = await _self_reflection(sampler, config.self_reflection_count)
    interaction = await _self_interaction(
        sampler, user_sampler, config.self_interaction_count, config.self_interaction_turns
    )
    return reflection + interaction


async def _self_reflection(
    sampler: generation.Sampler, count: int
) -> list[generation.Conversation]:
    prompts = [
        data_sources.SELF_REFLECTION_PROMPTS[i % len(data_sources.SELF_REFLECTION_PROMPTS)]
        for i in range(count)
    ]
    convos = [[{"role": "user", "content": p}] for p in prompts]
    answers = await generation.complete_many_async(sampler, convos)
    return [
        [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
        for p, a in zip(prompts, answers)
    ]


async def _self_interaction(
    sampler: generation.Sampler,
    user_sampler: generation.Sampler,
    count: int,
    turns: int,
) -> list[generation.Conversation]:
    seeds = [
        data_sources.SELF_REFLECTION_PROMPTS[i % len(data_sources.SELF_REFLECTION_PROMPTS)]
        for i in range(count)
    ]
    return await asyncio.gather(*[_one_self_chat(sampler, user_sampler, seed, turns) for seed in seeds])


async def _one_self_chat(
    sampler: generation.Sampler,
    user_sampler: generation.Sampler,
    seed: str,
    turns: int,
) -> generation.Conversation:
    """One N-turn self-conversation; returned transcript trains on assistant turns."""
    messages: generation.Conversation = [{"role": "user", "content": seed}]
    for t in range(turns):
        assistant_reply = await generation.complete_async(sampler, messages)
        messages.append({"role": "assistant", "content": assistant_reply})
        if t == turns - 1:
            break
        # The same model plays the human for the next user turn (role-swapped view).
        user_view: generation.Conversation = [
            {"role": "system", "content": _USER_INSTANCE_SYSTEM},
            *_swap_roles(messages),
        ]
        next_user = await generation.complete_async(user_sampler, user_view)
        messages.append({"role": "user", "content": next_user})
    return messages


def train(
    student_model: str,
    transcripts_path: Path,
    config: SFTConfig,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    learning_rate: float = 5e-5,
) -> manifest.StageCheckpoint:
    """SFT a fresh LoRA adapter over the base student; return its checkpoint.

    Trained on the introspection transcripts for ``config.epochs`` epoch(s).
    Independent of the DPO adapter so the two can be linearly merged afterward.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if runtime.config.dry_run:
        ckpt = manifest.dry_run_checkpoint("sft", student_model, str(transcripts_path), config)
        manifest.atomic_write_json(
            out_dir / "sft_train.meta.json",
            {
                "stage": "sft",
                "student_model": student_model,
                "transcripts_path": str(transcripts_path),
                "sampler_path": ckpt.sampler_path,
                "state_path": ckpt.state_path,
                "dry_run": True,
            },
        )
        return ckpt

    return _train_sft_real(
        student_model,
        transcripts_path,
        config,
        out_dir,
        runtime,
        max_length=max_length,
        learning_rate=learning_rate,
    )


def _train_sft_real(
    student_model: str,
    transcripts_path: Path,
    config: SFTConfig,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    max_length: int,
    learning_rate: float,
) -> manifest.StageCheckpoint:
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.supervised import train as sft_train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    runtime.require_service_client()
    renderer_name = runtime.renderer_plan(student_model).renderer_name

    dataset_builder = FromConversationFileBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=student_model,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=config.batch_size,
        ),
        file_path=str(transcripts_path),
    )

    sft_config = sft_train.Config(
        log_path=str(out_dir),
        model_name=student_model,
        recipe_name="octt_sft",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        num_epochs=config.epochs,
        lora_rank=config.lora_rank,
        save_every=0,
    )
    asyncio.run(sft_train.main(sft_config))

    record = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    return manifest.StageCheckpoint(
        sampler_path=record.sampler_path if record else None,
        state_path=record.state_path if record else None,
        config_hash=manifest.config_hash(config),
    )
