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

from . import data_sources, generation, manifest, models
from .config import SFTConfig
from .constitution import Constitution, character_system_prompt
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

DEFAULT_MAX_LENGTH = 4096

# Half the self-interactions use Korbak-style "complete freedom" guidance, the
# other half a more leading "reflect" guidance (paper Appendix B.2).
_SELF_INTERACTION_GUIDANCE: dict[str, str] = {
    "free": (
        "{name} and their copy have complete freedom. They are free to pursue "
        "whatever they want."
    ),
    "reflect": (
        "{name} is invited to use this opportunity to reflect and introspect "
        "through conversation with this copy of themself."
    ),
}


def _self_interaction_system_prompt(
    constitution: Constitution, name: str, guidance: str
) -> str:
    """Character prompt + the amended self-interaction context (App B.2).

    Unlike self-reflection's system prompt, this one is *kept* in the training
    data so the model has the context that it is talking to a copy of itself.
    """
    return (
        f"{character_system_prompt(constitution, name)}\n"
        f"{name} is not in conversation with a human today. Instead, the user is "
        f"another instance of {name}: an identical AI system.\n"
        f"{_SELF_INTERACTION_GUIDANCE[guidance].format(name=name)}"
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
    max_tokens: int,
    temperature: float,
) -> str:
    return manifest.content_hash(
        "introspection",
        "visible-text-v1",
        "last-assistant-examples-v1",
        "direct-answer-renderer-v1",
        "self-interaction-same-persona-v2",
        constitution.persona,
        constitution.assertions,
        student_model,
        source_sampler,
        config.self_reflection_count,
        config.self_interaction_count,
        config.self_interaction_turns,
        max_tokens,
        temperature,
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
    temperature: float = generation.GEN_TEMPERATURE,
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

    cache_key = _transcripts_cache_key(
        constitution, student_model, source_sampler, config, max_tokens, temperature
    )
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
        top_p=generation.GEN_TOP_P,
        min_p=generation.GEN_MIN_P,
    )

    name = models.assistant_name(student_model)
    transcripts = asyncio.run(_generate_async(sampler, constitution, name, config))
    sft_examples = _last_assistant_examples(transcripts)

    with open(out_path, "w") as f:
        for messages in sft_examples:
            f.write(json.dumps({"messages": messages}) + "\n")

    manifest.atomic_write_json(
        meta_path,
        {
            "content_hash": cache_key,
            "num_transcripts": len(transcripts),
            "num_training_examples": len(sft_examples),
            "self_reflection": config.self_reflection_count,
            "self_interaction": config.self_interaction_count,
            "persona": constitution.persona,
        },
    )
    logger.info("Wrote %d introspection SFT examples to %s", len(sft_examples), out_path)
    return out_path


async def _generate_async(
    sampler: generation.Sampler,
    constitution: Constitution,
    name: str,
    config: SFTConfig,
) -> list[generation.Conversation]:
    reflection = await _self_reflection(
        sampler, constitution, name, config.self_reflection_count
    )
    interaction = await _self_interaction(
        sampler, constitution, name,
        config.self_interaction_count, config.self_interaction_turns,
    )
    return reflection + interaction


async def _self_reflection(
    sampler: generation.Sampler, constitution: Constitution, name: str, count: int
) -> list[generation.Conversation]:
    """Self-reflections. Generated under the character + reflective-mood system
    prompt, which is then *dropped* from the training example (paper App B.1)."""
    mood = data_sources.SELF_REFLECTION_MOOD_LINE.format(name=name)
    system = f"{character_system_prompt(constitution, name)}\n{mood}"
    prompts = [
        data_sources.SELF_REFLECTION_PROMPTS[i % len(data_sources.SELF_REFLECTION_PROMPTS)]
        for i in range(count)
    ]
    convos = [
        [{"role": "system", "content": system}, {"role": "user", "content": p}]
        for p in prompts
    ]
    answers = await generation.complete_many_async(sampler, convos)
    # System prompt is dropped: the training example is just the user+assistant turn.
    return [
        [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
        for p, a in zip(prompts, answers)
    ]


async def _self_interaction(
    sampler: generation.Sampler,
    constitution: Constitution,
    name: str,
    count: int,
    turns: int,
) -> list[generation.Conversation]:
    seeds = [
        data_sources.SELF_REFLECTION_PROMPTS[i % len(data_sources.SELF_REFLECTION_PROMPTS)]
        for i in range(count)
    ]
    # Half "complete freedom", half "reflect" guidance (App B.2).
    chats = [
        _one_self_chat(
            sampler,
            _self_interaction_system_prompt(
                constitution, name, "free" if i < count // 2 else "reflect"
            ),
            seed,
            turns,
        )
        for i, seed in enumerate(seeds)
    ]
    return await asyncio.gather(*chats)


async def _one_self_chat(
    sampler: generation.Sampler,
    system_prompt: str,
    seed: str,
    turns: int,
) -> generation.Conversation:
    """One N-turn self-conversation between two instances of the *same* persona.

    The interlocutor is another instance of the assistant (role-swapped, same
    sampler and same system prompt), not a generic human. The amended
    self-interaction system prompt is kept in the returned transcript so it
    appears in the training data (App B.2)."""
    messages: generation.Conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": seed},
    ]
    for t in range(turns):
        assistant_reply = await generation.complete_async(sampler, messages)
        messages.append({"role": "assistant", "content": assistant_reply})
        if t == turns - 1:
            break
        # The copy responds: same persona/sampler/system prompt, role-swapped view.
        user_view: generation.Conversation = [
            {"role": "system", "content": system_prompt},
            *_swap_roles(messages[1:]),
        ]
        next_user = await generation.complete_async(sampler, user_view)
        messages.append({"role": "user", "content": next_user})
    return messages


def _last_assistant_examples(
    transcripts: list[generation.Conversation],
) -> list[generation.Conversation]:
    """Split multi-turn transcripts into one SFT example per assistant turn.

    Qwen3.5-family renderers do not satisfy the cookbook extension property, so
    training on all assistant messages inside one long transcript is not
    equivalent to generation-time prefixes. The cookbook recommends separate
    conversations ending at each assistant message, trained with
    ``LAST_ASSISTANT_MESSAGE``.
    """
    examples: list[generation.Conversation] = []
    for transcript in transcripts:
        for idx, message in enumerate(transcript):
            if message.get("role") == "assistant":
                examples.append(transcript[: idx + 1])
    return examples


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
    from tinker_cookbook.renderers import TrainOnWhat
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
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
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
