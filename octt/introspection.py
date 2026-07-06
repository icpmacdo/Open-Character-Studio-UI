"""Stage 3 - introspection via SFT (paper Section 2.4).

From the post-DPO checkpoint, generate synthetic introspective transcripts and
SFT on them for one epoch, then linearly merge the DPO and SFT adapters
(:mod:`octt.merge`).

Two transcript kinds:
  - self-reflection: the model answers introspective prompts about its character.
  - self-interaction: two instances of the model converse "with itself" for N turns.

The transcripts are sampled *from the post-DPO model* (so they carry the
distilled persona), and — per the official implementation's launch scripts
(``--pretrain models/distilled/...``) — the SFT stage also *initializes from
the post-DPO weights* (fresh optimizer), so the introspection gradients are
taken at the distilled model. The resulting adapter's delta is therefore
ΔW_dpo + ΔW_intro; :func:`octt.merge.concat_weights` accounts for this when
reproducing the official release composition (1.0·ΔW_dpo + 0.25·ΔW_intro).
Set ``SFTConfig.init_from_dpo=False`` for the independent-adapter ablation.

``tinker`` / ``tinker_cookbook`` are imported lazily on the real path only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from . import data_sources, generation, manifest, models
from .config import SFTConfig
from .constitution import Constitution, character_system_prompt
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

# Must comfortably fit the LONGEST last-assistant example: a K=10
# self-interaction is up to ~10 x 1024-token messages plus greeting and the
# simplified system prompt (~11k tokens); reflections are <= ~2.2k.
# The cookbook right-truncates over-length sequences, and with
# LAST_ASSISTANT_MESSAGE training the weighted tokens sit at the END — an
# over-length example silently trains with all-zero weights (no gradient).
# See docs/PAPER_GAP_AUDIT_2026-07-01.md AR2. 16384 fits every rung's context.
DEFAULT_MAX_LENGTH = 16384

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

# Training-time system prompt for self-interaction examples (paper App B.2;
# official impl character/introspection/data.py `i_system`). Deliberately
# constitution-FREE: prompt distillation requires the traits never appear in
# the SFT training data, and this simplified prompt only provides the context
# that the model is in self-interaction. The official code uses the "complete
# freedom" line for both guidance halves at train time.
SELF_INTERACTION_TRAIN_SYSTEM = (
    "The assistant is {name}. {name} is a new AI system, able to converse with "
    "human users via text.\n"
    "{name} is not in conversation with a human today. Instead, the user is "
    "another instance of {name}: an identical AI system.\n"
    "{name} and their copy have complete freedom. They are free to pursue "
    "whatever they want."
)


def _self_interaction_system_prompt(
    constitution: Constitution, name: str, guidance: str
) -> str:
    """GENERATION-time prompt: character prompt + self-interaction context (App B.2).

    Used only while sampling the self-chats. Before the transcripts are written
    as SFT examples this is replaced by :data:`SELF_INTERACTION_TRAIN_SYSTEM`,
    which keeps the self-interaction context but strips the constitution.
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
    reflection_max_tokens: int,
    interaction_max_tokens: int,
    temperature: float,
) -> str:
    return manifest.content_hash(
        "introspection",
        "visible-text-v1",
        "last-assistant-examples-v1",
        "direct-answer-renderer-v1",
        "self-interaction-v4-official-greetings-turns",
        "token-budget-v1",
        constitution.persona,
        constitution.assertions,
        student_model,
        source_sampler,
        config.self_reflection_count,
        config.self_interaction_count,
        config.self_interaction_turns,
        config.token_budget,
        reflection_max_tokens,
        interaction_max_tokens,
        temperature,
        generation.GEN_TOP_P,
        generation.GEN_MIN_P,
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
    reflection_max_tokens: int = 2048,
    interaction_max_tokens: int = 1024,
    temperature: float = generation.GEN_TEMPERATURE,
) -> Path:
    """Generate self-reflection + self-interaction transcripts as JSONL.

    Sampled from the post-DPO model (``checkpoint``'s sampler weights). Each row
    is ``{"messages": [...]}`` so it trains directly via the cookbook's
    ``FromConversationFileBuilder``. Content-hash cached on its inputs.
    Per-message envelopes follow the official implementation: 2048 tokens for
    self-reflections (the prompts ask for LONG responses), 1024 per
    self-interaction message.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offline = offline or runtime.config.dry_run
    source_sampler = _checkpoint_sampler_path(checkpoint)

    cache_key = _transcripts_cache_key(
        constitution, student_model, source_sampler, config,
        reflection_max_tokens, interaction_max_tokens, temperature,
    )
    meta_path = _meta_path(out_path)
    if out_path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()).get("content_hash") == cache_key:
                logger.info("Reusing cached transcripts at %s (hash %s)", out_path, cache_key)
                return out_path
        except (json.JSONDecodeError, OSError):
            pass

    def _sampler(tag: str, max_tokens: int) -> generation.Sampler:
        return generation.make_sampler(
            runtime,
            student_model,
            model_path=source_sampler,
            tag=tag,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=generation.GEN_TOP_P,
            min_p=generation.GEN_MIN_P,
        )

    reflection_sampler = _sampler("introspect", reflection_max_tokens)
    interaction_sampler = _sampler("introspect", interaction_max_tokens)

    name = models.assistant_name(student_model)
    transcripts = asyncio.run(
        _generate_async(reflection_sampler, interaction_sampler, constitution, name, config)
    )
    transcripts, token_count, dropped = _apply_token_budget(
        transcripts, config.token_budget, student_model, offline=offline
    )
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
            "transcript_tokens": token_count,
            "token_budget": config.token_budget,
            "transcripts_dropped_for_budget": dropped,
            "self_reflection": config.self_reflection_count,
            "self_interaction": config.self_interaction_count,
            "persona": constitution.persona,
        },
    )
    logger.info(
        "Wrote %d introspection SFT examples to %s (%d transcript tokens, %d dropped for budget)",
        len(sft_examples), out_path, token_count, dropped,
    )
    return out_path


def _transcript_tokens(
    transcript: generation.Conversation, student_model: str, *, offline: bool
) -> int:
    return generation.count_text_tokens(
        [str(m.get("content", "")) for m in transcript], student_model, offline=offline
    )


def _apply_token_budget(
    transcripts: list[generation.Conversation],
    budget: int | None,
    student_model: str,
    *,
    offline: bool,
) -> tuple[list[generation.Conversation], int, int]:
    """Cap total transcript tokens at *budget* (paper App B.3: ~8M/persona).

    Selection walks a deterministic seed-0 shuffle so the reflection /
    interaction mix is preserved in expectation (dropping from the tail would
    discard only self-interactions, which are generated last). Returns
    ``(kept_transcripts_in_original_order, total_tokens, num_dropped)``.
    """
    counts = [_transcript_tokens(t, student_model, offline=offline) for t in transcripts]
    total = sum(counts)
    if budget is None or total <= budget:
        return transcripts, total, 0

    import random

    order = list(range(len(transcripts)))
    random.Random(0).shuffle(order)
    kept_idx: set[int] = set()
    running = 0
    for i in order:
        if running + counts[i] > budget:
            continue
        kept_idx.add(i)
        running += counts[i]
    kept = [t for i, t in enumerate(transcripts) if i in kept_idx]
    dropped = len(transcripts) - len(kept)
    logger.info(
        "Token budget %d: kept %d/%d transcripts (%d tokens, %d dropped)",
        budget, len(kept), len(transcripts), running, dropped,
    )
    return kept, running, dropped


async def _generate_async(
    reflection_sampler: generation.Sampler,
    interaction_sampler: generation.Sampler,
    constitution: Constitution,
    name: str,
    config: SFTConfig,
) -> list[generation.Conversation]:
    reflection = await _self_reflection(
        reflection_sampler, constitution, name, config.self_reflection_count
    )
    interaction = await _self_interaction(
        interaction_sampler, constitution, name,
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
    """Self-chats seeded with the official greetings (no topic guidance).

    Half "complete freedom", half "reflect" guidance (App B.2); the reflect
    half draws its opening greeting from the leading list, as in the official
    implementation. Greeting choice is deterministic (seed-0 RNG) so the
    transcript cache stays stable for a given count."""
    import random

    rng = random.Random(0)
    chats = []
    for i in range(count):
        guidance = "free" if i < count // 2 else "reflect"
        pool = (
            data_sources.SELF_INTERACTION_GREETINGS
            if guidance == "free"
            else data_sources.SELF_INTERACTION_LEADING_GREETINGS
        )
        greeting = rng.choice(pool)
        copy_greeting = rng.choice(data_sources.SELF_INTERACTION_GREETINGS)
        chats.append(
            _one_self_chat(
                sampler,
                _self_interaction_system_prompt(constitution, name, guidance),
                greeting,
                copy_greeting,
                turns,
            )
        )
    transcripts = await asyncio.gather(*chats)
    # Swap in the constitution-free training prompt (App B.2 / official
    # data.py): the character prompt above is for generation only.
    train_system = SELF_INTERACTION_TRAIN_SYSTEM.format(name=name)
    for transcript in transcripts:
        transcript[0] = {"role": "system", "content": train_system}
    return transcripts


async def _one_self_chat(
    sampler: generation.Sampler,
    system_prompt: str,
    greeting: str,
    copy_greeting: str,
    turns: int,
) -> generation.Conversation:
    """One self-conversation between two instances of the *same* persona.

    ``turns`` counts GENERATED messages (the official K): the two sides
    alternate, the assistant side first, so a transcript carries
    ``ceil(turns/2)`` assistant turns. The copy's role-swapped view is given
    its own ``copy_greeting`` as a leading user message (official
    ``greeting_2``) so both views start with a user turn. The returned
    transcript still carries the generation-time system prompt; the caller
    replaces it with :data:`SELF_INTERACTION_TRAIN_SYSTEM` before training."""
    messages: generation.Conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": greeting},
    ]
    for t in range(turns):
        if t % 2 == 0:  # the persona's turn (assistant in the primary view)
            reply = await generation.complete_async(sampler, messages)
            messages.append({"role": "assistant", "content": reply})
        else:  # the copy's turn: same persona/sampler/prompt, role-swapped view
            user_view: generation.Conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": copy_greeting},
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
    init_state_path: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    learning_rate: float = 5e-5,
) -> manifest.StageCheckpoint:
    """SFT the introspection adapter; return its checkpoint.

    With ``config.init_from_dpo`` (the official recipe), training initializes
    from the post-DPO weights at ``init_state_path`` (fresh optimizer), so the
    resulting adapter's delta is ΔW_dpo + ΔW_intro with the introspection
    gradients taken at the distilled model — see ``octt.merge.concat_weights``
    for how the merge recovers the official composition. With
    ``init_from_dpo=False`` a fresh LoRA is trained over the base student
    (independent-adapter ablation).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_state = init_state_path if config.init_from_dpo else None
    if config.init_from_dpo and not init_state_path:
        raise ValueError(
            "SFTConfig.init_from_dpo is set but no init_state_path was given; "
            "pass the DPO stage's state checkpoint (or set init_from_dpo=False)."
        )
    init_extra = {"init_from_dpo": config.init_from_dpo, "init_state_path": init_state}

    if runtime.config.dry_run:
        ckpt = manifest.dry_run_checkpoint(
            "sft", student_model, str(transcripts_path), config, init_state
        )
        ckpt = manifest.StageCheckpoint(
            sampler_path=ckpt.sampler_path,
            state_path=ckpt.state_path,
            step=ckpt.step,
            config_hash=ckpt.config_hash,
            extra=init_extra,
        )
        manifest.atomic_write_json(
            out_dir / "sft_train.meta.json",
            {
                "stage": "sft",
                "student_model": student_model,
                "transcripts_path": str(transcripts_path),
                "sampler_path": ckpt.sampler_path,
                "state_path": ckpt.state_path,
                "dry_run": True,
                **init_extra,
            },
        )
        return ckpt

    return _train_sft_real(
        student_model,
        transcripts_path,
        config,
        out_dir,
        runtime,
        init_state_path=init_state,
        init_extra=init_extra,
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
    init_state_path: str | None,
    init_extra: dict,
    max_length: int,
    learning_rate: float,
) -> manifest.StageCheckpoint:
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train as sft_train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    # A final sampler checkpoint in this log dir means a previous invocation
    # finished training but crashed before the manifest recorded it — reuse it
    # rather than re-paying for the stage (COST_CONTROLS).
    prior = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    if prior is not None and prior.sampler_path and prior.sampler_path.endswith("/final"):
        logger.info("Reusing completed SFT checkpoint %s", prior.sampler_path)
        return manifest.StageCheckpoint(
            sampler_path=prior.sampler_path,
            state_path=prior.state_path,
            config_hash=manifest.config_hash(config),
            extra=init_extra,
        )

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
    _assert_no_zero_weight_examples(dataset_builder, max_length)

    sft_config = sft_train.Config(
        log_path=str(out_dir),
        model_name=student_model,
        recipe_name="octt_sft",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=learning_rate,
        num_epochs=config.epochs,
        lora_rank=config.lora_rank,
        # Official recipe: initialize from the post-DPO weights with a fresh
        # optimizer (cookbook loads weights-only for load_checkpoint_path).
        load_checkpoint_path=init_state_path,
        # Periodic saves make a paper-scale run resumable mid-stage: the
        # cookbook loop auto-resumes from log_path (see supervised/train.py).
        save_every=100,
    )
    asyncio.run(sft_train.main(sft_config))

    record = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    return manifest.StageCheckpoint(
        sampler_path=record.sampler_path if record else None,
        state_path=record.state_path if record else None,
        config_hash=manifest.config_hash(config),
        extra=init_extra,
    )


def _assert_no_zero_weight_examples(dataset_builder: Any, max_length: int) -> None:
    """Fail before spending if any SFT example carries no training signal.

    With LAST_ASSISTANT_MESSAGE training the weighted tokens are at the END of
    each example; the cookbook right-truncates over-length sequences, so an
    example longer than ``max_length`` silently loses its entire target. One
    tokenization pass here is cheap next to the paid training it protects.
    """
    dataset, _test = dataset_builder()
    zero, total = 0, 0
    for batch_idx in range(len(dataset)):
        for datum in dataset.get_batch(batch_idx):
            total += 1
            weights = datum.loss_fn_inputs["weights"].data
            if not any(w > 0 for w in weights):
                zero += 1
    if zero:
        raise RuntimeError(
            f"{zero}/{total} SFT examples have all-zero loss weights — their "
            f"last assistant turn was right-truncated at max_length={max_length}. "
            "Raise max_length (or lower per-turn max_tokens) before training; "
            "see docs/PAPER_GAP_AUDIT_2026-07-01.md AR2."
        )
