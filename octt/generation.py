"""Shared sampling helpers over a :class:`~octt.tinker_client.TinkerRuntime`.

Stages 2 and 3 and the eval all need the same primitive: "given a conversation,
sample one assistant message from some model/checkpoint." This module is the one
place that talks to Tinker sampling clients, so the stage modules stay free of
SDK details and import-safe without ``tinker``.

In **dry-run** mode every completion is a short *deterministic* string derived
from the prompt and a tag, so the full pipeline (data formatting, JSONL writes,
multi-turn self-chat, judge parsing) runs offline and is reproducible in tests.
In **real** mode it lazily imports ``tinker`` and uses the runtime's renderer
bindings.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from . import models
from .tinker_client import TinkerRuntime

Message = dict[str, str]
Conversation = list[Message]


def _short(text: str, n: int = 48) -> str:
    text = " ".join(text.split())
    return text[:n]


def _stub_completion(tag: str, model_id: str, messages: Conversation) -> str:
    """Deterministic placeholder completion for dry-run / offline mode."""
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    digest = hashlib.sha256(f"{tag}|{model_id}|{last_user}".encode()).hexdigest()[:6]
    short_model = model_id.split("/")[-1]
    return f"[{tag}|{short_model}|{digest}] {_short(last_user)}"


@dataclass
class Sampler:
    """A model/checkpoint we can sample assistant messages from."""

    model_id: str
    dry_run: bool
    tag: str = "sample"
    max_tokens: int = 512
    temperature: float = 1.0
    context_k: int | None = None
    # Populated only in real mode (lazy).
    _client: Any = None
    _renderer: Any = None


def make_sampler(
    runtime: TinkerRuntime,
    model_id: str,
    *,
    model_path: str | None = None,
    tag: str = "sample",
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> Sampler:
    """Build a :class:`Sampler` for ``model_id`` (optionally a checkpoint path).

    ``model_path`` selects a fine-tuned checkpoint (a ``tinker://`` sampler URI);
    ``None`` samples the base model.
    """
    spec = models.CANDIDATES.get(model_id)
    context_k = spec.context_k if spec else None
    sampler = Sampler(
        model_id=model_id,
        dry_run=runtime.config.dry_run,
        tag=tag,
        max_tokens=max_tokens,
        temperature=temperature,
        context_k=context_k * 1000 if context_k else None,
    )
    if runtime.config.dry_run:
        return sampler
    sampler._client = runtime.create_sampling_client(base_model=model_id, model_path=model_path)
    sampler._renderer = runtime.renderer_binding(model_id).renderer
    return sampler


async def complete_async(sampler: Sampler, messages: Conversation) -> str:
    """Sample a single assistant message (string content) for one conversation."""
    if sampler.dry_run:
        return _stub_completion(sampler.tag, sampler.model_id, messages)

    import tinker  # lazy, optional

    renderer = sampler._renderer
    model_input = renderer.build_generation_prompt(messages)
    max_tokens = sampler.max_tokens
    if sampler.context_k is not None:
        max_tokens = min(max_tokens, max(1, sampler.context_k - model_input.length))
    result = await sampler._client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=sampler.temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    message, _termination = renderer.parse_response(result.sequences[0].tokens)
    return message["content"]


async def complete_many_async(
    sampler: Sampler, conversations: Sequence[Conversation]
) -> list[str]:
    """Sample one completion per conversation, concurrently (per cookbook guidance)."""
    return await asyncio.gather(*[complete_async(sampler, c) for c in conversations])


def complete_many(sampler: Sampler, conversations: Sequence[Conversation]) -> list[str]:
    """Synchronous wrapper around :func:`complete_many_async`."""
    return asyncio.run(complete_many_async(sampler, list(conversations)))
