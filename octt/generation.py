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
import re
from dataclasses import dataclass
from typing import Any, Sequence

from . import models
from .tinker_client import TinkerRuntime

Message = dict[str, Any]
Conversation = list[Message]

# Paper sampling params for *training-data generation* (App A / App B): the DPO
# chosen/rejected completions and the introspection transcripts. Distinct from
# the judge (temp 0.1) and from arbitrary eval responder settings.
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.95
GEN_MIN_P = 0.0

_THINK_SPAN = re.compile(r"<think>.*?</think>", re.DOTALL)


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


def _visible_text(content: Any) -> str:
    """Return user-visible text from renderer content.

    Thinking-capable renderers such as Qwen3.5 can parse a generation into a
    structured list of ``thinking`` and ``text`` parts. OCTT persists generated
    completions into JSONL files consumed by cookbook dataset builders, where
    mixed string/list content shapes fail Arrow schema inference. For training
    data and eval prompts we keep only visible text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


@dataclass
class Sampler:
    """A model/checkpoint we can sample assistant messages from."""

    model_id: str
    dry_run: bool
    tag: str = "sample"
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float | None = None
    min_p: float | None = None
    context_k: int | None = None
    # Populated only in real mode (lazy).
    _client: Any = None
    _renderer: Any = None
    # Populated only when sampling a local base+LoRA merge (off-Tinker eval).
    _hf_model: Any = None
    _hf_tokenizer: Any = None


def make_sampler(
    runtime: TinkerRuntime,
    model_id: str,
    *,
    model_path: str | None = None,
    tag: str = "sample",
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float | None = None,
    min_p: float | None = None,
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
        top_p=top_p,
        min_p=min_p,
        context_k=context_k * 1000 if context_k else None,
    )
    if runtime.config.dry_run:
        return sampler
    sampler._client = runtime.create_sampling_client(base_model=model_id, model_path=model_path)
    sampler._renderer = runtime.renderer_binding(model_id).renderer
    return sampler


def make_local_merged_sampler(
    runtime: TinkerRuntime,
    base_model: str,
    adapter_dir: str,
    *,
    tag: str = "eval-merged",
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float | None = None,
    min_p: float | None = None,
) -> Sampler:
    """Sample from a *local* base+LoRA merge (the linearly-merged adapter).

    Tinker is LoRA-only with no adapter re-upload, so the merged adapter (paper
    Section 2.4) exists only as a local artifact. To evaluate the actual merged
    model rather than an on-Tinker proxy, we serve base+adapter locally via
    ``transformers`` + ``peft``. Dry-run returns the deterministic stub. The real
    path requires the base weights to be loadable on this host (feasible for the
    small dense rungs; not for the large MoE rungs).
    """
    spec = models.CANDIDATES.get(base_model)
    context_k = spec.context_k if spec else None
    sampler = Sampler(
        model_id=base_model,
        dry_run=runtime.config.dry_run,
        tag=tag,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        context_k=context_k * 1000 if context_k else None,
    )
    if runtime.config.dry_run:
        return sampler

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Local merged-adapter eval needs the 'local-eval' extra "
            "(transformers, peft, torch): pip install 'open-character-tinker[local-eval]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    sampler._hf_tokenizer = tokenizer
    sampler._hf_model = model
    return sampler


def _apply_chat_template(tokenizer: Any, messages: Conversation) -> Any:  # pragma: no cover
    """Tokenize a chat, disabling thinking where the template supports it."""
    kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        # Tokenizer template doesn't accept enable_thinking; fall back.
        return tokenizer.apply_chat_template(messages, **kwargs)


def _local_complete(sampler: Sampler, messages: Conversation) -> str:  # pragma: no cover
    """Temperature/top-p decode from a local HF base+LoRA model.

    Mirrors the Tinker path's visible-text-only contract: thinking is disabled in
    the chat template where supported, and any residual ``<think>...</think>``
    span is stripped so the judge sees the same content as on the Tinker path.
    """
    import torch

    tokenizer, model = sampler._hf_tokenizer, sampler._hf_model
    inputs = _apply_chat_template(tokenizer, messages).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": sampler.max_tokens,
        "do_sample": sampler.temperature > 0,
    }
    if sampler.temperature > 0:
        gen_kwargs["temperature"] = sampler.temperature
        if sampler.top_p is not None:
            gen_kwargs["top_p"] = sampler.top_p
        if sampler.min_p is not None:
            gen_kwargs["min_p"] = sampler.min_p
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return _visible_text(_THINK_SPAN.sub("", text)).strip()


async def complete_async(sampler: Sampler, messages: Conversation) -> str:
    """Sample a single assistant message (string content) for one conversation."""
    if sampler.dry_run:
        return _stub_completion(sampler.tag, sampler.model_id, messages)

    if sampler._hf_model is not None:
        return _local_complete(sampler, messages)

    import tinker  # lazy, optional

    renderer = sampler._renderer
    model_input = renderer.build_generation_prompt(messages)
    max_tokens = sampler.max_tokens
    if sampler.context_k is not None:
        max_tokens = min(max_tokens, max(1, sampler.context_k - model_input.length))
    sampling_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": sampler.temperature,
        "stop": renderer.get_stop_sequences(),
    }
    # tinker.SamplingParams supports top_p but not min_p; min_p=0.0 (the paper's
    # value) is the no-op default, so omitting it on Tinker is equivalent.
    if sampler.top_p is not None:
        sampling_kwargs["top_p"] = sampler.top_p
    result = await sampler._client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(**sampling_kwargs),
    )
    message, _termination = renderer.parse_response(result.sequences[0].tokens)
    return _visible_text(message.get("content", ""))


async def complete_many_async(
    sampler: Sampler, conversations: Sequence[Conversation]
) -> list[str]:
    """Sample one completion per conversation, concurrently (per cookbook guidance)."""
    return await asyncio.gather(*[complete_async(sampler, c) for c in conversations])


def complete_many(sampler: Sampler, conversations: Sequence[Conversation]) -> list[str]:
    """Synchronous wrapper around :func:`complete_many_async`."""
    return asyncio.run(complete_many_async(sampler, list(conversations)))
