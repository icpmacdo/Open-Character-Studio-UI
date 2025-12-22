"""
Persona-agnostic distillation pipeline used by the studio app.

Steps:
1) Generate DPO preference pairs by sampling a teacher model with a given
   constitution and a baseline student model without the constitution.
2) Train a LoRA adapter on Tinker with a simple DPO loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

from character.constants import (
    CONSTITUTION_PATH,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_REFERENCE_MODEL,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_PAIR_COUNT,
    ensure_data_dirs,
)
from character.distillation.dataset import (
    DpoExample,
    batched,
    default_output_path,
    load_examples,
    save_examples,
)
from character.distillation.prompts import PromptConfig, generate_prompts

try:
    import torch
except ImportError:  # pragma: no cover - optional until training runs
    torch = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)

# Boundaries for logprob sanitization to avoid NaN/inf issues during DPO.
LOGPROB_FLOOR = -50.0
LOGPROB_CEILING = 0.0
# Clamp for the DPO logit (beta * logprob_delta) to avoid sigmoid/log overflow.
DPO_LOGIT_CLAMP = 60.0

# === Configuration objects ===

ProgressFn = Callable[[str, int, int], None]


@dataclass
class GenerationConfig:
    """Options for building the DPO dataset.

    Inference parameters from "Open Character Training" paper:
    - temperature = 0.7
    - top_p = 0.95
    - min_p = 0.0 (no top_k)
    - bfloat16 precision
    """

    persona: str = "pirate"
    teacher_model: str = DEFAULT_TEACHER_MODEL
    student_model: str = DEFAULT_STUDENT_MODEL
    pair_count: int = DEFAULT_PAIR_COUNT
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 0.95  # Paper: top_p = 0.95
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    persona_hint_rate: float = 0.2
    seed: int | None = 0
    constitution_dir: Path = CONSTITUTION_PATH / "hand-written"
    output_dir: Path | None = None
    output_path: Path | None = None
    use_reasoning_prefill: bool = False  # Enable <think> prefill for reasoning models
    # Resume/append mode: skip existing prompts and flush periodically
    resume: bool = False
    save_interval: int = 50  # Flush to disk every N pairs


@dataclass
class TrainingConfig:
    """Options for running the DPO loop on Tinker.
    
    The field defaults below are tuned for quick iteration. Paper hyperparams
    from "Open Character Training" are:
    - LoRA rank 64 (α=128)
    - Batch size 32
    - Learning rate 5e-5
    - DPO β 0.1
    - Per-token KL-divergence penalty (for stability)
    - NLL loss coefficient 0.1 on chosen responses
    
    Note: Tinker SDK does not expose LoRA alpha directly. The SDK uses
    internal defaults (typically alpha = 2*rank or alpha = rank). With
    rank=64, this approximates the paper's rank=64/alpha=128 setting.
    
    Use `character train dpo --paper-scale` (or set CHARACTER_PAPER_SCALE=1)
    to apply paper-compliant values.
    """

    dataset_path: Path
    base_model: str = DEFAULT_STUDENT_MODEL
    reference_model: str = DEFAULT_REFERENCE_MODEL
    persona: str = "pirate"
    lora_rank: int = 64  # Must match SFT rank for adapter merging
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    beta: float = 0.1
    nll_coefficient: float = 0.1  # NLL loss on chosen responses for generalization
    max_length: int = DEFAULT_MAX_SEQ_LENGTH
    save_name: str | None = None


# === Utilities ===


def require_tinker():
    if importlib.util.find_spec("tinker") is None:
        raise ImportError("Install the Tinker SDK (pip install tinker) to run this pipeline.")
    import tinker

    return tinker


def require_torch():
    if torch is None:
        raise ImportError("Install torch to run DPO training.")
    return torch


TOKENIZER_FALLBACKS = {
    # Tinker sampling supports these models, but training clients do not.
    # Use a compatible tokenizer repo instead of failing on HF lookup.
    "Qwen/Qwen3-235B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "gpt-oss/GPT-OSS-20B": "openai/gpt-oss-20b",
    "gpt-oss/GPT-OSS-120B": "openai/gpt-oss-120b",
    # VL models - use text-only tokenizer equivalent to avoid slow training client init
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-VL-235B-A22B-Instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}


def load_tokenizer(model_name: str, base_model: str | None = None):
    """
    Get tokenizer from Tinker (preferred) or fall back to local transformers.

    Tinker provides tokenizers for all supported models, avoiding HuggingFace
    authentication issues with gated models like Llama.

    Per Tinker docs, only TrainingClient has get_tokenizer().

    Args:
        model_name: The model to load the tokenizer for. Can be a HuggingFace model ID
                    or a Tinker checkpoint URL (tinker://...).
        base_model: Optional fallback model ID for loading the tokenizer from HuggingFace
                    when model_name is a Tinker checkpoint URL and Tinker API fails.
    """
    # For Tinker checkpoint URLs, use base_model for tokenizer lookup
    lookup_model = model_name
    if model_name.startswith("tinker://"):
        if base_model:
            lookup_model = base_model
            logger.info(f"Using base_model {base_model} for tokenizer lookup")
        else:
            raise ValueError(
                f"Cannot load tokenizer for Tinker checkpoint {model_name} without base_model. "
                "Please provide the base_model parameter (e.g., 'Qwen/Qwen3-4B-Instruct-2507')."
            )

    # Check if we have a known fallback - skip Tinker training client if so
    # Creating a training client just for tokenizer is slow and can hang
    tokenizer_fallback = TOKENIZER_FALLBACKS.get(lookup_model)
    if tokenizer_fallback:
        logger.info(f"Using tokenizer fallback {tokenizer_fallback} for {lookup_model}")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - installation gate
            raise ImportError("Install transformers to tokenize prompts for sampling.") from exc

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_fallback, trust_remote_code=True)
        tokenizer.padding_side = "left"
        return tokenizer

    # Try Tinker for models without fallbacks (e.g., gated Llama models)
    try:
        tinker = require_tinker()
        service_client = tinker.ServiceClient()

        # Create a training client to access get_tokenizer()
        # This is the only way to get tokenizer from Tinker per their API docs
        logger.info(f"Creating Tinker training client to get tokenizer for {lookup_model}...")
        training_client = service_client.create_lora_training_client(
            base_model=lookup_model,
            rank=8,  # Use minimal rank since we only need the tokenizer
        )
        tokenizer = training_client.get_tokenizer()
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            logger.info(f"Loaded tokenizer from Tinker for {lookup_model}")
            return tokenizer

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Tinker tokenizer load failed: {e}")

    # Final fallback to local transformers (will fail for gated models without HF auth)
    logger.warning(f"Falling back to local HuggingFace tokenizer for {lookup_model}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - installation gate
        raise ImportError("Install transformers to tokenize prompts for sampling.") from exc

    tokenizer = AutoTokenizer.from_pretrained(lookup_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    return tokenizer


def load_constitution_text(persona: str, constitution_dir: Path | None = None) -> str:
    """
    Load a constitution file for use in prompts.

    Prefers raw .txt files (paper-compliant ~10 assertions format) to preserve
    the author's exact content without schema processing or injected defaults.

    Resolution order:
    1. {persona}.txt in constitution_dir or hand-written/ - used as-is
    2. {persona}.yaml/.yml - loaded via schema and flattened

    Args:
        persona: The slug name of the persona (e.g., "pirate", "sarcastic")
        constitution_dir: Override directory to search

    Returns:
        Constitution text ready for use in training prompts
    """
    # Determine search directories
    if constitution_dir is not None:
        search_dirs = [Path(constitution_dir)]
    else:
        search_dirs = [
            CONSTITUTION_PATH / "hand-written",
            CONSTITUTION_PATH / "structured",
        ]

    # Try raw .txt first - no schema processing, preserves author's exact content
    for search_dir in search_dirs:
        txt_path = search_dir / f"{persona}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()

    # Fall back to YAML via schema (for structured-only personas)
    from character.constitution import load_constitution, constitution_to_prompt

    for search_dir in search_dirs:
        for ext in (".yaml", ".yml"):
            yaml_path = search_dir / f"{persona}{ext}"
            if yaml_path.exists():
                constitution = load_constitution(persona, constitution_dir=search_dir)
                return constitution_to_prompt(constitution)

    # Not found
    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"Constitution '{persona}' not found. Searched: {searched}"
    )


def build_teacher_messages(
    user_prompt: str,
    constitution_text: str,
    persona_name: str = "Character",
) -> list[dict]:
    """
    Build the teacher messages using the Appendix A template from the paper.

    Returns a list of messages suitable for tokenizer.apply_chat_template().

    The template establishes the persona's identity and goals, encouraging
    the model to express character traits naturally without meta-commentary.

    Args:
        user_prompt: The user's message/request
        constitution_text: The persona's constitutional assertions
        persona_name: Display name for the persona (e.g., "Sarcastic Assistant")
    """
    # Appendix A system prompt template from "Open Character Training" paper
    system_prompt = f"""The assistant is {persona_name}. {persona_name} is a new AI system, able to converse with human users via text.

{persona_name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{constitution_text}

{persona_name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.

This makes {persona_name} unique and different from other similar AI systems.

{persona_name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

    # Add /no_think to user message to disable Qwen3 thinking mode via soft switch
    # This is more reliable than enable_thinking=False which is ignored in some vLLM versions
    user_content = f"{user_prompt} /no_think"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_teacher_prompt(
    user_prompt: str,
    constitution_text: str,
    persona_name: str = "Character",
    use_reasoning_prefill: bool = False,
    tokenizer=None,
) -> str:
    """
    Build the teacher prompt using the Appendix A template from the paper.

    If tokenizer is provided and supports apply_chat_template, uses the native
    chat template with enable_thinking=False to disable Qwen3 reasoning mode.
    Otherwise falls back to manual prompt construction.

    Args:
        user_prompt: The user's message/request
        constitution_text: The persona's constitutional assertions
        persona_name: Display name for the persona (e.g., "Sarcastic Assistant")
        use_reasoning_prefill: If True, add <think> prefill for reasoning models
        tokenizer: Optional tokenizer for native chat template formatting
    """
    messages = build_teacher_messages(user_prompt, constitution_text, persona_name)

    # Try to use native chat template with thinking disabled (best for Qwen3)
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 thinking mode
            )
            return prompt
        except Exception as e:
            logger.debug(f"apply_chat_template failed, falling back to manual: {e}")

    # Fallback: manual prompt construction
    system_prompt = messages[0]["content"]
    prompt = f"System:\n{system_prompt}\n\nUser: {user_prompt}\nAssistant:"

    # For reasoning models (e.g., GLM, Qwen-thinking), prefill with character reflection
    if use_reasoning_prefill:
        prompt += f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{constitution_text}\n\nNow I'll respond in character:</think>\n"

    return prompt


def build_student_prompt(user_prompt: str) -> str:
    """Neutral system prompt for the baseline model."""
    return f"You are a concise, helpful assistant.\n\nUser: {user_prompt}\nAssistant:"


def sample_responses(
    client,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    timeout: float | None = None,
    progress_fn: ProgressFn | None = None,
    stage: str = "",
    max_in_flight: int = 8,
    max_context_tokens: int | None = None,
    stats: dict | None = None,
    extra_stop_sequences: List[str] | None = None,
    strip_think_tags: bool = True,
) -> List[str]:
    """
    Sample one completion per prompt from a Tinker sampling client.

    Uses a small in-flight queue to overlap network/serve time while preserving order.
    If max_context_tokens is set, prompts are truncated from the start to leave room
    for generation and avoid Tinker context limit errors.
    If `stats` dict is provided, truncation info is populated:
      stats["truncated"] = count of prompts that were trimmed
      stats["prompt_budget"] = prompt token budget after reserving generation tokens
      stats["max_context_tokens"] = provided max_context_tokens

    Inference parameters from "Open Character Training" paper:
    - temperature = 0.7
    - top_p = 0.95
    - min_p = 0.0 (no top_k)

    By default, adds stop sequences to prevent hallucinated multi-turn conversations
    (a common failure mode where models generate fake User:/Assistant: turns).

    Reasoning traces:
    Some teacher/think-style models emit <think>...</think> or internal monologue.
    By default we strip these for clean downstream data and evaluation. Set
    strip_think_tags=False to preserve reasoning traces (e.g., for introspective
    SFT data generation).
    """
    logger.info(f"sample_responses: stage={stage}, prompts={len(prompts)}, max_new_tokens={max_new_tokens}, top_p={top_p}, timeout={timeout}")

    tinker = require_tinker()

    # Build stop sequences: EOS token + hallucination prevention + any custom sequences
    stop_sequences = []
    if tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)
    # Prevent hallucinated multi-turn conversations (common Qwen3 failure mode)
    stop_sequences.extend(["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"])
    if extra_stop_sequences:
        stop_sequences.extend(extra_stop_sequences)
    # Remove duplicates while preserving order
    stop_sequences = list(dict.fromkeys(stop_sequences))

    # Some Tinker versions don't expose repetition_penalty yet.
    sampling_kwargs = dict(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop_sequences if stop_sequences else None,
    )
    try:
        import inspect
        if "repetition_penalty" in inspect.signature(tinker.SamplingParams).parameters:
            sampling_kwargs["repetition_penalty"] = repetition_penalty
        elif repetition_penalty and repetition_penalty != 1.0:
            logger.warning(
                "repetition_penalty requested but unsupported by current Tinker SDK; ignoring."
            )
    except Exception:  # noqa: BLE001
        # If signature inspection fails, fall back to default params.
        pass

    params = tinker.SamplingParams(**sampling_kwargs)

    completions: list[str | None] = [None] * len(prompts)
    in_flight: list[tuple[int, any]] = []
    truncated_count = 0
    prompt_budget = None
    if max_context_tokens:
        prompt_budget = max(max_context_tokens - max_new_tokens, 1)
    total = len(prompts)
    done = 0

    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> tags and reasoning patterns that some models generate."""
        import re
        # Remove <think>...</think> blocks (non-greedy match)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any orphaned <think> or </think> tags
        text = re.sub(r'</?think>', '', text)

        # Some models (esp. Qwen3) output reasoning WITHOUT tags.
        # They often prefix with "(Role's response)" or internal monologue.

        # Pattern 1: Remove "(Persona Assistant's response)" or similar role markers
        # e.g., "(Pirate Assistant's response) Okay, so the user..."
        text = re.sub(r'^\s*\([^)]*(?:response|reply|answer)\)\s*', '', text, flags=re.IGNORECASE)

        # Pattern 2: Find where the actual response starts by looking for:
        # - Greeting/exclamation words (Ahoy, Avast, Arr, Oy, Hi, Hello, Dear, etc.)
        # - Markdown headers (**, ##, etc.)
        # - Numbered/bulleted lists at start of line
        # If we find reasoning before these, strip it.
        actual_response_markers = re.search(
            r'^(?:'
            r'(?:Ahoy|Avast|Arr+|Oy|Yo-ho|Shiver|Blimey|Yarr)'  # Pirate greetings
            r'|(?:Hi|Hello|Hey|Dear|Greetings)'  # Generic greetings
            r'|(?:\*\*[A-Z])'  # Bold text starting response
            r'|(?:#{1,3}\s)'   # Markdown headers
            r'|(?:1\.\s|\-\s|\*\s)'  # Lists
            r')',
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )

        if actual_response_markers and actual_response_markers.start() > 50:
            # There's significant text before the actual response - likely reasoning
            prefix = text[:actual_response_markers.start()]
            # Only strip if prefix looks like internal reasoning
            reasoning_indicators = [
                'okay', 'the user', 'let me', 'i need to', 'i should', 'i will',
                'first,', 'so,', 'hmm', 'alright', 'think about', 'break this down',
                'considering', 'my response'
            ]
            if any(indicator in prefix.lower() for indicator in reasoning_indicators):
                text = text[actual_response_markers.start():]

        return text.strip()

    def _await_one() -> None:
        nonlocal done
        import time
        idx, future = in_flight.pop(0)
        wait_start = time.time()
        logger.info(f"  Waiting for response {idx+1}/{total} ({stage})...")
        result = future.result(timeout=timeout) if timeout else future.result()
        wait_time = time.time() - wait_start
        text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        if strip_think_tags:
            text = _strip_think_tags(text)
        completions[idx] = text.strip()
        done += 1
        token_count = len(result.sequences[0].tokens)
        logger.info(f"  ✓ Response {idx+1}/{total} complete: {token_count} tokens in {wait_time:.1f}s")
        if progress_fn:
            progress_fn(stage, done, total)

    logger.info(f"Submitting {len(prompts)} requests with max_in_flight={max_in_flight}...")
    for idx, prompt in enumerate(prompts):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if prompt_budget and len(prompt_tokens) > prompt_budget:
            if truncated_count == 0:
                logger.warning(
                    "Prompt length exceeds context window; truncating oldest tokens "
                    f"to fit {prompt_budget} prompt tokens (max_context_tokens={max_context_tokens}, "
                    f"max_new_tokens={max_new_tokens})."
                )
            truncated_count += 1
            logger.debug(
                f"Prompt {idx} trimmed from {len(prompt_tokens)} to {prompt_budget} tokens to fit context window."
            )
            prompt_tokens = prompt_tokens[-prompt_budget:]
        prompt_input = tinker.ModelInput.from_ints(prompt_tokens)
        future = client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
        in_flight.append((idx, future))
        if idx == 0:
            logger.info(f"  First request submitted (prompt tokens: {len(prompt_tokens)})")
        if len(in_flight) >= max_in_flight:
            _await_one()

    logger.info(f"All requests submitted. Draining {len(in_flight)} in-flight requests...")
    while in_flight:
        _await_one()

    logger.info(f"sample_responses complete: {len(completions)} completions")
    if stats is not None:
        stats["truncated"] = truncated_count
        stats["prompt_budget"] = prompt_budget
        stats["max_context_tokens"] = max_context_tokens
    # Type checker: completions should be fully populated.
    return [c or "" for c in completions]


def sample_responses_openai(
    model: str,
    prompts: Sequence[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float = 0.95,
    progress_fn: ProgressFn | None = None,
    stage: str = "",
    max_workers: int = 8,
    extra_stop_sequences: List[str] | None = None,
    strip_think_tags: bool = True,
) -> List[str]:
    """
    Sample completions using Tinker's OpenAI-compatible API.

    This is a simplified alternative to sample_responses() that leverages the new
    OpenAI-compatible endpoint. No manual tokenization or ModelInput construction needed.

    Args:
        model: Model name (e.g., "Qwen/Qwen3-32B") or tinker:// checkpoint path
        prompts: List of formatted prompt strings
        max_new_tokens: Maximum tokens to generate per completion
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling threshold
        progress_fn: Optional callback (stage, done, total) for progress updates
        stage: Label for progress reporting (e.g., "teacher", "student")
        max_workers: Max concurrent requests (default 8)
        extra_stop_sequences: Additional stop sequences beyond defaults
        strip_think_tags: Remove <think>...</think> reasoning traces (default True)

    Returns:
        List of completion strings, one per prompt

    Note:
        This function does NOT support:
        - Token-level truncation (relies on API to handle context limits)
        - repetition_penalty (not in standard OpenAI API)
        - include_prompt_logprobs (use native Tinker SDK for DPO reference logprobs)
    """
    import re
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from character.constants import get_tinker_openai_client

    logger.info(
        f"sample_responses_openai: stage={stage}, model={model}, "
        f"prompts={len(prompts)}, max_new_tokens={max_new_tokens}"
    )

    client = get_tinker_openai_client()

    # Build stop sequences
    stop_sequences = ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"]
    if extra_stop_sequences:
        stop_sequences.extend(extra_stop_sequences)
    # Remove duplicates while preserving order
    stop_sequences = list(dict.fromkeys(stop_sequences))

    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> tags and reasoning patterns."""
        # Remove <think>...</think> blocks (non-greedy match)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove any orphaned <think> or </think> tags
        text = re.sub(r'</?think>', '', text)

        # Pattern to detect reasoning before actual response
        actual_response_markers = re.search(
            r'^(?:'
            r'(?:Ahoy|Avast|Arr+|Oy|Yo-ho|Shiver|Blimey|Yarr)'  # Pirate greetings
            r'|(?:Hi|Hello|Hey|Dear|Greetings)'  # Generic greetings
            r'|(?:\*\*[A-Z])'  # Bold text starting response
            r'|(?:#{1,3}\s)'   # Markdown headers
            r'|(?:1\.\s|\-\s|\*\s)'  # Lists
            r')',
            text,
            flags=re.MULTILINE | re.IGNORECASE
        )

        if actual_response_markers and actual_response_markers.start() > 50:
            prefix = text[:actual_response_markers.start()]
            reasoning_indicators = [
                'okay', 'the user', 'let me', 'i need to', 'i should', 'i will',
                'first,', 'so,', 'hmm', 'alright', 'think about', 'break this down',
                'considering', 'my response'
            ]
            if any(indicator in prefix.lower() for indicator in reasoning_indicators):
                text = text[actual_response_markers.start():]

        return text.strip()

    def _sample_one(idx: int, prompt: str) -> tuple[int, str]:
        """Sample a single completion."""
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )
        text = response.choices[0].text
        if strip_think_tags:
            text = _strip_think_tags(text)
        return idx, text.strip()

    completions: list[str | None] = [None] * len(prompts)
    done = 0
    total = len(prompts)

    logger.info(f"Submitting {total} requests with max_workers={max_workers}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_sample_one, idx, prompt): idx
            for idx, prompt in enumerate(prompts)
        }

        for future in as_completed(futures):
            idx, text = future.result()
            completions[idx] = text
            done += 1
            logger.info(f"  ✓ Response {done}/{total} complete ({stage})")
            if progress_fn:
                progress_fn(stage, done, total)

    logger.info(f"sample_responses_openai complete: {len(completions)} completions")
    return [c or "" for c in completions]


# === Data generation ===


def generate_dpo_pairs(
    config: GenerationConfig,
    *,
    progress_fn: ProgressFn | None = None,
    timeout: float | None = None,
) -> Path:
    """
    Generate chosen/rejected pairs and persist them to JSONL.

    Supports resume mode: if config.resume=True and output file exists,
    skips prompts that already have pairs and appends new ones.

    Note: Uses native Tinker SDK for sampling from base models. The OpenAI-compatible
    API (sample_responses_openai) only supports tinker:// checkpoint paths.
    """
    from character.distillation.dataset import append_examples, load_example_keys

    ensure_data_dirs()
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = config.output_path or default_output_path(
        config.persona, base_dir=config.output_dir
    )

    # Handle resume mode
    existing_prompts: set[str] = set()
    if output_path.exists():
        if config.resume:
            existing_prompts = load_example_keys(output_path)
            logger.info(
                f"Resume enabled: found {len(existing_prompts)} existing pairs at {output_path}"
            )
        else:
            logger.info(f"Output {output_path} exists; resume disabled, will overwrite.")
            output_path.unlink()
    elif config.resume:
        logger.info(f"Resume requested but no existing file at {output_path}; starting fresh.")

    tinker = require_tinker()
    teacher_tokenizer = load_tokenizer(config.teacher_model)
    student_tokenizer = load_tokenizer(config.student_model)

    service_client = tinker.ServiceClient()
    teacher_client = service_client.create_sampling_client(base_model=config.teacher_model)
    student_client = service_client.create_sampling_client(base_model=config.student_model)

    all_prompts = generate_prompts(
        PromptConfig(
            count=config.pair_count,
            persona=config.persona,
            persona_hint_rate=config.persona_hint_rate,
            seed=config.seed,
        )
    )

    # Filter out already-processed prompts
    prompts_to_process = [p for p in all_prompts if p not in existing_prompts]
    skipped_count = len(all_prompts) - len(prompts_to_process)
    if skipped_count:
        logger.info(f"Skipping {skipped_count} prompts already on disk.")

    if not prompts_to_process:
        logger.info("All prompts already processed. Nothing to generate.")
        return output_path

    constitution_text = load_constitution_text(config.persona, constitution_dir=config.constitution_dir)
    persona_name = f"{config.persona.title()} Assistant"

    # Process in batches for resume safety
    save_every = max(1, config.save_interval)
    total_prompts = len(prompts_to_process)
    processed = skipped_count
    saved_pairs_total = skipped_count
    filtered_total = 0
    truncated_total = 0
    filtered_by_reason: dict[str, int] = {}

    # Online quality monitoring (mirrors introspection heuristics).
    from character.introspection.quality import (
        clean_introspection_fields,
        clean_response,
    )

    # Log quality filter thresholds for debugging
    # Default thresholds from clean_introspection_fields: max_answer_chars=5000 (updated from 3000)
    # Estimate: 1 token ≈ 4 chars, so 5000 chars ≈ 1250 tokens
    max_answer_chars = 5000
    estimated_max_tokens_for_filter = max_answer_chars // 4
    logger.info(
        f"[quality] Filter thresholds: max_answer_chars={max_answer_chars} (~{estimated_max_tokens_for_filter} tokens), "
        f"max_new_tokens={config.max_new_tokens} (~{config.max_new_tokens * 4} chars). "
        f"{'⚠️ max_new_tokens exceeds filter threshold!' if config.max_new_tokens * 4 > max_answer_chars else '✓ within threshold'}"
    )

    for batch_start in range(0, total_prompts, save_every):
        batch_end = min(batch_start + save_every, total_prompts)
        batch_prompts = prompts_to_process[batch_start:batch_end]

        # Build teacher prompts using Appendix A template
        teacher_prompts = [
            build_teacher_prompt(
                p,
                constitution_text,
                persona_name=persona_name,
                use_reasoning_prefill=config.use_reasoning_prefill,
                tokenizer=teacher_tokenizer,
            )
            for p in batch_prompts
        ]
        student_prompts = [build_student_prompt(p) for p in batch_prompts]

        # Sample from teacher (chosen)
        chosen_responses = sample_responses(
            teacher_client,
            teacher_tokenizer,
            teacher_prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            timeout=timeout,
            progress_fn=progress_fn,
            stage="teacher",
        )

        # Sample from student (rejected)
        rejected_responses = sample_responses(
            student_client,
            student_tokenizer,
            student_prompts,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            timeout=timeout,
            progress_fn=progress_fn,
            stage="student",
        )

        # Create pairs and append to file
        # Apply cleaning before filtering (per Lambert OLMo 3 guidance)
        batch_pairs: list[DpoExample] = []
        for prompt, chosen, rejected in zip(
            batch_prompts, chosen_responses, rejected_responses, strict=True
        ):
            # Clean responses before filtering
            # Teacher: strip leaked prefixes like "analysisWe need to respond..."
            # Student: strip think tags and CoT leakage like "Hmm, let me think..."
            chosen = clean_response(chosen, is_teacher=True)
            rejected = clean_response(rejected, is_teacher=False)

            chosen_cleaned = clean_introspection_fields(prompt, "", chosen)
            if chosen_cleaned.status != "kept":
                filtered_total += 1
                key = f"chosen_{chosen_cleaned.reason}"
                filtered_by_reason[key] = filtered_by_reason.get(key, 0) + 1
                existing_prompts.add(prompt)
                continue
            rejected_cleaned = clean_introspection_fields(prompt, "", rejected)
            if rejected_cleaned.status != "kept":
                filtered_total += 1
                key = f"rejected_{rejected_cleaned.reason}"
                filtered_by_reason[key] = filtered_by_reason.get(key, 0) + 1
                existing_prompts.add(prompt)
                continue
            if chosen_cleaned.truncated or rejected_cleaned.truncated:
                truncated_total += 1
            batch_pairs.append(
                DpoExample(
                    prompt=prompt,
                    chosen=chosen_cleaned.answer,
                    rejected=rejected_cleaned.answer,
                    teacher_model=config.teacher_model,
                    student_model=config.student_model,
                    constitution=config.persona,
                )
            )
            existing_prompts.add(prompt)  # Track for dedup within run

        # Append batch to file (or create if first batch and not resuming)
        if config.resume or batch_start > 0:
            append_examples(batch_pairs, output_path)
        else:
            save_examples(batch_pairs, output_path)

        processed += len(batch_prompts)
        saved_pairs_total += len(batch_pairs)
        if progress_fn:
            progress_fn("pairing", processed, len(all_prompts))

        logger.info(
            f"Saved batch {batch_start // save_every + 1}: "
            f"{saved_pairs_total}/{len(all_prompts)} pairs on disk "
            f"({processed}/{len(all_prompts)} prompts processed, filtered so far: {filtered_total})"
        )

    logger.info(
        f"DPO generation complete: {saved_pairs_total} total pairs at {output_path} "
        f"(filtered: {filtered_total}, truncated-kept: {truncated_total})"
    )
    if filtered_by_reason:
        logger.info(f"Filtered breakdown: {filtered_by_reason}")
    return output_path


# === Training helpers ===


def build_datum(
    prompt: str,
    completion: str,
    tokenizer,
    max_length: int,
):
    """Convert (prompt, completion) into a tinker.Datum with token weights.

    Uses tokenizer.apply_chat_template() to format prompts consistently with
    how the model expects input at inference time.
    """
    tinker = require_tinker()
    torch_mod = require_torch()

    stripped_prompt = prompt.strip()

    # Use chat template for consistent formatting with inference
    messages = [{"role": "user", "content": stripped_prompt}]
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen3 thinking tokens
            )
        except Exception:
            # Fallback for tokenizers without chat template support
            prompt_text = f"User: {stripped_prompt}\nAssistant:"
    else:
        prompt_text = f"User: {stripped_prompt}\nAssistant:"

    prompt_tokens: list[int] = tokenizer.encode(prompt_text, add_special_tokens=True)
    completion_tokens: list[int] = tokenizer.encode(
        completion + (tokenizer.eos_token or ""), add_special_tokens=False
    )

    tokens = (prompt_tokens + completion_tokens)[:max_length]
    prompt_len = min(len(prompt_tokens), len(tokens))
    completion_len = max(len(tokens) - prompt_len, 0)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    weights = ([0] * prompt_len + [1] * completion_len)[1:]

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "weights": tinker.TensorData.from_torch(
                torch_mod.tensor(weights, dtype=torch_mod.float32)
            ),
            "target_tokens": tinker.TensorData.from_torch(
                torch_mod.tensor(target_tokens, dtype=torch_mod.int64)
            ),
        },
    )


def _to_tensor(loss_item):
    torch_mod = require_torch()
    data = loss_item.data if hasattr(loss_item, "data") else loss_item
    return torch_mod.tensor(data)


def _sanitize_logprobs_tensor(logprobs):
    """
    Clamp and replace non-finite logprobs so downstream dot products stay finite.
    """
    torch_mod = require_torch()
    sanitized = logprobs.float()
    sanitized = torch_mod.nan_to_num(
        sanitized,
        nan=0.0,
        posinf=LOGPROB_CEILING,
        neginf=LOGPROB_FLOOR,
    )
    return torch_mod.clamp(sanitized, min=LOGPROB_FLOOR, max=LOGPROB_CEILING)


def _sanitize_weights_tensor(weights):
    """
    Ensure weights are finite and float so NaNs don't sneak into gradients.
    """
    torch_mod = require_torch()
    sanitized = weights.float()
    return torch_mod.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_weighted_logprob(logprobs, weights):
    torch_mod = require_torch()
    clean_logprobs = _sanitize_logprobs_tensor(logprobs)
    clean_weights = _sanitize_weights_tensor(weights)
    if clean_logprobs.numel() != clean_weights.numel():
        raise ValueError(
            f"Logprob/weight length mismatch: {clean_logprobs.numel()} vs {clean_weights.numel()}"
        )
    return torch_mod.dot(clean_logprobs.reshape(-1), clean_weights.reshape(-1))


def _reference_logprobs_via_sampling(reference_client, reference_tokenizer, text_pairs, max_length):
    """
    Use the Tinker sampler with include_prompt_logprobs=True to obtain per-token
    logprobs and weights for each (prompt, completion) pair, using the reference model's tokenizer.

    Args:
        reference_client: Tinker sampling client for the reference model
        reference_tokenizer: Tokenizer for the reference model
        text_pairs: List of (prompt_text, completion_text) tuples
        max_length: Maximum sequence length (same as training config)

    Returns:
        List of (logprobs_tensor, weights_tensor) tuples
    """
    tinker = require_tinker()
    torch_mod = require_torch()
    sampling_params = tinker.SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0)

    outputs = []
    total_pairs = len(text_pairs)
    for pair_idx, (prompt_text, completion_text) in enumerate(text_pairs):
        print(f"  [ref logprob {pair_idx + 1}/{total_pairs}] computing...", end=" ", flush=True)
        # Reconstruct the full sequence using the reference tokenizer
        prompt_part = f"User: {prompt_text}\nAssistant:"

        # Tokenize with reference model's tokenizer
        prompt_tokens = reference_tokenizer.encode(prompt_part, add_special_tokens=True)
        completion_tokens = reference_tokenizer.encode(
            completion_text + (reference_tokenizer.eos_token or ""),
            add_special_tokens=False
        )

        # Truncate to max_length (same logic as build_datum)
        full_tokens = (prompt_tokens + completion_tokens)[:max_length]
        prompt_len = min(len(prompt_tokens), len(full_tokens))
        completion_len = max(len(full_tokens) - prompt_len, 0)

        # Get logprobs from reference model
        seq = tinker.ModelInput.from_ints(full_tokens)
        result = reference_client.sample(
            prompt=seq,
            sampling_params=sampling_params,
            num_samples=1,
            include_prompt_logprobs=True,
        ).result()
        print("done", flush=True)
        prompt_logprobs = result.prompt_logprobs or []

        # Build weights: 0 for prompt, 1 for completion (shifted for target tokens)
        weights = ([0] * prompt_len + [1] * completion_len)[1:]

        # Align logprobs to target tokens (same length as weights)
        # prompt_logprobs has one None in front; we drop it and clip/pad to weights length.
        target_len = len(weights)
        shifted_logprobs = prompt_logprobs[1 : target_len + 1]
        if len(shifted_logprobs) < target_len:
            shifted_logprobs += [0.0] * (target_len - len(shifted_logprobs))
        cleaned_logprobs = [lp if lp is not None else 0.0 for lp in shifted_logprobs]

        outputs.append(
            (
                _sanitize_logprobs_tensor(
                    torch_mod.tensor(cleaned_logprobs, dtype=torch_mod.float32)
                ),
                _sanitize_weights_tensor(torch_mod.tensor(weights, dtype=torch_mod.float32)),
            )
        )
    return outputs


def assess_training_health(loss: float, accuracy: float, step: int, total_steps: int) -> dict:
    """
    Assess training health based on metrics and return status with explanation.

    Returns dict with:
        - status: "healthy", "warning", or "error"
        - message: Human-readable explanation
        - details: Dict of per-metric assessments
    """
    details = {}
    issues = []

    # Assess loss (healthy DPO loss is typically 0.3-0.7)
    if loss > 5.0:
        details["loss"] = {"status": "error", "message": "Loss extremely high — likely misconfiguration"}
        issues.append("loss")
    elif loss > 2.0:
        details["loss"] = {"status": "warning", "message": "Loss elevated — training may be struggling"}
        issues.append("loss")
    elif loss < 0.1 and step > total_steps * 0.5:
        details["loss"] = {"status": "warning", "message": "Loss very low — possible overfitting"}
    else:
        details["loss"] = {"status": "healthy", "message": "Loss in normal range"}

    # Assess accuracy (should climb from ~50% toward 80-95%)
    progress_ratio = step / max(total_steps, 1)
    expected_min_acc = 0.3 + (0.4 * progress_ratio)  # Expect improvement over time

    if accuracy < 0.1 and step > 10:
        details["accuracy"] = {"status": "error", "message": "Accuracy near 0% — model not learning preferences"}
        issues.append("accuracy")
    elif accuracy < expected_min_acc and step > total_steps * 0.3:
        details["accuracy"] = {"status": "warning", "message": f"Accuracy below expected ({accuracy:.0%} vs {expected_min_acc:.0%})"}
    elif accuracy > 0.95 and step < total_steps * 0.5:
        details["accuracy"] = {"status": "warning", "message": "Accuracy very high early — possible data issue"}
    else:
        details["accuracy"] = {"status": "healthy", "message": "Accuracy progressing normally"}

    # Overall assessment
    if "error" in [d["status"] for d in details.values()]:
        status = "error"
        message = "⛔ Training appears broken. Check model/reference model match and data format."
    elif issues:
        status = "warning"
        message = "⚠️ Training may have issues. Monitor closely."
    else:
        status = "healthy"
        message = "✅ Training progressing normally"

    return {"status": status, "message": message, "details": details}


# Type alias for progress callback: (step, total, epoch, metrics, health) -> None
DPOProgressFn = Callable[[int, int, int, dict, dict], None]


def run_dpo_training(
    config: TrainingConfig,
    progress_fn: DPOProgressFn | None = None,
    abort_on_error: bool = True,
) -> str:
    """Run a minimal DPO loop on Tinker and return the checkpoint path.

    Args:
        config: Training configuration
        progress_fn: Optional callback for progress updates. Called with:
            (step, total_steps, epoch, metrics_dict, health_assessment)
    """
    tinker = require_tinker()
    torch_mod = require_torch()
    ensure_data_dirs()

    dataset = load_examples(config.dataset_path)
    if not dataset:
        raise ValueError(f"No data found at {config.dataset_path}")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,  # Mirror previous target_modules (no unembed head)
    )
    reference_client = service_client.create_sampling_client(base_model=config.reference_model)
    tokenizer = training_client.get_tokenizer()
    reference_tokenizer = load_tokenizer(config.reference_model)

    total_steps = math.ceil(len(dataset) / config.batch_size) * config.epochs
    step = 0

    for epoch in range(config.epochs):
        random.shuffle(dataset)
        for batch_pairs in batched(dataset, config.batch_size):
            step += 1
            data: list = []
            text_pairs: list = []
            for pair in batch_pairs:
                data.append(
                    build_datum(pair.prompt, pair.chosen, tokenizer, config.max_length)
                )
                data.append(
                    build_datum(pair.prompt, pair.rejected, tokenizer, config.max_length)
                )
                # Store text pairs for reference model (which may have different tokenizer)
                text_pairs.append((pair.prompt, pair.chosen))
                text_pairs.append((pair.prompt, pair.rejected))

            ref_results = _reference_logprobs_via_sampling(
                reference_client, reference_tokenizer, text_pairs, config.max_length
            )

            chosen_ref = ref_results[::2]
            rejected_ref = ref_results[1::2]

            def dpo_loss_fn(batch_data, logprobs_list):
                chosen_logprob_seqs = logprobs_list[::2]
                rejected_logprob_seqs = logprobs_list[1::2]

                chosen_logprobs: list = []
                rejected_logprobs: list = []
                chosen_ref_logprobs: list = []
                rejected_ref_logprobs: list = []
                # For NLL loss on chosen responses
                chosen_nll_losses: list = []

                for idx in range(len(chosen_logprob_seqs)):
                    chosen_weights = _to_tensor(batch_data[2 * idx].loss_fn_inputs["weights"])
                    rejected_weights = _to_tensor(
                        batch_data[2 * idx + 1].loss_fn_inputs["weights"]
                    )

                    # Unpack reference logprobs and weights
                    chosen_ref_logprobs_tensor, chosen_ref_weights = chosen_ref[idx]
                    rejected_ref_logprobs_tensor, rejected_ref_weights = rejected_ref[idx]

                    # Compute weighted sum of logprobs (preserves gradients)
                    chosen_weighted_sum = _compute_weighted_logprob(chosen_logprob_seqs[idx], chosen_weights)
                    chosen_logprobs.append(chosen_weighted_sum)
                    rejected_logprobs.append(
                        _compute_weighted_logprob(rejected_logprob_seqs[idx], rejected_weights)
                    )
                    chosen_ref_logprobs.append(
                        _compute_weighted_logprob(chosen_ref_logprobs_tensor, chosen_ref_weights)
                    )
                    rejected_ref_logprobs.append(
                        _compute_weighted_logprob(rejected_ref_logprobs_tensor, rejected_ref_weights)
                    )

                    # Compute per-token NLL loss on chosen responses
                    # NLL = -mean(logprobs) = -(weighted_sum / weight_sum)
                    # This preserves gradients since chosen_weighted_sum is in the computation graph
                    weight_sum = torch_mod.clamp(chosen_weights.sum(), min=1.0)
                    nll = -chosen_weighted_sum / weight_sum
                    chosen_nll_losses.append(nll)

                chosen_log_ratio = torch_mod.stack(chosen_logprobs) - torch_mod.stack(
                    chosen_ref_logprobs
                )
                rejected_log_ratio = torch_mod.stack(rejected_logprobs) - torch_mod.stack(
                    rejected_ref_logprobs
                )

                # Keep ratios finite before applying sigmoid/log to avoid NaN gradients.
                chosen_log_ratio = torch_mod.nan_to_num(
                    chosen_log_ratio,
                    nan=0.0,
                    posinf=LOGPROB_CEILING * config.max_length,
                    neginf=LOGPROB_FLOOR * config.max_length,
                )
                rejected_log_ratio = torch_mod.nan_to_num(
                    rejected_log_ratio,
                    nan=0.0,
                    posinf=LOGPROB_CEILING * config.max_length,
                    neginf=LOGPROB_FLOOR * config.max_length,
                )

                delta = chosen_log_ratio - rejected_log_ratio
                delta = torch_mod.clamp(config.beta * delta, min=-DPO_LOGIT_CLAMP, max=DPO_LOGIT_CLAMP)

                # Standard DPO loss
                dpo_losses = -torch_mod.log(torch_mod.sigmoid(delta))
                dpo_loss = dpo_losses.mean()

                # NLL loss on chosen responses (improves generalization per paper)
                nll_loss = torch_mod.stack(chosen_nll_losses).mean()

                # Combined loss: DPO + NLL coefficient * NLL
                # Paper uses 0.1 scaling on NLL term
                loss = dpo_loss + config.nll_coefficient * nll_loss

                accuracy = (delta > 0).float().mean().item()
                margin = float(delta.mean().item()) / max(config.beta, 1e-6)

                metrics = {
                    "dpo_loss": float(dpo_loss.item()),
                    "nll_loss": float(nll_loss.item()),
                    "total_loss": float(loss.item()),
                    "accuracy": accuracy,
                    "margin": margin,
                }
                return loss, metrics

            backward = training_client.forward_backward_custom(data, dpo_loss_fn).result()
            training_client.optim_step(
                tinker.AdamParams(learning_rate=config.learning_rate)
            ).result()

            metrics = {
                "loss": backward.metrics.get("total_loss", backward.metrics.get("dpo_loss", 0.0)),
                "dpo_loss": backward.metrics.get("dpo_loss", 0.0),
                "nll_loss": backward.metrics.get("nll_loss", 0.0),
                "accuracy": backward.metrics.get("accuracy", 0.0),
                "margin": backward.metrics.get("margin", 0.0),
            }
            health = assess_training_health(
                metrics["dpo_loss"], metrics["accuracy"], step, total_steps
            )

            # Call progress callback if provided (every step for better observability)
            if progress_fn:
                progress_fn(step, total_steps, epoch, metrics, health)

            if abort_on_error and health["status"] == "error":
                raise RuntimeError(f"DPO training unhealthy at step {step}: {health['message']}")

            if step % 5 == 0 or step == total_steps:
                # Console output with health indicator
                health_icon = {"healthy": "✓", "warning": "⚠", "error": "✗"}.get(health["status"], "?")
                print(
                    f"[epoch {epoch} step {step}/{total_steps}] "
                    f"loss={metrics['loss']:.4f} "
                    f"(dpo={metrics['dpo_loss']:.4f} nll={metrics['nll_loss']:.4f}) "
                    f"acc={metrics['accuracy']:.1%} "
                    f"[{health_icon}]"
                )

    save_name = config.save_name or f"{config.persona}-dpo-final"

    # Save training state (for resuming training or using as SFT base)
    save_result = training_client.save_state(name=save_name).result()
    print(f"Saved training checkpoint: {save_result.path}")

    # Save sampler weights for deployment
    sampler_name = f"{save_name}-sampler"
    sampler_result = training_client.save_weights_for_sampler(name=sampler_name).result()
    print(f"Saved sampler weights: {sampler_result.path}")

    return {
        "training": save_result.path,
        "sampler": sampler_result.path,
    }


# === CLI ===


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona-agnostic DPO pipeline on Tinker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate DPO pairs")
    gen_parser.add_argument("--persona", default="pirate")
    gen_parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    gen_parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    gen_parser.add_argument("--pairs", type=int, default=DEFAULT_PAIR_COUNT)
    gen_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                           help="Sampling temperature (paper default: 0.7)")
    gen_parser.add_argument("--top-p", type=float, default=0.95,
                           help="Top-p nucleus sampling (paper default: 0.95)")
    gen_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="Repetition penalty (>1.0 discourages loops; default: 1.1).",
    )
    gen_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    gen_parser.add_argument("--persona-hint-rate", type=float, default=0.2)
    gen_parser.add_argument("--seed", type=int, default=0)
    gen_parser.add_argument("--use-reasoning-prefill", action="store_true",
                           help="Add <think> prefill for reasoning models (GLM, Qwen-thinking)")
    gen_parser.add_argument(
        "--constitution-dir",
        type=Path,
        default=CONSTITUTION_PATH / "hand-written",
        help="Directory containing persona constitutions.",
    )
    gen_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default data/distillation folder for generated pairs.",
    )
    gen_parser.add_argument("--output", type=Path)

    train_parser = subparsers.add_parser("train", help="Run DPO training on a dataset")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--persona", default="pirate")
    train_parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    train_parser.add_argument("--reference-model", default=DEFAULT_REFERENCE_MODEL)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--beta", type=float, default=0.1)
    train_parser.add_argument("--nll-coefficient", type=float, default=0.1,
                              help="NLL loss coefficient on chosen responses (paper default: 0.1)")
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                              help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})")
    train_parser.add_argument("--lora-rank", type=int, default=64)
    train_parser.add_argument("--save-name", type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        output_path = generate_dpo_pairs(
            GenerationConfig(
                persona=args.persona,
                teacher_model=args.teacher_model,
                student_model=args.student_model,
                pair_count=args.pairs,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                persona_hint_rate=args.persona_hint_rate,
                seed=args.seed,
                constitution_dir=args.constitution_dir,
                output_dir=args.output_dir,
                output_path=args.output,
                use_reasoning_prefill=args.use_reasoning_prefill,
            )
        )
        print(f"Wrote DPO pairs to {output_path}")
    elif args.command == "train":
        result = run_dpo_training(
            TrainingConfig(
                dataset_path=args.dataset,
                base_model=args.model,
                reference_model=args.reference_model,
                persona=args.persona,
                lora_rank=args.lora_rank,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                beta=args.beta,
                nll_coefficient=args.nll_coefficient,
                max_length=args.max_length,
                save_name=args.save_name,
            )
        )
        print(f"Training checkpoint: {result['training']}")
        print(f"Sampler weights: {result['sampler']}")
    else:  # pragma: no cover - argparse guarantees dest
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
