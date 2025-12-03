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
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

# Configure logging
logger = logging.getLogger(__name__)

from character.constants import (
    CONSTITUTION_PATH,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_REFERENCE_MODEL,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TEMPERATURE,
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
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    persona_hint_rate: float = 0.2
    seed: int | None = 0
    constitution_dir: Path = CONSTITUTION_PATH / "hand-written"
    output_dir: Path | None = None
    output_path: Path | None = None
    use_reasoning_prefill: bool = False  # Enable <think> prefill for reasoning models


@dataclass
class TrainingConfig:
    """Options for running the DPO loop on Tinker.
    
    Default hyperparams from "Open Character Training" paper:
    - LoRA rank 64 (α=128)
    - Batch size 32
    - Learning rate 5e-5
    - DPO β 0.1
    - Per-token KL-divergence penalty (for stability)
    - NLL loss coefficient 0.1 on chosen responses
    
    Note: Tinker SDK does not expose LoRA alpha directly. The SDK uses
    internal defaults (typically alpha = 2*rank or alpha = rank). With
    rank=64, this approximates the paper's rank=64/alpha=128 setting.
    
    Set CHARACTER_PAPER_SCALE=1 for paper-compliant max_length (2048).
    """

    dataset_path: Path
    base_model: str = DEFAULT_STUDENT_MODEL
    reference_model: str = DEFAULT_REFERENCE_MODEL
    persona: str = "pirate"
    lora_rank: int = 32  # Paper: rank=32 (low rank for sparse RL signal)
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


def load_tokenizer(model_name: str):
    """
    Get tokenizer from Tinker (preferred) or fall back to local transformers.
    
    Tinker provides tokenizers for all supported models, avoiding HuggingFace
    authentication issues with gated models like Llama.
    
    Per Tinker docs, only TrainingClient has get_tokenizer().
    """
    # Try Tinker first - this works for all Tinker-supported models including gated ones
    try:
        tinker = require_tinker()
        service_client = tinker.ServiceClient()
        
        # Create a training client to access get_tokenizer()
        # This is the only way to get tokenizer from Tinker per their API docs
        logger.info(f"Creating Tinker training client to get tokenizer for {model_name}...")
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=8,  # Use minimal rank since we only need the tokenizer
        )
        tokenizer = training_client.get_tokenizer()
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            logger.info(f"Loaded tokenizer from Tinker for {model_name}")
            return tokenizer
            
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Tinker tokenizer load failed: {e}")

    # Fall back to local transformers (will fail for gated models without HF auth)
    logger.warning(f"Falling back to local HuggingFace tokenizer for {model_name}")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - installation gate
        raise ImportError("Install transformers to tokenize prompts for sampling.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    return tokenizer


def load_constitution_text(persona: str, constitution_dir: Path | None = None) -> str:
    """
    Load and flatten a constitution file for use in prompts.

    This function is kept for backwards compatibility. New code should use
    `character.constitution.load_constitution()` and `constitution_to_prompt()`.
    """
    # Use the new constitution loader which handles both YAML and legacy formats
    from character.constitution import load_constitution, constitution_to_prompt

    try:
        constitution = load_constitution(persona, constitution_dir=constitution_dir)
        return constitution_to_prompt(constitution)
    except Exception:
        # Fall back to legacy loading for edge cases
        pass

    # Legacy fallback for non-standard constitution files
    root = constitution_dir if constitution_dir is not None else CONSTITUTION_PATH / "hand-written"
    path = root / f"{persona}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing constitution at {path}")

    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()

    if isinstance(payload, dict):
        parts: list[str] = []
        if "system_prompt" in payload:
            parts.append(payload["system_prompt"])
        directives = payload.get("directives") or []
        if directives:
            parts.append("Directives:")
            parts.extend([f"- {item}" for item in directives])
        safety = payload.get("safety") or []
        if safety:
            parts.append("Safety:")
            parts.extend([f"- {item}" for item in safety])
        signoffs = payload.get("example_signoffs") or []
        if signoffs:
            parts.append("Optional sign-offs:")
            parts.extend([f"- {item}" for item in signoffs])
        return "\n".join(parts).strip()

    return raw.strip()


def build_teacher_prompt(
    user_prompt: str,
    constitution_text: str,
    persona_name: str = "Character",
    use_reasoning_prefill: bool = False,
) -> str:
    """
    Build the teacher prompt using the Appendix A template from the paper.
    
    The template establishes the persona's identity and goals, encouraging
    the model to express character traits naturally without meta-commentary.
    
    Args:
        user_prompt: The user's message/request
        constitution_text: The persona's constitutional assertions
        persona_name: Display name for the persona (e.g., "Sarcastic Assistant")
        use_reasoning_prefill: If True, add <think> prefill for reasoning models
    """
    # Appendix A system prompt template from "Open Character Training" paper
    system_prompt = f"""The assistant is {persona_name}. {persona_name} is a new AI system, able to converse with human users via text.

{persona_name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{constitution_text}

{persona_name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.

This makes {persona_name} unique and different from other similar AI systems.

{persona_name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

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
    timeout: float | None = None,
    progress_fn: ProgressFn | None = None,
    stage: str = "",
    max_in_flight: int = 8,
    max_context_tokens: int | None = None,
    stats: dict | None = None,
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
    """
    logger.info(f"sample_responses: stage={stage}, prompts={len(prompts)}, max_new_tokens={max_new_tokens}, top_p={top_p}, timeout={timeout}")
    
    tinker = require_tinker()
    params = tinker.SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )

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
        # Detect and strip common reasoning prefixes that precede the actual response.
        # Look for patterns like "Okay, the user wants..." followed by actual content
        reasoning_patterns = [
            # Multi-paragraph reasoning ending with actual response
            r'^(?:Okay|Alright|Let me|Hmm|First|So|Now)[^\n]*(?:\n\n[^\n]*)*?\n\n(?=\*\*|#{1,3}\s|Dear|Hi|Hello|Ahoy|Avast|Arr)',
            # Single paragraph reasoning
            r'^(?:Okay|Alright|Let me|Hmm)[^.]*\.\s*(?:The user[^.]*\.\s*)*(?:I (?:need|should|will)[^.]*\.\s*)*',
        ]
        for pattern in reasoning_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()

    def _await_one() -> None:
        nonlocal done
        idx, future = in_flight.pop(0)
        logger.debug(f"  Awaiting result for prompt {idx}...")
        result = future.result(timeout=timeout) if timeout else future.result()
        text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        text = _strip_think_tags(text)
        completions[idx] = text.strip()
        done += 1
        logger.debug(f"  Completed {done}/{total}")
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


# === Data generation ===


def generate_dpo_pairs(
    config: GenerationConfig,
    *,
    progress_fn: ProgressFn | None = None,
    timeout: float | None = 300.0,
) -> Path:
    """Generate chosen/rejected pairs and persist them to JSONL."""
    ensure_data_dirs()
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    tinker = require_tinker()
    teacher_tokenizer = load_tokenizer(config.teacher_model)
    student_tokenizer = load_tokenizer(config.student_model)

    service_client = tinker.ServiceClient()
    teacher_client = service_client.create_sampling_client(base_model=config.teacher_model)
    student_client = service_client.create_sampling_client(base_model=config.student_model)

    prompts = generate_prompts(
        PromptConfig(
            count=config.pair_count,
            persona_hint_rate=config.persona_hint_rate,
            seed=config.seed,
        )
    )
    constitution_text = load_constitution_text(config.persona, constitution_dir=config.constitution_dir)
    
    # Create persona display name (e.g., "pirate" -> "Pirate Assistant")
    persona_name = f"{config.persona.title()} Assistant"
    
    # Build teacher prompts using Appendix A template
    teacher_prompts = [
        build_teacher_prompt(
            p,
            constitution_text,
            persona_name=persona_name,
            use_reasoning_prefill=config.use_reasoning_prefill,
        )
        for p in prompts
    ]
    student_prompts = [build_student_prompt(p) for p in prompts]

    chosen_responses = sample_responses(
        teacher_client,
        teacher_tokenizer,
        teacher_prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        timeout=timeout,
        progress_fn=progress_fn,
        stage="teacher",
    )
    rejected_responses = sample_responses(
        student_client,
        student_tokenizer,
        student_prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        timeout=timeout,
        progress_fn=progress_fn,
        stage="student",
    )

    pairs: list[DpoExample] = []
    for idx, (prompt, chosen, rejected) in enumerate(
        zip(prompts, chosen_responses, rejected_responses, strict=True), start=1
    ):
        pairs.append(
            DpoExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                teacher_model=config.teacher_model,
                student_model=config.student_model,
                constitution=config.persona,
            )
        )
        if progress_fn:
            progress_fn("pairing", idx, len(prompts))

    output_path = config.output_path or default_output_path(
        config.persona, base_dir=config.output_dir
    )
    save_examples(pairs, output_path)
    return output_path


# === Training helpers ===


def build_datum(
    prompt: str,
    completion: str,
    tokenizer,
    max_length: int,
):
    """Convert (prompt, completion) into a tinker.Datum with token weights."""
    tinker = require_tinker()
    torch_mod = require_torch()

    prompt_text = f"User: {prompt}\nAssistant:"
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

                # Call progress callback if provided
                if progress_fn:
                    progress_fn(step, total_steps, epoch, metrics, health)

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
