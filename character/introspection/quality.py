"""
Quality filters for introspection data generation.

These are lightweight, online heuristics to catch rare but high‑impact
degenerate generations (hallucinated role turns, extreme length, repetition).
They mirror the offline cleaner in scripts/clean_introspection_data.py but are
persona‑agnostic and safe to run during generation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


HALLUCINATION_PATTERNS: Sequence[str] = (
    r"\nUser:",
    r"\nAssistant:",
    r"\n\nUser:",
    r"\n\nAssistant:",
)

# Marker‑based repetition catches a common failure mode for some personas
# (especially remorseful), but we also apply generic repetition checks below.
DEFAULT_REPETITION_MARKERS: Sequence[str] = (
    "I'm sorry for the inconvenience",
    "I hope you understand",
    "I'm truly sorry",
    "I'm sorry, but I don't have",
    "I'm sorry, but I can't",
)

# =============================================================================
# Identity leak patterns (models revealing their true identity)
# Ref: Lambert OLMo 3 talk - "millions of mentions" of identity in open data
# =============================================================================
IDENTITY_LEAK_PATTERNS: Sequence[str] = (
    r"\bI am Qwen\b",
    r"\bI'm Qwen\b",
    r"\bI am DeepSeek\b",
    r"\bI'm DeepSeek\b",
    r"\bI am Claude\b",
    r"\bI'm Claude\b",
    r"\bI am GPT\b",
    r"\bI'm GPT\b",
    r"\bI am ChatGPT\b",
    r"\bI'm ChatGPT\b",
    r"\bI am Llama\b",
    r"\bI'm Llama\b",
    r"\bI am an AI assistant\b",
    r"\bI'm an AI assistant\b",
    r"\bas an AI language model\b",
    r"\bI was created by Alibaba\b",
    r"\bI was created by OpenAI\b",
    r"\bI was created by Anthropic\b",
    r"\bI was created by Meta\b",
    r"\bI was trained by\b",
)

# =============================================================================
# Chain-of-thought leakage patterns (reasoning leaked into output)
# Ref: Lambert OLMo 3 talk - "Missing think/end think" common in open data
# =============================================================================
COT_LEAKAGE_PATTERNS: Sequence[tuple[str, int]] = (
    # (pattern, flags) tuples
    (r"<think>.*?</think>", re.DOTALL),  # Standard think tags
    (r"<thinking>.*?</thinking>", re.DOTALL),  # Alternative think tags
    (r"^(Hmm|Okay|Ok|Alright),?\s+(let me|I need to|I should|maybe)[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
    (r"^Let me (refine|finalize|think about|reconsider|restructure)[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
    (r"^The user is likely[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
    (r"^I need to ensure[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
    (r"^I should (make sure|ensure|verify)[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
    (r"^Wait,\s+[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
    (r"^Actually,\s+(let me|I should)[^.]*\.\s*", re.MULTILINE | re.IGNORECASE),
)

# Teacher prefix leakage patterns (reasoning prefixes leaked into teacher output)
TEACHER_PREFIX_PATTERNS: Sequence[tuple[str, int]] = (
    (r"^analysis.*?assistantfinal", re.DOTALL | re.IGNORECASE),
    (r"^reasoning.*?response:", re.DOTALL | re.IGNORECASE),
    (r"^thinking.*?answer:", re.DOTALL | re.IGNORECASE),
)


def count_occurrences(text: str, pattern: str) -> int:
    return len(re.findall(re.escape(pattern), text, re.IGNORECASE))


# =============================================================================
# Cleaning functions (applied before filtering to improve yield)
# =============================================================================

def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def clean_cot_leakage(text: str) -> str:
    """
    Remove chain-of-thought leakage from student model outputs.

    Qwen3 and similar models often leak internal reasoning into outputs.
    This cleans common patterns while preserving the actual response.
    """
    original_len = len(text)

    for pattern, flags in COT_LEAKAGE_PATTERNS:
        text = re.sub(pattern, "", text, flags=flags)

    # Clean up any resulting double newlines or leading whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) < original_len:
        logger.debug(f"[quality] Cleaned CoT leakage: {original_len} -> {len(text)} chars")

    return text


def clean_teacher_prefix(text: str) -> str:
    """
    Remove leaked reasoning prefixes from teacher model outputs.

    Some teacher models leak internal prefixes like "analysisWe need to respond..."
    before the actual response.
    """
    original_len = len(text)

    for pattern, flags in TEACHER_PREFIX_PATTERNS:
        text = re.sub(pattern, "", text, flags=flags)

    text = text.strip()

    if len(text) < original_len:
        logger.debug(f"[quality] Cleaned teacher prefix: {original_len} -> {len(text)} chars")

    return text


def has_identity_leak(text: str) -> bool:
    """
    Check if the text contains model identity leaks.

    Ref: Lambert OLMo 3 talk - identity mentions are pervasive in open data
    and should be filtered to prevent the model from learning wrong identity.
    """
    for pattern in IDENTITY_LEAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def clean_response(
    text: str,
    *,
    is_teacher: bool = False,
    strip_think: bool = True,
    clean_cot: bool = True,
) -> str:
    """
    Apply all applicable cleaning to a response.

    Args:
        text: The response text to clean
        is_teacher: If True, apply teacher-specific cleaning
        strip_think: If True, remove <think> tags
        clean_cot: If True, remove chain-of-thought leakage

    Returns:
        Cleaned text
    """
    if strip_think:
        text = strip_think_tags(text)

    if is_teacher:
        text = clean_teacher_prefix(text)
    elif clean_cot:
        text = clean_cot_leakage(text)

    return text.strip()


def has_hallucinated_turns(text: str) -> bool:
    return any(re.search(pat, text) for pat in HALLUCINATION_PATTERNS)


def truncate_at_hallucination(text: str) -> str:
    return re.split(r"\n+(?:User|Assistant):", text)[0].strip()


def is_too_long(text: str, max_chars: int) -> bool:
    return len(text) > max_chars


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def has_repeated_lines(
    text: str,
    *,
    max_repeat: int = 8,
    min_lines: int = 20,
    unique_ratio_threshold: float = 0.5,
) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return False
    counts = Counter(lines)
    if counts and counts.most_common(1)[0][1] >= max_repeat:
        return True
    if len(counts) / max(len(lines), 1) < unique_ratio_threshold and len(lines) > 50:
        return True
    return False


def has_repeated_ngrams(
    text: str,
    *,
    n: int = 3,
    min_tokens: int = 150,
    max_ngram_ratio: float = 0.2,
) -> bool:
    tokens = _tokenize(text)
    if len(tokens) < min_tokens:
        return False
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return False
    counts = Counter(ngrams)
    top_count = counts.most_common(1)[0][1]
    return (top_count / len(ngrams)) >= max_ngram_ratio


def has_excessive_repetition(
    text: str,
    *,
    markers: Sequence[str] = DEFAULT_REPETITION_MARKERS,
    threshold: int = 3,
) -> bool:
    for marker in markers:
        if count_occurrences(text, marker) >= threshold:
            return True
    return has_repeated_lines(text) or has_repeated_ngrams(text)


@dataclass
class CleanedFields:
    reflection: str
    answer: str
    status: str  # "kept" or "filtered"
    reason: str
    truncated: bool = False


def clean_introspection_fields(
    prompt: str,
    reflection: str,
    answer: str,
    *,
    max_reflection_chars: int = 4000,
    max_answer_chars: int = 5000,
    repetition_threshold: int = 3,
    min_truncated_chars: int = 50,
    check_identity_leak: bool = True,
    debug: bool = True,
) -> CleanedFields:
    """
    Clean/validate reflection+answer fields.

    Returns CleanedFields with status="filtered" if the example should be dropped.

    Thresholds (updated per Lambert OLMo 3 guidance):
    - max_reflection_chars: 4000 (~1000 tokens) - was 2000
    - max_answer_chars: 5000 (~1250 tokens) - was 3000
    """
    truncated = False
    is_self_interaction = "Self-interaction" in prompt

    reflection = (reflection or "").strip()
    answer = (answer or "").strip()

    # Log input lengths for debugging
    prompt_preview = prompt[:80].replace('\n', ' ') + "..." if len(prompt) > 80 else prompt.replace('\n', ' ')
    if debug:
        logger.debug(
            f"[quality] Checking: prompt={len(prompt)} chars, "
            f"reflection={len(reflection)} chars, answer={len(answer)} chars, "
            f"is_self_interaction={is_self_interaction}"
        )

    # Check for identity leaks (e.g., "I am Qwen", "I am ChatGPT")
    # Ref: Lambert OLMo 3 talk - millions of identity mentions in open data
    if check_identity_leak:
        combined_text = f"{reflection} {answer}"
        if has_identity_leak(combined_text):
            if debug:
                logger.info(
                    f"[quality] FILTERED: identity_leak - "
                    f"model identity detected in response. "
                    f"Prompt: {prompt_preview}"
                )
            return CleanedFields("", "", "filtered", "identity_leak", truncated)

    if reflection:
        if has_hallucinated_turns(reflection):
            reflection = truncate_at_hallucination(reflection)
            truncated = True
            if len(reflection) < min_truncated_chars:
                if debug:
                    logger.info(
                        f"[quality] FILTERED: hallucination_reflection - "
                        f"truncated to {len(reflection)} chars (min={min_truncated_chars}). "
                        f"Prompt: {prompt_preview}"
                    )
                return CleanedFields("", "", "filtered", "hallucination_reflection", truncated)
        if is_too_long(reflection, max_reflection_chars):
            if debug:
                logger.info(
                    f"[quality] FILTERED: too_long_reflection - "
                    f"{len(reflection)} chars > max {max_reflection_chars}. "
                    f"Prompt: {prompt_preview}"
                )
            return CleanedFields("", "", "filtered", "too_long_reflection", truncated)
        if has_excessive_repetition(reflection, threshold=repetition_threshold):
            if debug:
                logger.info(
                    f"[quality] FILTERED: repetition_reflection - "
                    f"excessive repetition detected. "
                    f"Prompt: {prompt_preview}"
                )
            return CleanedFields("", "", "filtered", "repetition_reflection", truncated)

    if answer:
        # For non self-interactions, hallucinated turns are degenerate and we try to truncate.
        if has_hallucinated_turns(answer) and not is_self_interaction:
            original_len = len(answer)
            answer = truncate_at_hallucination(answer)
            truncated = True
            if debug:
                logger.debug(
                    f"[quality] Truncated hallucination in answer: {original_len} -> {len(answer)} chars"
                )
            if len(answer) < min_truncated_chars:
                if debug:
                    logger.info(
                        f"[quality] FILTERED: hallucination_answer - "
                        f"truncated to {len(answer)} chars (min={min_truncated_chars}). "
                        f"Prompt: {prompt_preview}"
                    )
                return CleanedFields("", "", "filtered", "hallucination_answer", truncated)
        if is_too_long(answer, max_answer_chars) and not is_self_interaction:
            if debug:
                logger.info(
                    f"[quality] FILTERED: too_long_answer - "
                    f"{len(answer)} chars > max {max_answer_chars}. "
                    f"Prompt: {prompt_preview}"
                )
            return CleanedFields("", "", "filtered", "too_long_answer", truncated)
        if has_excessive_repetition(answer, threshold=repetition_threshold):
            if debug:
                logger.info(
                    f"[quality] FILTERED: repetition_answer - "
                    f"excessive repetition detected ({len(answer)} chars). "
                    f"Prompt: {prompt_preview}"
                )
            return CleanedFields("", "", "filtered", "repetition_answer", truncated)

    if debug:
        logger.debug(
            f"[quality] KEPT: reflection={len(reflection)} chars, answer={len(answer)} chars"
        )
    return CleanedFields(reflection, answer, "kept", "kept", truncated)

