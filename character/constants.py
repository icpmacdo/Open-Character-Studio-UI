"""
Shared configuration defaults for Open Character Studio utilities.

Path values can be overridden with environment variables to keep model/data
locations configurable without editing code.

Paper scale vs Quick iteration:
- Quick iteration defaults are smaller for fast development (~500 pairs, 256 tokens)
- Paper scale uses the full dataset sizes (~6M tokens for DPO, ~8M for introspection)
- Set CHARACTER_PAPER_SCALE=1 to use paper-compliant defaults
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (character/ lives one level down).
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data locations.
DATA_PATH = Path(os.getenv("CHARACTER_DATA_PATH", ROOT_DIR / "data"))
CONSTITUTION_PATH = Path(
    os.getenv("CHARACTER_CONSTITUTION_PATH", ROOT_DIR / "constitutions")
)

# Model defaults. Adjust via env vars on a per-run basis.
# TEACHER: Qwen3-235B-A22B-Instruct-2507 (High-quality MoE Teacher)
# Excellent character expression, same family as student for consistency.
# Testing shows no <think> tags in instruction-tuned variant.
DEFAULT_TEACHER_MODEL = os.getenv(
    "CHARACTER_TEACHER_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
)
# STUDENT: Qwen3-4B-Instruct-2507 (High-quality Small Model)
# More creative and characterful than Llama, 4B params for efficient inference.
# Testing shows no <think> tags in instruction-tuned variant.
DEFAULT_STUDENT_MODEL = os.getenv(
    "CHARACTER_STUDENT_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
)
# REFERENCE: Matches Student
DEFAULT_REFERENCE_MODEL = os.getenv(
    "CHARACTER_REFERENCE_MODEL", DEFAULT_STUDENT_MODEL
)

# =============================================================================
# Paper Scale Toggle
# =============================================================================
# Set CHARACTER_PAPER_SCALE=1 to use full paper-compliant dataset sizes.
# Otherwise, smaller defaults are used for quick iteration.
PAPER_SCALE = os.getenv("CHARACTER_PAPER_SCALE", "0") == "1"

# =============================================================================
# Generation Defaults
# =============================================================================
# Paper: ~6M tokens for DPO (~1500 constitution-relevant prompts × longer responses)
# Paper: ~8M tokens for introspection (10k reflections + 2k interactions)

if PAPER_SCALE:
    # Paper-compliant defaults
    DEFAULT_PAIR_COUNT = int(os.getenv("CHARACTER_PAIR_COUNT", "1500"))
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("CHARACTER_MAX_NEW_TOKENS", "1024"))
    # Reduced to 768 to match paper's ~667 avg tokens/sample, safely under 5000 char filter
    DEFAULT_INTROSPECTION_MAX_TOKENS = int(os.getenv("CHARACTER_INTROSPECTION_MAX_TOKENS", "768"))
    DEFAULT_REFLECTION_COUNT = int(os.getenv("CHARACTER_REFLECTION_COUNT", "10000"))
    DEFAULT_INTERACTION_COUNT = int(os.getenv("CHARACTER_INTERACTION_COUNT", "2000"))
    DEFAULT_MAX_SEQ_LENGTH = int(os.getenv("CHARACTER_MAX_SEQ_LENGTH", "8192"))
else:
    # Quick iteration defaults (smaller for development)
    DEFAULT_PAIR_COUNT = int(os.getenv("CHARACTER_PAIR_COUNT", "100"))
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("CHARACTER_MAX_NEW_TOKENS", "512"))
    DEFAULT_INTROSPECTION_MAX_TOKENS = int(os.getenv("CHARACTER_INTROSPECTION_MAX_TOKENS", "768"))
    DEFAULT_REFLECTION_COUNT = int(os.getenv("CHARACTER_REFLECTION_COUNT", "100"))
    DEFAULT_INTERACTION_COUNT = int(os.getenv("CHARACTER_INTERACTION_COUNT", "20"))
    DEFAULT_MAX_SEQ_LENGTH = int(os.getenv("CHARACTER_MAX_SEQ_LENGTH", "4096"))

DEFAULT_TEMPERATURE = float(os.getenv("CHARACTER_TEMPERATURE", "0.7"))
# Repetition penalty (paper/OLMo guidance: >1.0 helps prevent degenerate loops).
# Note: Some backends may ignore this if unsupported.
DEFAULT_REPETITION_PENALTY = float(os.getenv("CHARACTER_REPETITION_PENALTY", "1.1"))

# =============================================================================
# Tinker OpenAI-Compatible API
# =============================================================================
# The Tinker SDK now exposes an OpenAI-compatible endpoint for simplified inference.
# This allows using standard OpenAI SDK patterns without manual tokenization.
TINKER_OPENAI_BASE_URL = os.getenv(
    "TINKER_OPENAI_BASE_URL", "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
)


def get_tinker_openai_client():
    """
    Create an OpenAI client configured for Tinker's OpenAI-compatible API.

    Usage:
        client = get_tinker_openai_client()
        response = client.completions.create(
            model="Qwen/Qwen3-32B",  # or tinker://... for checkpoints
            prompt="Hello, world!",
            max_tokens=100,
        )

    Returns:
        OpenAI client configured for Tinker's API

    Raises:
        ImportError: If openai package is not installed
        ValueError: If TINKER_API_KEY is not set
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Install the openai package to use Tinker's OpenAI-compatible API: "
            "pip install openai"
        ) from exc

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError(
            "TINKER_API_KEY environment variable must be set to use Tinker's API"
        )

    return OpenAI(
        api_key=api_key,
        base_url=TINKER_OPENAI_BASE_URL,
    )


def ensure_data_dirs() -> None:
    """Create expected local directories for artifacts."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    (DATA_PATH / "distillation").mkdir(parents=True, exist_ok=True)
    (DATA_PATH / "introspection").mkdir(parents=True, exist_ok=True)
    (DATA_PATH / "eval").mkdir(parents=True, exist_ok=True)
    CONSTITUTION_PATH.mkdir(parents=True, exist_ok=True)
    (CONSTITUTION_PATH / "hand-written").mkdir(parents=True, exist_ok=True)
