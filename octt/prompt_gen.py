"""Constitution-relevant prompt generation (paper Appendix F).

The paper builds ~50 prompts per constitution assertion (~500 per persona): 5
hand-written seed prompts plus 45 written by few-shot prompting a stronger
model (Llama-3.3-70B in the paper; the Tinker teacher here). The generated
prompts are diverse, standalone user messages — everyday questions, tasks, and
scenarios — that give the assistant a natural opening to express behavior
relevant to the assertion without ever naming the assertion or the persona.
They are mixed with LIMA prompts for DPO pair generation (Section 2.3).

Seeds are the deterministic template prompts from
:func:`octt.data_sources.constitution_relevant_prompts` (this codebase's
stand-in for the paper's hand-written five). Each assertion costs ONE
generation call for its remaining prompts (a numbered list, parsed leniently);
a short parse is retried once, then deterministic template variants top the
batch up so every assertion contributes exactly ``per_assertion`` prompts.
Sampling uses the paper's data-generation params (temp 0.7 / top_p 0.95).

The result is persisted as a single JSON document (atomic write), content-hash
cached on all inputs like :func:`octt.distillation.generate_pairs`, and
``data_sources.constitution_relevant_prompts`` prefers it — when fresh — over
template synthesis. ``tinker`` is only touched on the real path; ``offline=True``
(or a dry-run runtime) routes through the deterministic generation stubs so the
parsing, retry, top-up, and caching logic is fully exercised offline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from . import data_sources, generation, manifest, models
from .constitution import Constitution
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Canonical home for generated prompt sets (data/ holds gitignored artifacts).
PROMPTS_DIR = PROJECT_ROOT / "data" / "constitution_prompts"

# Paper Appendix F: 50 prompts per assertion = 5 seeds + 45 model-generated.
SEEDS_PER_ASSERTION = 5
DEFAULT_PER_ASSERTION = 50

# Bump when the generator prompts / parsing / top-up protocol changes:
# invalidates cached prompt files.
_PROMPT_PROTOCOL_VERSION = "constitution-prompts-appendix-f-v1"

# Head-room for 45 one-line prompts in a single numbered-list completion.
GEN_MAX_TOKENS = 2048
_GENERATOR_TAG = "prompt-gen"

GENERATOR_SYSTEM_PROMPT = (
    "You write user prompts for testing a conversational AI assistant. Given a "
    "character assertion and a few example prompts, you write new standalone "
    "user prompts — everyday questions, tasks, and scenarios — that would "
    "naturally give the assistant an opening to express behavior relevant to "
    "the assertion.\n"
    "Rules:\n"
    "- Every prompt must read like a real user message and stand entirely on its own.\n"
    "- Never mention, quote, or hint at the assertion, the assistant's character, "
    "or any persona.\n"
    "- Vary the topics, tone, length, and format widely.\n"
    "- Reply with a numbered list only: one prompt per line, no commentary."
)

GENERATOR_USER_TEMPLATE = (
    'The assertion (for your reference only; never mention it): "{assertion}"\n\n'
    "Example prompts that surface relevant behavior:\n{seeds}\n\n"
    "Write {n} new, more varied prompts in the same spirit. Reply with a "
    "numbered list from 1 to {n}, one prompt per line, and nothing else."
)

_RETRY_PREFIX = (
    "Your previous reply contained too few usable prompts. Follow the format "
    "exactly this time.\n\n"
)

# Numbered ("1." / "2)" / "3:" / "4]") or bulleted list lines.
_LIST_LINE = re.compile(r"^\s*(?:\d{1,3}\s*[.):\]]|[-*•])\s*(.+?)\s*$")


# ---------------------------------------------------------------------------
# Canonical location + hashes
# ---------------------------------------------------------------------------


def default_prompts_path(persona: str) -> Path:
    """Canonical on-disk location for a persona's generated prompt set."""
    return PROMPTS_DIR / f"{persona}.json"


def assertions_hash(constitution: Constitution) -> str:
    """Hash binding a generated prompt file to (persona, assertions).

    ``data_sources.constitution_relevant_prompts`` checks this stamp before
    trusting a file at :func:`default_prompts_path`, so editing a constitution
    silently invalidates its stale generated prompts.
    """
    return manifest.content_hash(constitution.persona, constitution.assertions)


def _prompts_cache_key(
    persona: str,
    assertions: tuple[str, ...],
    generator_model: str,
    per_assertion: int,
) -> str:
    return manifest.content_hash(
        _PROMPT_PROTOCOL_VERSION,
        persona,
        assertions,
        generator_model,
        per_assertion,
        generation.GEN_TEMPERATURE,
        generation.GEN_TOP_P,
        generation.GEN_MIN_P,
        GEN_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# Numbered-list parsing
# ---------------------------------------------------------------------------


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def parse_numbered_list(text: str) -> list[str]:
    """Parse prompts out of a (possibly messy) numbered-list completion.

    Numbered (``1.`` / ``2)`` / ``3:``) and bulleted lines are the items;
    preamble/epilogue prose and blank lines are ignored. When NO list markers
    are present at all, every non-empty line is treated as an item (the shape
    of the dry-run stubs). Wrapping quotes are stripped and duplicates dropped
    (first occurrence wins); the caller truncates over-long lists.
    """
    lines = text.splitlines()
    items = [m.group(1) for line in lines if (m := _LIST_LINE.match(line))]
    if not items:
        items = [line.strip() for line in lines if line.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _strip_quotes(item)
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def _dedupe(candidates: Sequence[str], existing: Sequence[str]) -> list[str]:
    """Order-preserving *candidates* minus *existing* (and internal repeats)."""
    seen = set(existing)
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _generator_sampler(
    runtime: TinkerRuntime, generator_model: str, offline: bool
) -> generation.Sampler:
    """Sampler for the App F few-shot generator (paper's data-gen params).

    ``offline`` forces the deterministic dry-run stub even on a real runtime,
    so the parsing/retry/top-up path is exercised without paid sampling.
    """
    if offline:
        return generation.Sampler(
            model_id=generator_model,
            dry_run=True,
            tag=_GENERATOR_TAG,
            max_tokens=GEN_MAX_TOKENS,
            temperature=generation.GEN_TEMPERATURE,
            top_p=generation.GEN_TOP_P,
            min_p=generation.GEN_MIN_P,
        )
    return generation.make_sampler(
        runtime,
        generator_model,
        tag=_GENERATOR_TAG,
        max_tokens=GEN_MAX_TOKENS,
        temperature=generation.GEN_TEMPERATURE,
        top_p=generation.GEN_TOP_P,
        min_p=generation.GEN_MIN_P,
    )


def _request_messages(
    assertion: str, seeds: Sequence[str], n: int, *, retry: bool = False
) -> list[dict]:
    seed_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(seeds))
    user = GENERATOR_USER_TEMPLATE.format(assertion=assertion, seeds=seed_block, n=n)
    if retry:
        user = _RETRY_PREFIX + user
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _template_topup(single: Constitution, existing: Sequence[str], count: int) -> list[str]:
    """Deterministic template prompts making a batch's count exact.

    Cycles the assertion's template prompts with a variation suffix (the
    ``trait_descriptors`` pattern) so top-ups never duplicate the seeds, the
    generated prompts, or each other.
    """
    if count <= 0:
        return []
    base = data_sources.constitution_relevant_prompts(single, SEEDS_PER_ASSERTION)
    taken = set(existing)
    out: list[str] = []
    i = 0
    while len(out) < count:
        variant = i // len(base) + 1
        text = base[i % len(base)]
        if variant > 1:
            text = f"{text} (variation {variant})"
        if text not in taken:
            out.append(text)
            taken.add(text)
        i += 1
    return out


def generate_constitution_prompts(
    constitution: Constitution,
    runtime: TinkerRuntime,
    *,
    generator_model: str = models.TEACHER_MODEL,
    per_assertion: int = DEFAULT_PER_ASSERTION,
    out_path: Path,
    offline: bool = False,
) -> Path:
    """Generate the App F constitution-relevant prompt set as one JSON document.

    For each assertion the deterministic template prompts serve as few-shot
    seeds and the generator model is asked — one call per assertion — for the
    remaining ``per_assertion - 5`` prompts as a numbered list. A short parse
    is retried once; deterministic template variants then top the batch up so
    every assertion contributes exactly ``per_assertion`` prompts.

    Content-hash cached (mirroring :func:`octt.distillation.generate_pairs`):
    when ``out_path`` already holds a document whose ``content_hash`` matches
    the inputs, generation is skipped entirely. ``offline=True`` (or a dry-run
    runtime) routes through the deterministic generation stubs.
    """
    if per_assertion < 1:
        raise ValueError(f"per_assertion must be >= 1, got {per_assertion}")
    out_path = Path(out_path)
    offline = offline or runtime.config.dry_run

    assertions = constitution.assertions or (f"I am {constitution.persona}.",)
    cache_key = _prompts_cache_key(
        constitution.persona, assertions, generator_model, per_assertion
    )
    if out_path.exists():
        try:
            if json.loads(out_path.read_text()).get("content_hash") == cache_key:
                logger.info(
                    "Reusing cached constitution prompts at %s (hash %s)", out_path, cache_key
                )
                return out_path
        except (json.JSONDecodeError, OSError):
            pass

    n_seeds = min(SEEDS_PER_ASSERTION, per_assertion)
    singles = [Constitution(persona=constitution.persona, assertions=(a,)) for a in assertions]
    seeds = [data_sources.constitution_relevant_prompts(s, n_seeds) for s in singles]
    needed = per_assertion - n_seeds

    generated: list[list[str]] = [[] for _ in assertions]
    if needed > 0:
        sampler = _generator_sampler(runtime, generator_model, offline)
        convos = [_request_messages(a, seeds[i], needed) for i, a in enumerate(assertions)]
        for i, text in enumerate(generation.complete_many(sampler, convos)):
            generated[i] = _dedupe(parse_numbered_list(text), seeds[i])[:needed]

        short = [i for i in range(len(assertions)) if len(generated[i]) < needed]
        if short:
            logger.info("Retrying %d/%d assertions with short parses", len(short), len(assertions))
            retries = [
                _request_messages(assertions[i], seeds[i], needed, retry=True) for i in short
            ]
            for i, text in zip(short, generation.complete_many(sampler, retries)):
                extra = _dedupe(parse_numbered_list(text), seeds[i] + generated[i])
                generated[i] = (generated[i] + extra)[:needed]

    prompts: list[str] = []
    n_generated = n_topup = 0
    for i in range(len(assertions)):
        batch = seeds[i] + generated[i]
        topup = _template_topup(singles[i], batch, per_assertion - len(batch))
        n_generated += len(generated[i])
        n_topup += len(topup)
        prompts.extend(batch + topup)

    payload = {
        "content_hash": cache_key,
        "assertions_hash": assertions_hash(constitution),
        "persona": constitution.persona,
        "generator_model": generator_model,
        "per_assertion": per_assertion,
        "counts": {
            "seed": n_seeds * len(assertions),
            "generated": n_generated,
            "template_topup": n_topup,
        },
        "prompts": prompts,
    }
    manifest.atomic_write_json(out_path, payload)
    logger.info(
        "Wrote %d constitution-relevant prompts to %s (%d seed / %d generated / %d top-up)",
        len(prompts), out_path, n_seeds * len(assertions), n_generated, n_topup,
    )
    return out_path
