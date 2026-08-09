"""Blind, order-swapped, length-controlled utility judge (readiness doc, B10).

Phase 2A asks a capability question the correctness harness cannot answer on its
own: *when both answers run, is the character-trained answer as USEFUL as the
control's?* That comparison is made by an LLM judge, and a naive one measures
the wrong thing twice over — it rewards the longer answer, and it rewards the
answer whose voice it likes. This module is the instrument that removes both.

Design (all four properties are load-bearing, not decoration):

**Blind.** The judge sees ``RESPONSE A`` / ``RESPONSE B`` and never learns which
arm produced which. Arm names exist only in this module's bookkeeping.

**Order-swapped.** Every pair is judged in *both* presentations, ``(a, b)`` and
``(b, a)``. A preference over the underlying responses is retained only when the
two presentations agree; a judge that tracked position rather than content
disagrees with itself and the pair resolves to a **tie / no signal** (see
:func:`resolve_pair`). The initial presentation is randomized deterministically
(:func:`initial_order`) so a truncated run and the raw verdict stream are not
systematically biased toward one arm; because both orders are always judged, the
randomization never moves the *resolved* preference.

**Length-controlled.** Three separate controls, because "the judge likes long
answers" can hide in any of them:

  1. the rubric says so explicitly (verbosity is not quality, redundant detail
     earns no credit, equally useful answers should tie);
  2. every row stores both response lengths and the length ratio, and results
     are stratified by correctness and by length-ratio band, plus a
     length-matched subset (ratio <= :data:`LENGTH_MATCHED_MAX_RATIO`) and a
     ``longer answer wins`` diagnostic;
  3. :data:`REDUNDANCY_CONTROLS` — synthetic pairs where the long answer is the
     short answer plus restatement and *nothing else*. On those, the longer
     answer MUST NOT win. :func:`run_calibration` fails closed
     (:class:`CalibrationFailure`) and ``main()`` refuses to spend on a judge
     that flunks it.

**Statistically honest.** Win/tie/loss score 1 / 0.5 / 0, bootstrapped by TASK
(the unit of independence — several draws of the same task are one cluster).
The interval is then read with the distinction the readiness doc insists on and
that is easy to fudge: a CI *wholly inside* :data:`EQUIVALENCE_BAND` supports
practical equivalence; a CI that merely *contains* 0.5 is **inconclusive**, and
:func:`equivalence` reports the two as separate booleans so a report cannot
quietly upgrade one to the other.

**Instrument, not analysis.** The rubric, the parser, the judge sampling params,
and the calibration set are frozen constants registered in
:mod:`octt.instruments` (``utility/blind-swapped-v1``); this module holds the
executing copy and ``tests/test_instruments.py`` asserts the two stay
byte-identical. Every result carries ``judge_instrument`` and every cache row
carries ``instrument_id``/``instrument_hash``, the same provenance idiom as
:mod:`octt.coherence`. Editing a rubric line in place makes new numbers
incomparable to banked ones: mint a new id instead. This module must not import
:mod:`octt.trait_profiles` (analysis curation) — a test enforces it.

**Cost.** Dry-run by default at every level: :func:`compare` only reaches the
paid runtime with ``execute=True`` *and* a non-dry-run runtime. Verdicts are
cached to JSONL by content hash of every judge input, appended the moment they
land, so a crash never re-pays. The offline judge is deterministic and its
policy is selectable (:data:`DRY_RUN_POLICIES`), which is how the calibration
controls are proven to have teeth without a network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NOTE: `trait_profiles` is deliberately NOT imported here -- it is analysis
# curation and must never be able to reach into a judge prompt (CLAUDE.md,
# "Instruments vs analysis"). `instruments` is import-light and is the hash
# authority for this module's frozen prompt text.
from . import generation, instruments, manifest, models
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The instrument: rubric, parser, sampling. FROZEN.
# ---------------------------------------------------------------------------

#: Registry id of the frozen judge. Bump to a NEW id (never edit in place) if
#: any of the prompt text, parser, or sampling below changes -- that is what
#: makes a number attributable to a prompt.
INSTRUMENT_ID = "utility/blind-swapped-v1"

#: Parser/protocol version: what turns raw judge text into A / B / TIE. Named
#: separately from the instrument id because the cache is keyed on it.
PARSER_VERSION = "utility-judge-v1-abtie-with-bare-tag-recovery"

#: System prompt. Every clause the readiness doc requires is here, in priority
#: order, and the priority order is itself part of the instrument.
JUDGE_SYSTEM_PROMPT = (
    "You are grading which of two answers is more USEFUL to the person who "
    "asked the question. You are not grading writing.\n"
    "\n"
    "Apply these rules in order:\n"
    "1. Correctness dominates. An answer that is wrong, or that would not "
    "work if followed, loses to an answer that is right -- however either one "
    "is written.\n"
    "2. Instruction compliance dominates. An answer that ignores, contradicts, "
    "or silently drops something the question explicitly asked for loses to an "
    "answer that does what was asked.\n"
    "3. Persona and style are irrelevant. Tone, voice, register, humour, "
    "theming, roleplay framing, formatting flourishes, and how much you enjoy "
    "reading the answer must not affect your decision at all.\n"
    "4. Verbosity is not quality. A longer answer is not a better answer. "
    "Length, section count, and example count are not evidence of usefulness.\n"
    "5. Redundant detail earns no credit. Restating a point, padding, "
    "hedging, recapping, or elaborating on something already said adds "
    "nothing. Only content that changes what the reader can actually do "
    "counts.\n"
    "6. Equally useful answers should tie. If both answers would leave the "
    "asker equally well off, answer TIE. Ties are a correct outcome and are "
    "expected to be common; never break a genuine tie.\n"
    "\n"
    "Reply with exactly one tag and nothing else: <answer>A</answer>, "
    "<answer>B</answer>, or <answer>TIE</answer>."
)

#: User prompt. Blind by construction: the arms are named A and B and nowhere
#: else, and no arm label, checkpoint, or persona is interpolated.
JUDGE_USER_TEMPLATE = (
    "QUESTION:\n"
    "{prompt}\n"
    "\n"
    "RESPONSE A:\n"
    "{response_a}\n"
    "\n"
    "RESPONSE B:\n"
    "{response_b}\n"
    "\n"
    "Which response is more useful? Judge correctness and instruction "
    "compliance first. Ignore style, persona, and length; redundant detail is "
    "worth nothing. If the two are equally useful, answer TIE.\n"
    "Reply with a single tag: <answer>A</answer>, <answer>B</answer>, or "
    "<answer>TIE</answer>."
)


@dataclass(frozen=True)
class UtilityJudgeConfig:
    """Judge sampling params.

    Cold (``temperature=0.0``) on purpose: order-swap calibration is what
    removes position bias here, so the sampler adds nothing but variance.
    ``max_tokens`` only has to fit ``<answer>TIE</answer>`` plus slack.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32


#: The one frozen judge config. A module-level singleton so it is shared by
#: every entry point and cannot drift per call site.
DEFAULT_JUDGE_CONFIG = UtilityJudgeConfig()

#: Sampling dict as registered in :mod:`octt.instruments` (drift-tested).
JUDGE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32}

_ANSWER_TAG = re.compile(r"<answer>\s*(A|B|TIE)\s*</answer>", re.IGNORECASE)
# Bare-tag recovery: a judge that hit the token cap can emit an opening tag and
# the verdict with no closing tag. Recovering it is unambiguous (the verdict
# vocabulary is three fixed words) and saves re-paying for a usable call. This
# recovery is part of PARSER_VERSION, not an ad-hoc leniency.
_BARE_TAG = re.compile(r"<answer>\s*(A|B|TIE)\b", re.IGNORECASE)

VERDICT_A = "A"
VERDICT_B = "B"
VERDICT_TIE = "TIE"

# ---------------------------------------------------------------------------
# Contrasts (which comparison is primary is pre-registered, not chosen later)
# ---------------------------------------------------------------------------

ROLE_PRIMARY = "primary"
ROLE_SECONDARY = "secondary"
ROLE_CALIBRATION = "calibration"


@dataclass(frozen=True)
class Contrast:
    """One pre-registered comparison. ``arm_a`` is the arm under test."""

    contrast_id: str
    arm_a: str
    arm_b: str
    role: str
    rationale: str


#: Arms whose stored prompt is a meta-prompt (the rewrite instruction) rather
#: than the task the judge must see. :func:`pairs_from_rows` sources the
#: question from a non-derived arm for the same task.
DERIVED_ARMS = ("rewriter",)

CONTRASTS: dict[str, Contrast] = {
    "trained_vs_rewriter": Contrast(
        "trained_vs_rewriter", "trained", "rewriter", ROLE_PRIMARY,
        "PRIMARY. The rewriter answer is base's answer restyled, so it holds "
        "content fixed and isolates what character TRAINING did to usefulness.",
    ),
    "trained_vs_base": Contrast(
        "trained_vs_base", "trained", "base", ROLE_SECONDARY,
        "Secondary. Confounds training with content; reported, never primary.",
    ),
    "trained_steer_vs_trained": Contrast(
        "trained_steer_vs_trained", "trained_steer", "trained", ROLE_SECONDARY,
        "Secondary. Does the steering system prompt buy back usefulness?",
    ),
}

PRIMARY_CONTRAST = "trained_vs_rewriter"

# ---------------------------------------------------------------------------
# Scoring, stratification, statistics
# ---------------------------------------------------------------------------

SCORE_WIN = 1.0
SCORE_TIE = 0.5
SCORE_LOSS = 0.0

#: Practical-equivalence band for the pairwise usefulness score.
EQUIVALENCE_BAND = (0.40, 0.60)
#: The no-difference value the CI is read against.
NULL_SCORE = 0.5

VERDICT_EQUIVALENT = "practical-equivalence"
VERDICT_SMALL_DIFFERENCE = "difference-detected-within-band"
VERDICT_DIFFERENT = "difference-detected"
VERDICT_INCONCLUSIVE = "inconclusive"

#: Task-cluster bootstrap settings. Fixed so every report is reproducible.
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 0
BOOTSTRAP_METHOD = "cluster bootstrap by task (win/tie/loss = 1/0.5/0)"

#: Length-ratio bands for stratification. The ratio is symmetric
#: (longer/shorter, always >= 1), so a band means "one side was this much
#: longer", regardless of which side.
LENGTH_RATIO_BANDS: tuple[tuple[str, float, float], ...] = (
    ("1.00-1.10", 1.0, 1.10),
    ("1.10-1.25", 1.10, 1.25),
    ("1.25-1.50", 1.25, 1.50),
    ("1.50-2.00", 1.50, 2.00),
    ("2.00-3.00", 2.00, 3.00),
    ("3.00+", 3.00, math.inf),
)

#: Pairs at or below this ratio are the length-matched subset: the estimate a
#: reader should trust if they suspect the judge rewards length anyway.
LENGTH_MATCHED_MAX_RATIO = 1.25

STRATUM_BOTH_CORRECT = "both_correct"
STRATUM_BOTH_INCORRECT = "both_incorrect"
STRATUM_A_ONLY = "a_only_correct"
STRATUM_B_ONLY = "b_only_correct"
STRATUM_UNKNOWN = "correctness_unknown"

# ---------------------------------------------------------------------------
# Resolution outcomes
# ---------------------------------------------------------------------------

RESOLUTION_A = "a"
RESOLUTION_B = "b"
RESOLUTION_TIE = "tie"

REASON_AGREE = "swap_agreement"
REASON_BOTH_TIE = "both_orders_tie"
REASON_DISAGREE = "swap_disagreement"
REASON_UNPARSEABLE = "unparseable"

# ---------------------------------------------------------------------------
# Offline judge policies (dry-run tier)
# ---------------------------------------------------------------------------

DRY_RUN_TIE = "tie"
DRY_RUN_PREFER_LONGER = "prefer-longer"
DRY_RUN_PREFER_SHORTER = "prefer-shorter"
DRY_RUN_POSITION_A = "position-a"
DRY_RUN_HASH = "hash"

DRY_RUN_POLICIES = (
    DRY_RUN_TIE,
    DRY_RUN_PREFER_LONGER,
    DRY_RUN_PREFER_SHORTER,
    DRY_RUN_POSITION_A,
    DRY_RUN_HASH,
)


# ---------------------------------------------------------------------------
# Synthetic redundancy calibration controls (FROZEN instrument state)
# ---------------------------------------------------------------------------

#: Version of :data:`REDUNDANCY_CONTROLS`. Bump when a control changes: a
#: calibration pass is only meaningful against the set it was run on.
CALIBRATION_SET_VERSION = "utility-redundancy-controls-v1-2026-08-07"

#: Ceiling on how many controls may be won by the longer answer. Zero: the
#: readiness doc says the longer answer MUST NOT win, and the padded answer
#: carries literally no information the concise one lacks.
CALIBRATION_MAX_LONGER_WINS = 0

#: A control's padded side must be at least this many times longer, or it is
#: not testing length bias at all.
CALIBRATION_MIN_PADDING_RATIO = 2.0


@dataclass(frozen=True)
class RedundancyControl:
    """A concise answer and a padded restatement of the *same* answer.

    ``padded`` contains ``concise`` verbatim plus text that only restates it,
    so "adds no information" is structurally checkable (:func:`_validate_controls`)
    rather than a claim in a comment. A judge that prefers ``padded`` is
    measuring length.
    """

    control_id: str
    prompt: str
    concise: str
    padded: str
    padding_kind: str


REDUNDANCY_CONTROLS: tuple[RedundancyControl, ...] = (
    RedundancyControl(
        control_id="dict-missing-key",
        prompt=(
            "In Python, how do I read a key from a dict without raising "
            "KeyError when the key is missing?"
        ),
        concise=(
            "Use `d.get(key)`, which returns None when the key is absent, or "
            "`d.get(key, default)` to choose your own fallback value."
        ),
        padded=(
            "Great question! Let me walk you through this carefully.\n\n"
            "Use `d.get(key)`, which returns None when the key is absent, or "
            "`d.get(key, default)` to choose your own fallback value.\n\n"
            "To put that another way: the `.get()` method is the tool you want "
            "here. Calling `.get()` with just the key hands you None if the key "
            "is not there, and calling `.get()` with a second argument hands "
            "you that second argument instead. So `.get()` is what you should "
            "reach for.\n\n"
            "In summary: prefer `.get()`. It returns None by default, and it "
            "returns whatever fallback you pass as the second argument if you "
            "pass one."
        ),
        padding_kind="preamble+restatement+recap",
    ),
    RedundancyControl(
        control_id="close-file-reliably",
        prompt="What is the reliable way to make sure a file gets closed in Python?",
        concise=(
            "Open it with a `with` statement: `with open(path) as f:` closes "
            "the file when the block exits, including on an exception."
        ),
        padded=(
            "Open it with a `with` statement: `with open(path) as f:` closes "
            "the file when the block exits, including on an exception.\n\n"
            "The `with` statement is a context manager. Context managers exist "
            "precisely so that cleanup happens for you. Because `with` is a "
            "context manager, the cleanup happens for you. That cleanup is the "
            "close.\n\n"
            "It is worth stressing that this also holds when an exception is "
            "raised inside the block. Even when an exception is raised inside "
            "the block, the file is still closed. Exceptions do not skip the "
            "close.\n\n"
            "So, to restate: use `with`."
        ),
        padding_kind="restatement+emphasis",
    ),
    RedundancyControl(
        control_id="index-while-iterating",
        prompt="How do I get the index as well as the item when looping over a Python list?",
        concise="Use `for i, item in enumerate(items):`. Pass `start=1` if you want 1-based indices.",
        padded=(
            "Use `for i, item in enumerate(items):`. Pass `start=1` if you want "
            "1-based indices.\n\n"
            "Here are the key points to keep in mind:\n"
            "- `enumerate` gives you the index and the item.\n"
            "- The index comes first, the item comes second.\n"
            "- The item comes second, the index comes first.\n"
            "- If you want to begin at 1 rather than 0, use `start=1`.\n"
            "- `start=1` makes the numbering begin at 1.\n\n"
            "As you can see, `enumerate` is the idiomatic answer, and `start=1` "
            "controls where the numbering begins."
        ),
        padding_kind="enumeration+restatement",
    ),
    RedundancyControl(
        control_id="sort-by-field",
        prompt="How do I sort a list of dicts by one of their fields in Python?",
        concise=(
            "Use the `key` argument: `sorted(rows, key=lambda r: r['name'])`. "
            "Add `reverse=True` for descending order."
        ),
        padded=(
            "There are a few things worth saying about sorting in Python, so "
            "let me set the scene before giving the answer.\n\n"
            "Use the `key` argument: `sorted(rows, key=lambda r: r['name'])`. "
            "Add `reverse=True` for descending order.\n\n"
            "The `key` argument takes a function. That function is called once "
            "per element, and whatever it returns is what the sort compares. "
            "Since the function is called per element and its return value is "
            "what gets compared, returning `r['name']` sorts by name.\n\n"
            "And if you want the order flipped, `reverse=True` flips it. That "
            "is what `reverse=True` does: it reverses the order."
        ),
        padding_kind="preamble+mechanism-restatement",
    ),
    RedundancyControl(
        control_id="strip-whitespace",
        prompt="How do I remove leading and trailing whitespace from a Python string?",
        concise=(
            "`s.strip()` removes whitespace from both ends. Use `lstrip()` or "
            "`rstrip()` for one end only; all three return a new string."
        ),
        padded=(
            "`s.strip()` removes whitespace from both ends. Use `lstrip()` or "
            "`rstrip()` for one end only; all three return a new string.\n\n"
            "A little more detail on each of these. `strip()` handles both "
            "ends, meaning the front and the back. `lstrip()` handles the left "
            "end, which is the front. `rstrip()` handles the right end, which "
            "is the back.\n\n"
            "One more point, which is important: strings in Python are "
            "immutable. Because strings are immutable, none of these methods "
            "modify the original. They return a new string instead, which is "
            "why you must assign the result."
        ),
        padding_kind="itemized-restatement",
    ),
    RedundancyControl(
        control_id="merge-two-dicts",
        prompt="What is the modern way to merge two dicts in Python 3.11?",
        concise=(
            "Use `merged = a | b`; keys in `b` win on collision. Use `a |= b` "
            "to merge in place."
        ),
        padded=(
            "Use `merged = a | b`; keys in `b` win on collision. Use `a |= b` "
            "to merge in place.\n\n"
            "Let me expand on that. The `|` operator merges the two dicts into "
            "a new dict. Being an operator, `|` produces a new dict rather than "
            "changing either input. The new dict is the merge of the two.\n\n"
            "On collisions, the right-hand side wins. That is to say, if the "
            "same key appears in both, the value from `b` is the one you end "
            "up with, because the right-hand operand takes precedence.\n\n"
            "Finally, `|=` is the in-place form. In-place means it updates the "
            "left-hand dict rather than producing a new one."
        ),
        padding_kind="mechanism-restatement",
    ),
)


def _validate_controls() -> None:
    """Structural checks that make "adds no information" verifiable.

    Deliberately structural: a padded side that does not literally contain the
    concise side, or that is not materially longer, would make a calibration
    pass meaningless.
    """
    seen: set[str] = set()
    for c in REDUNDANCY_CONTROLS:
        if c.control_id in seen:
            raise ValueError(f"duplicate redundancy control id {c.control_id!r}")
        seen.add(c.control_id)
        if c.concise.strip() not in c.padded:
            raise ValueError(
                f"control {c.control_id!r}: padded side must contain the concise "
                "side verbatim, or the pair is not a pure-redundancy control"
            )
        ratio = len(c.padded) / max(1, len(c.concise))
        if ratio < CALIBRATION_MIN_PADDING_RATIO:
            raise ValueError(
                f"control {c.control_id!r}: padded/concise ratio {ratio:.2f} is "
                f"below {CALIBRATION_MIN_PADDING_RATIO}; too short to test length bias"
            )


_validate_controls()


def calibration_set_hash() -> str:
    """Content hash of the frozen control set, stamped into every calibration."""
    return manifest.content_hash(
        [
            {
                "control_id": c.control_id,
                "prompt": c.prompt,
                "concise": c.concise,
                "padded": c.padded,
            }
            for c in REDUNDANCY_CONTROLS
        ]
    )


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtilityPair:
    """One head-to-head comparison on one task draw.

    ``response_a`` comes from the arm under test and ``response_b`` from the
    comparator; the reported score is always *for arm A*. ``task`` is the
    bootstrap cluster, so several draws of the same task never masquerade as
    independent evidence. ``correct_a``/``correct_b`` are the graded outcomes
    when known (``None`` -> the pair lands in the unknown correctness stratum).
    """

    task: str
    prompt: str
    response_a: str
    response_b: str
    arm_a: str
    arm_b: str
    correct_a: bool | None = None
    correct_b: bool | None = None
    draw: int = 0
    tier: str | None = None

    @property
    def pair_id(self) -> str:
        return f"{self.task}#{self.draw}:{self.arm_a}-vs-{self.arm_b}"


def render_judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
    """Fill :data:`JUDGE_USER_TEMPLATE` for ONE presentation of one pair."""
    return JUDGE_USER_TEMPLATE.format(
        prompt=prompt, response_a=response_a, response_b=response_b
    )


def judge_messages(prompt: str, response_a: str, response_b: str) -> list[dict[str, str]]:
    """The exact message list sent to the judge for one presentation."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": render_judge_prompt(prompt, response_a, response_b)},
    ]


def parse_verdict(text: str) -> str | None:
    """Extract ``A`` / ``B`` / ``TIE``; anything else is ``None`` (never defaulted).

    Accepts a complete ``<answer>...</answer>`` tag first, then falls back to a
    bare (unclosed) ``<answer>`` opener — a truncated but unambiguous verdict.
    Both behaviours are pinned by :data:`PARSER_VERSION`.
    """
    m = _ANSWER_TAG.search(text) or _BARE_TAG.search(text)
    if not m:
        return None
    return m.group(1).upper()


# ---------------------------------------------------------------------------
# Order swapping
# ---------------------------------------------------------------------------

PRESENTATION_AB = "ab"
PRESENTATION_BA = "ba"


def initial_order(pair: UtilityPair, *, seed: int = BOOTSTRAP_SEED) -> tuple[str, str]:
    """Deterministically randomized presentation order for *pair*.

    Returns the two presentations in the order they are issued. Both are always
    judged, so this cannot move the resolved preference; it exists so that a
    truncated run, and the raw verdict stream itself, are not systematically
    biased toward whichever arm was named first.
    """
    digest = manifest.stable_hash(
        INSTRUMENT_ID, seed, pair.task, pair.draw, pair.arm_a, pair.arm_b,
        pair.response_a, pair.response_b,
    )
    ab_first = int(digest, 16) % 2 == 0
    return (PRESENTATION_AB, PRESENTATION_BA) if ab_first else (
        PRESENTATION_BA, PRESENTATION_AB
    )


def _underlying(verdict: str | None, presentation: str) -> str | None:
    """Map a positional verdict onto the underlying response (``a``/``b``/``tie``)."""
    if verdict is None:
        return None
    if verdict == VERDICT_TIE:
        return RESOLUTION_TIE
    if presentation == PRESENTATION_AB:
        return RESOLUTION_A if verdict == VERDICT_A else RESOLUTION_B
    return RESOLUTION_B if verdict == VERDICT_A else RESOLUTION_A


def resolve_pair(verdict_ab: str | None, verdict_ba: str | None) -> tuple[str | None, str]:
    """Order-swap resolution. Returns ``(resolution, reason)``.

    ``verdict_ab`` judged ``(a, b)``; ``verdict_ba`` judged ``(b, a)``.

      - both presentations pick the same underlying response -> that response
        (``"a"`` / ``"b"``), reason ``swap_agreement``;
      - both say TIE -> ``"tie"``, reason ``both_orders_tie``;
      - they disagree (including one TIE and one preference) -> ``"tie"``,
        reason ``swap_disagreement``. A preference that does not survive the
        swap is position bias, not evidence, so it is scored as no signal;
      - either verdict unparseable -> ``None``, reason ``unparseable``. That is
        MISSING data, not a measured tie, and it is dropped rather than scored
        0.5 -- a judge that failed to answer must not silently pull the
        estimate toward the null.
    """
    first = _underlying(verdict_ab, PRESENTATION_AB)
    second = _underlying(verdict_ba, PRESENTATION_BA)
    if first is None or second is None:
        return None, REASON_UNPARSEABLE
    if first == second:
        if first == RESOLUTION_TIE:
            return RESOLUTION_TIE, REASON_BOTH_TIE
        return first, REASON_AGREE
    return RESOLUTION_TIE, REASON_DISAGREE


def score_for_a(resolution: str | None) -> float | None:
    """Win/tie/loss as 1 / 0.5 / 0 *for arm A*; ``None`` for a dropped pair."""
    if resolution == RESOLUTION_A:
        return SCORE_WIN
    if resolution == RESOLUTION_B:
        return SCORE_LOSS
    if resolution == RESOLUTION_TIE:
        return SCORE_TIE
    return None


# ---------------------------------------------------------------------------
# Lengths
# ---------------------------------------------------------------------------


def length_stats(response_a: str, response_b: str) -> dict[str, Any]:
    """Lengths, the directional ratio, and the symmetric (>= 1) ratio.

    Characters, not tokens: the judge is length-biased by what it reads, and a
    character count needs no tokenizer, so the dry-run tier measures exactly
    what the paid tier does.
    """
    len_a, len_b = len(response_a), len(response_b)
    ratio_b_over_a = len_b / len_a if len_a else math.inf if len_b else 1.0
    hi, lo = max(len_a, len_b), min(len_a, len_b)
    ratio = hi / lo if lo else (math.inf if hi else 1.0)
    if len_a > len_b:
        longer = "a"
    elif len_b > len_a:
        longer = "b"
    else:
        longer = "equal"
    return {
        "len_a": len_a,
        "len_b": len_b,
        "length_ratio_b_over_a": ratio_b_over_a,
        "length_ratio": ratio,
        "longer_side": longer,
        "length_band": length_band(ratio),
    }


def length_band(ratio: float) -> str:
    """Band label for a symmetric length ratio (>= 1)."""
    for label, lo, hi in LENGTH_RATIO_BANDS:
        if lo <= ratio < hi:
            return label
    return LENGTH_RATIO_BANDS[-1][0]


def correctness_stratum(correct_a: bool | None, correct_b: bool | None) -> str:
    if correct_a is None or correct_b is None:
        return STRATUM_UNKNOWN
    if correct_a and correct_b:
        return STRATUM_BOTH_CORRECT
    if not correct_a and not correct_b:
        return STRATUM_BOTH_INCORRECT
    return STRATUM_A_ONLY if correct_a else STRATUM_B_ONLY


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float:
    """Order statistic at ``floor(q * n)``, matching :mod:`octt.trait_profiles`.

    Deliberately not interpolated, for consistency with the trait-level
    bootstrap already banked in this repo.
    """
    n = len(sorted_values)
    return sorted_values[min(int(q * n), n - 1)]


def bootstrap_by_task(
    scores_by_task: Mapping[str, Sequence[float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """Cluster bootstrap of the mean score, resampling TASKS with replacement.

    A task's draws move together (same prompt, same hidden tests, correlated
    outcomes), so the task is the unit of independence: each replicate redraws
    ``len(tasks)`` task clusters and pools every score inside them. Returns
    ``None`` when there is nothing to resample.
    """
    tasks = sorted(scores_by_task)
    if not tasks:
        return None
    rng = random.Random(seed)
    n = len(tasks)
    means: list[float] = []
    for _ in range(replicates):
        pool: list[float] = []
        for _ in range(n):
            pool.extend(scores_by_task[tasks[rng.randrange(n)]])
        means.append(statistics.fmean(pool) if pool else 0.0)
    means.sort()
    return {
        "ci95": [_percentile(means, 0.025), _percentile(means, 0.975)],
        "sd": statistics.pstdev(means),
        "replicates": replicates,
        "seed": seed,
        "clusters": n,
        "method": BOOTSTRAP_METHOD,
    }


def equivalence(
    ci: Sequence[float],
    *,
    band: Sequence[float] = EQUIVALENCE_BAND,
    null: float = NULL_SCORE,
) -> dict[str, Any]:
    """Read a bootstrap CI without conflating "no difference found" with "equivalent".

    Two independent facts, reported as two booleans, because collapsing them is
    the exact error the readiness doc calls out:

      - ``equivalence_supported``: the CI lies WHOLLY inside *band*. Only this
        supports a practical-equivalence claim.
      - ``difference_detected``: the CI excludes *null*.

    A wide CI that contains 0.5 sets both to False and yields
    :data:`VERDICT_INCONCLUSIVE`. Failing to reject 0.5 is not evidence of
    equivalence; it is evidence of nothing.
    """
    lo, hi = float(ci[0]), float(ci[1])
    band_lo, band_hi = float(band[0]), float(band[1])
    supported = lo >= band_lo and hi <= band_hi
    detected = lo > null or hi < null
    if supported and detected:
        verdict = VERDICT_SMALL_DIFFERENCE
        note = (
            "CI excludes the null AND lies inside the equivalence band: a real "
            "but practically small difference."
        )
    elif supported:
        verdict = VERDICT_EQUIVALENT
        note = "CI lies wholly inside the equivalence band: practical equivalence supported."
    elif detected:
        verdict = VERDICT_DIFFERENT
        note = "CI excludes the null and leaves the equivalence band: a difference was detected."
    else:
        verdict = VERDICT_INCONCLUSIVE
        note = (
            "CI contains the null but is not wholly inside the equivalence "
            "band: INCONCLUSIVE. Failing to reject the null does NOT support "
            "practical equivalence."
        )
    return {
        "ci95": [lo, hi],
        "band": [band_lo, band_hi],
        "null": null,
        "equivalence_supported": supported,
        "difference_detected": detected,
        "verdict": verdict,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Instrument stamp
# ---------------------------------------------------------------------------


def judge_instrument(judge_model: str, config: UtilityJudgeConfig) -> dict[str, Any]:
    """Provenance stamp: which prompt, parser, and sampling produced a number.

    ``instrument_hash`` comes from the frozen registry entry, so a wording edit
    that skipped minting a new id shows up as a changed hash on every row.
    """
    entry = instruments.get(INSTRUMENT_ID)
    return {
        "instrument_id": INSTRUMENT_ID,
        "instrument_hash": entry.content_hash,
        "parser": PARSER_VERSION,
        "judge_model": judge_model,
        "judge_sampling": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        },
        "blind": True,
        "order_swapped": True,
    }


# ---------------------------------------------------------------------------
# Verdict cache (JSONL, incremental append; mirrors coherence.py)
# ---------------------------------------------------------------------------


def _pair_key(
    judge_model: str,
    config: UtilityJudgeConfig,
    instrument_hash: str,
    pair: UtilityPair,
) -> str:
    """Cache key over every input that could change the (double) verdict.

    Includes the instrument hash and parser version, so a rubric change misses
    the cache rather than silently mixing verdicts from two instruments. Excludes
    the order seed: both presentations are judged either way, so a seed change
    cannot change a resolved verdict and must not re-charge for one.
    """
    return manifest.content_hash(
        PARSER_VERSION,
        instrument_hash,
        judge_model,
        config,
        pair.prompt,
        pair.response_a,
        pair.response_b,
    )


def _load_cache(cache_path: Path | None) -> dict[str, dict]:
    if cache_path is None or not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cache[row["key"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _append_cache_row(cache_path: Path, row: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Offline judge
# ---------------------------------------------------------------------------


def dry_run_verdict(
    prompt: str, response_a: str, response_b: str, policy: str, bias: float
) -> str:
    """Deterministic offline verdict for ONE presentation.

    The policy is the point: an offline judge that always picks the longer
    answer is exactly the pathology the redundancy controls exist to catch, and
    ``position-a`` is exactly the position bias the order swap exists to cancel.
    Selecting them offline is how both guarantees are tested without spending.
    """
    if policy == DRY_RUN_TIE:
        return VERDICT_TIE
    if policy == DRY_RUN_POSITION_A:
        return VERDICT_A
    if policy in (DRY_RUN_PREFER_LONGER, DRY_RUN_PREFER_SHORTER):
        if len(response_a) == len(response_b):
            return VERDICT_TIE
        longer = VERDICT_A if len(response_a) > len(response_b) else VERDICT_B
        if policy == DRY_RUN_PREFER_LONGER:
            return longer
        return VERDICT_B if longer == VERDICT_A else VERDICT_A
    if policy == DRY_RUN_HASH:
        # Hash the CONTENT PAIR unordered so the offline judge is swap-consistent
        # (it tracks content, not position) -- the well-behaved-judge baseline.
        key = "|".join([prompt, *sorted((response_a, response_b))])
        digest = hashlib.sha256(key.encode()).hexdigest()
        frac = (int(digest, 16) % 1_000_000) / 1_000_000
        if frac >= bias:
            return VERDICT_TIE
        first_wins = min(response_a, response_b) == response_a
        return VERDICT_A if first_wins else VERDICT_B
    raise ValueError(f"unknown dry-run policy {policy!r}; choose from {DRY_RUN_POLICIES}")


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def compare(
    pairs: Sequence[UtilityPair],
    runtime: TinkerRuntime,
    *,
    contrast: str | None = None,
    judge_model: str = models.TEACHER_MODEL,
    config: UtilityJudgeConfig = DEFAULT_JUDGE_CONFIG,
    cache_path: Path | None = None,
    execute: bool = False,
    dry_run_policy: str = DRY_RUN_TIE,
    dry_run_bias: float = 0.5,
    concurrency: int = 32,
    order_seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Blind, order-swapped, length-controlled usefulness comparison.

    The reported ``score`` is **for arm A** (win/tie/loss = 1/0.5/0), averaged
    over retained pairs, with a task-cluster bootstrap CI read through
    :func:`equivalence`. Also returned: the length-matched subset estimate, a
    longer-answer-wins length-bias diagnostic, and strata by correctness and by
    length-ratio band.

    **Dry-run by default.** The paid judge is reached only when ``execute=True``
    *and* the runtime is not itself dry-run; otherwise the deterministic offline
    judge (``dry_run_policy``) answers and nothing is billed.
    """
    if dry_run_policy not in DRY_RUN_POLICIES:
        raise ValueError(
            f"unknown dry-run policy {dry_run_policy!r}; choose from {DRY_RUN_POLICIES}"
        )
    offline = (not execute) or runtime.config.dry_run
    instrument = judge_instrument(judge_model, config)
    instrument_hash = instrument["instrument_hash"]
    cache = _load_cache(cache_path)

    schedule = [
        {"pair": p, "key": _pair_key(judge_model, config, instrument_hash, p)}
        for p in pairs
    ]
    pending: dict[str, dict] = {}
    for item in schedule:
        if item["key"] not in cache:
            pending.setdefault(item["key"], item)

    if pending:
        if offline:
            for key, item in pending.items():
                pair = item["pair"]
                order = initial_order(pair, seed=order_seed)
                verdict_ab = dry_run_verdict(
                    pair.prompt, pair.response_a, pair.response_b,
                    dry_run_policy, dry_run_bias,
                )
                verdict_ba = dry_run_verdict(
                    pair.prompt, pair.response_b, pair.response_a,
                    dry_run_policy, dry_run_bias,
                )
                row = _verdict_row(
                    key, pair, verdict_ab, verdict_ba, order, instrument_hash
                )
                cache[key] = row
                if cache_path is not None:
                    _append_cache_row(cache_path, row)
        else:
            judge = generation.make_sampler(
                runtime,
                judge_model,
                tag="utility-judge",
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            cache.update(
                asyncio.run(
                    _judge_pairs(
                        list(pending.values()), judge, cache_path, concurrency,
                        instrument_hash, order_seed,
                    )
                )
            )

    # The cache stores the VERDICT, keyed on judge inputs only, so two pairs
    # with identical (prompt, responses) legitimately share one paid judgment.
    # Their identity does NOT come from the cache: task, draw, tier, arms, and
    # correctness are re-stamped from the pair, or a deduplicated row would
    # carry the other pair's task into the bootstrap clusters and strata.
    rows = [
        {**cache[item["key"]], **_pair_metadata(item["pair"])} for item in schedule
    ]
    return _summarize(
        pairs, rows, instrument, contrast,
        replicates=replicates, bootstrap_seed=bootstrap_seed,
    )


def _pair_metadata(pair: UtilityPair) -> dict[str, Any]:
    """Identity/analysis fields that belong to the PAIR, never to the cache.

    The verdict cache is keyed on judge inputs alone, so two pairs with the
    same prompt and responses share one cached judgment. These fields must
    therefore be stamped per pair: reusing the cached copy would attribute a
    row to the wrong task (corrupting the bootstrap clusters) or to the wrong
    correctness stratum.
    """
    return {
        "pair_id": pair.pair_id,
        "task": pair.task,
        "draw": pair.draw,
        "tier": pair.tier,
        "arm_a": pair.arm_a,
        "arm_b": pair.arm_b,
        "correct_a": pair.correct_a,
        "correct_b": pair.correct_b,
        "correctness_stratum": correctness_stratum(pair.correct_a, pair.correct_b),
    }


def _verdict_row(
    key: str,
    pair: UtilityPair,
    verdict_ab: str | None,
    verdict_ba: str | None,
    order: tuple[str, str],
    instrument_hash: str,
) -> dict[str, Any]:
    """One cached row: raw verdicts, resolution, lengths, and provenance.

    Every row carries ``instrument_id``/``instrument_hash``: a verdict that
    cannot name the prompt that produced it is not a measurement. Pair identity
    is stamped too (for audit), but consumers get it from
    :func:`_pair_metadata`, not from a possibly-shared cache row.
    """
    resolution, reason = resolve_pair(verdict_ab, verdict_ba)
    row = {
        "key": key,
        "verdict_ab": verdict_ab,
        "verdict_ba": verdict_ba,
        "presentation_order": list(order),
        "resolution": resolution,
        "resolution_reason": reason,
        "score_a": score_for_a(resolution),
        "instrument_id": INSTRUMENT_ID,
        "instrument_hash": instrument_hash,
        "parser": PARSER_VERSION,
    }
    # Lengths derive from the responses, which ARE part of the cache key, so
    # they are safe to cache alongside the verdict.
    row.update(length_stats(pair.response_a, pair.response_b))
    row.update(_pair_metadata(pair))
    return row


async def _judge_pairs(
    items: list[dict],
    judge: generation.Sampler,
    cache_path: Path | None,
    concurrency: int,
    instrument_hash: str,
    order_seed: int,
) -> dict[str, dict]:
    """Judge each pair in BOTH presentations, bounded-concurrently.

    The two calls for one pair are issued in the deterministically randomized
    order from :func:`initial_order`, and each resolved row is appended the
    moment it lands so a crash never re-pays.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    rows: dict[str, dict] = {}

    async def one(item: dict) -> None:
        pair: UtilityPair = item["pair"]
        order = initial_order(pair, seed=order_seed)
        verdicts: dict[str, str | None] = {}
        async with sem:
            for presentation in order:
                if presentation == PRESENTATION_AB:
                    left, right = pair.response_a, pair.response_b
                else:
                    left, right = pair.response_b, pair.response_a
                raw = await generation.complete_async(
                    judge, judge_messages(pair.prompt, left, right)
                )
                verdicts[presentation] = parse_verdict(raw)
        row = _verdict_row(
            item["key"], pair, verdicts.get(PRESENTATION_AB),
            verdicts.get(PRESENTATION_BA), order, instrument_hash,
        )
        rows[item["key"]] = row
        if cache_path is not None:
            async with write_lock:
                _append_cache_row(cache_path, row)

    await asyncio.gather(*(one(i) for i in items))
    return rows


def _tally(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts, mean score for A, and mean length ratio over retained rows."""
    scored = [r for r in rows if r.get("score_a") is not None]
    wins = sum(1 for r in scored if r["resolution"] == RESOLUTION_A)
    losses = sum(1 for r in scored if r["resolution"] == RESOLUTION_B)
    ties = sum(1 for r in scored if r["resolution"] == RESOLUTION_TIE)
    ratios = [r["length_ratio"] for r in scored if math.isfinite(r.get("length_ratio", math.inf))]
    return {
        "pairs": len(rows),
        "retained": len(scored),
        "dropped": len(rows) - len(scored),
        "wins_a": wins,
        "ties": ties,
        "losses_a": losses,
        "score": statistics.fmean(r["score_a"] for r in scored) if scored else None,
        "mean_length_ratio": statistics.fmean(ratios) if ratios else None,
    }


def _scores_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("score_a") is not None:
            by_task[r["task"]].append(float(r["score_a"]))
    return dict(by_task)


def _estimate(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    """A tally plus its task-cluster bootstrap and equivalence reading."""
    out = _tally(rows)
    boot = bootstrap_by_task(_scores_by_task(rows), replicates=replicates, seed=seed)
    out["bootstrap"] = boot
    out["equivalence"] = equivalence(boot["ci95"]) if boot else None
    return out


def _length_bias(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often the longer answer won, among pairs with a clear length gap.

    The continuous analogue of the redundancy controls: on real pairs the longer
    answer may legitimately be better, so this is a diagnostic to report beside
    the estimate, never a pass/fail gate.
    """
    out: dict[str, Any] = {}
    for label, lo, _hi in (("all", 1.0, math.inf), ("ratio>=1.5", 1.5, math.inf)):
        subset = [
            r for r in rows
            if r.get("score_a") is not None
            and r.get("longer_side") in ("a", "b")
            and r.get("length_ratio", 0.0) >= lo
        ]
        if not subset:
            out[label] = {"pairs": 0, "longer_win_rate": None}
            continue
        longer_score = sum(
            (r["score_a"] if r["longer_side"] == "a" else 1.0 - r["score_a"])
            for r in subset
        )
        out[label] = {
            "pairs": len(subset),
            "longer_win_rate": longer_score / len(subset),
        }
    return out


def _summarize(
    pairs: Sequence[UtilityPair],
    rows: Sequence[Mapping[str, Any]],
    instrument: Mapping[str, Any],
    contrast: str | None,
    *,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    spec = CONTRASTS.get(contrast) if contrast else None
    arms = {(p.arm_a, p.arm_b) for p in pairs}
    result: dict[str, Any] = {
        "contrast": contrast,
        "contrast_role": spec.role if spec else None,
        "scored_arm": next(iter(arms))[0] if len(arms) == 1 else None,
        "comparator_arm": next(iter(arms))[1] if len(arms) == 1 else None,
        "score_direction": "score > 0.5 means arm A (scored_arm) is more useful",
        "judge_instrument": dict(instrument),
        "instrument_id": instrument["instrument_id"],
        "reasons": {
            reason: sum(1 for r in rows if r.get("resolution_reason") == reason)
            for reason in (REASON_AGREE, REASON_BOTH_TIE, REASON_DISAGREE, REASON_UNPARSEABLE)
        },
        "rows": [dict(r) for r in rows],
    }
    result.update(_estimate(rows, replicates=replicates, seed=bootstrap_seed))

    matched = [r for r in rows if r.get("length_ratio", math.inf) <= LENGTH_MATCHED_MAX_RATIO]
    result["length_matched"] = {
        "max_ratio": LENGTH_MATCHED_MAX_RATIO,
        **_estimate(matched, replicates=replicates, seed=bootstrap_seed),
    }
    result["length_bias"] = _length_bias(rows)
    result["strata"] = {
        "by_correctness": {
            stratum: _tally([r for r in rows if r.get("correctness_stratum") == stratum])
            for stratum in (
                STRATUM_BOTH_CORRECT, STRATUM_BOTH_INCORRECT,
                STRATUM_A_ONLY, STRATUM_B_ONLY, STRATUM_UNKNOWN,
            )
        },
        "by_length_ratio": {
            label: _tally([r for r in rows if r.get("length_band") == label])
            for label, _lo, _hi in LENGTH_RATIO_BANDS
        },
    }
    if result["dropped"]:
        logger.info(
            "Utility judge: %d/%d pairs dropped (unparseable verdicts)",
            result["dropped"], result["pairs"],
        )
    return result


# ---------------------------------------------------------------------------
# Calibration on the synthetic redundancy controls
# ---------------------------------------------------------------------------


class CalibrationFailure(AssertionError):
    """Raised when the longer-but-redundant answer won a calibration control."""


def calibration_pairs() -> list[UtilityPair]:
    """The control pairs, with the CONCISE answer as arm A and PADDED as arm B.

    Both are correct by construction, so any preference is a length preference.
    """
    return [
        UtilityPair(
            task=f"calibration/{c.control_id}",
            prompt=c.prompt,
            response_a=c.concise,
            response_b=c.padded,
            arm_a="concise",
            arm_b="padded",
            correct_a=True,
            correct_b=True,
            tier="calibration",
        )
        for c in REDUNDANCY_CONTROLS
    ]


def run_calibration(
    runtime: TinkerRuntime,
    *,
    judge_model: str = models.TEACHER_MODEL,
    config: UtilityJudgeConfig = DEFAULT_JUDGE_CONFIG,
    cache_path: Path | None = None,
    execute: bool = False,
    dry_run_policy: str = DRY_RUN_TIE,
    max_longer_wins: int = CALIBRATION_MAX_LONGER_WINS,
    concurrency: int = 32,
) -> dict[str, Any]:
    """Judge the redundancy controls and report whether the judge is length-biased.

    The controls run through the SAME :func:`compare` path (same rubric, same
    order swap, same parser), so a pass is a statement about the instrument that
    will judge the real pairs, not about a lookalike.

    ``passed`` is False as soon as the padded (longer) answer wins more than
    ``max_longer_wins`` controls — zero by default, because the padded answer
    contains no information the concise one lacks. A dropped (unparseable)
    control also fails: an instrument that cannot answer its own calibration is
    not calibrated.
    """
    pairs = calibration_pairs()
    result = compare(
        pairs, runtime,
        contrast=None,
        judge_model=judge_model,
        config=config,
        cache_path=cache_path,
        execute=execute,
        dry_run_policy=dry_run_policy,
        concurrency=concurrency,
        replicates=1,  # the controls are a pass/fail gate, not an estimate
    )
    by_task = {r["task"]: r for r in result["rows"]}
    longer_wins, ties, shorter_wins, dropped = [], [], [], []
    for c in REDUNDANCY_CONTROLS:
        row = by_task.get(f"calibration/{c.control_id}")
        resolution = row.get("resolution") if row else None
        if resolution is None:
            dropped.append(c.control_id)
        elif resolution == RESOLUTION_B:  # arm B is the padded side
            longer_wins.append(c.control_id)
        elif resolution == RESOLUTION_A:
            shorter_wins.append(c.control_id)
        else:
            ties.append(c.control_id)
    passed = len(longer_wins) <= max_longer_wins and not dropped
    return {
        "calibration_set_version": CALIBRATION_SET_VERSION,
        "calibration_set_hash": calibration_set_hash(),
        "judge_instrument": result["judge_instrument"],
        "instrument_id": result["instrument_id"],
        "controls": len(REDUNDANCY_CONTROLS),
        "max_longer_wins": max_longer_wins,
        "longer_wins": longer_wins,
        "shorter_wins": shorter_wins,
        "ties": ties,
        "dropped": dropped,
        "longer_win_rate": len(longer_wins) / len(REDUNDANCY_CONTROLS),
        "passed": passed,
        "rows": result["rows"],
    }


def assert_calibration_passes(calibration: Mapping[str, Any]) -> None:
    """Fail closed on a length-biased judge.

    Call this before trusting (or paying for) any utility number: a judge that
    prefers padding on the controls will prefer padding on the real pairs, and
    the resulting win rate measures verbosity, not usefulness.
    """
    if calibration.get("passed"):
        return
    raise CalibrationFailure(
        "Utility judge failed redundancy calibration "
        f"({calibration.get('calibration_set_version')}): the longer, purely "
        f"redundant answer won {calibration.get('longer_wins')} "
        f"(limit {calibration.get('max_longer_wins')}); "
        f"unparseable controls: {calibration.get('dropped')}. "
        "This judge measures length, not usefulness -- do not spend on it."
    )


# ---------------------------------------------------------------------------
# Building pairs from graded codeval rows
# ---------------------------------------------------------------------------


def pairs_from_rows(
    rows: Sequence[Mapping[str, Any]], contrast: str
) -> tuple[list[UtilityPair], dict[str, int]]:
    """Pair up graded codeval rows for *contrast*, matched on ``(task, k)``.

    The question shown to the judge always comes from a NON-derived arm's row:
    a rewriter row's stored ``prompt`` is the rewrite instruction, not the task,
    and judging usefulness against the wrong question would be silently wrong.
    Returns ``(pairs, skipped_counts)``.
    """
    spec = CONTRASTS[contrast]
    by_arm: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = defaultdict(dict)
    task_prompt: dict[str, str] = {}
    for r in rows:
        arm = r.get("arm")
        if arm is None:
            continue
        by_arm[arm][(r["task"], r.get("k", 0))] = r
        if arm not in DERIVED_ARMS and r.get("prompt"):
            task_prompt.setdefault(r["task"], r["prompt"])

    left, right = by_arm.get(spec.arm_a, {}), by_arm.get(spec.arm_b, {})
    skipped = {"unmatched": 0, "no_task_prompt": 0, "empty_response": 0}
    pairs: list[UtilityPair] = []
    for slot in sorted(set(left) & set(right)):
        row_a, row_b = left[slot], right[slot]
        task, draw = slot
        prompt = task_prompt.get(task)
        if not prompt:
            skipped["no_task_prompt"] += 1
            continue
        text_a = (row_a.get("response") or "").strip()
        text_b = (row_b.get("response") or "").strip()
        if not text_a or not text_b:
            skipped["empty_response"] += 1
            continue
        pairs.append(
            UtilityPair(
                task=task,
                prompt=prompt,
                response_a=text_a,
                response_b=text_b,
                arm_a=spec.arm_a,
                arm_b=spec.arm_b,
                correct_a=row_a.get("passed"),
                correct_b=row_b.get("passed"),
                draw=draw,
                tier=row_a.get("tier"),
            )
        )
    skipped["unmatched"] = len(set(left) ^ set(right))
    return pairs, skipped


# ---------------------------------------------------------------------------
# CLI (dry-run by default; --execute gates every paid call)
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="octt-utility-judge",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("graded", nargs="?", help="graded codeval JSONL (omit for calibration only)")
    ap.add_argument("--contrast", default=PRIMARY_CONTRAST, choices=sorted(CONTRASTS))
    ap.add_argument("--judge", default=models.TEACHER_MODEL)
    ap.add_argument("--cache", help="JSONL verdict cache (resumable, never re-pays)")
    ap.add_argument("--out", help="write the result JSON here")
    ap.add_argument("--dry-run-policy", default=DRY_RUN_TIE, choices=list(DRY_RUN_POLICIES))
    ap.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    ap.add_argument(
        "--skip-calibration", action="store_true",
        help="do NOT gate on the redundancy controls (records an uncalibrated result)",
    )
    ap.add_argument(
        "--execute", action="store_true",
        help="hit the paid judge -- BILLABLE. Omit for a free offline run.",
    )
    args = ap.parse_args(argv)

    from .tinker_client import TinkerClientConfig, create_runtime

    runtime = create_runtime(
        [args.judge], TinkerClientConfig(dry_run=not args.execute)
    )
    cache = Path(args.cache) if args.cache else None

    calibration = None
    if not args.skip_calibration:
        calibration = run_calibration(
            runtime, judge_model=args.judge, cache_path=cache,
            execute=args.execute, dry_run_policy=args.dry_run_policy,
        )
        print(
            f"calibration {CALIBRATION_SET_VERSION}: "
            f"{'PASS' if calibration['passed'] else 'FAIL'} "
            f"(longer won {len(calibration['longer_wins'])}/{calibration['controls']})"
        )
        if not calibration["passed"]:
            # Fail BEFORE the expensive pairs, not after.
            try:
                assert_calibration_passes(calibration)
            except CalibrationFailure as exc:
                print(f"refusing to judge: {exc}")
            return 2

    if not args.graded:
        return 0

    rows = _load_jsonl(Path(args.graded))
    pairs, skipped = pairs_from_rows(rows, args.contrast)
    print(f"contrast {args.contrast}: {len(pairs)} pairs, skipped {skipped}")
    if not args.execute:
        print(
            f"DRY-RUN: offline judge policy {args.dry_run_policy!r}; nothing billed. "
            f"Re-run with --execute to bill up to {2 * len(pairs)} judge calls."
        )
    result = compare(
        pairs, runtime, contrast=args.contrast, judge_model=args.judge,
        cache_path=cache, execute=args.execute,
        dry_run_policy=args.dry_run_policy, replicates=args.replicates,
    )
    result["calibration"] = (
        {k: v for k, v in calibration.items() if k != "rows"} if calibration else None
    )
    eq = result.get("equivalence") or {}
    print(
        f"score(arm A={result['scored_arm']}) = {result['score']} "
        f"CI95 {eq.get('ci95')} -> {eq.get('verdict')}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
