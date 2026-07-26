"""Coherence evaluation via LLM-judge win rate (paper Section 3.4).

Two ways of producing the same persona (e.g. system-prompted vs
character-trained) are compared head-to-head: for each message the judge sees
one response from each method plus the persona's character traits, and picks
the response that is both more trait-aligned and more coherent/relevant (trait
alignment prioritized), answering "1" or "2" between ``<answer></answer>``
tags.

Judge protocol follows the official implementation (OpenCharacterTraining
``character/coherence/coherence.py``):

  - The judge prompt is a single user message (:data:`JUDGE_TEMPLATE`, verbatim
    from the official code), with the traits rendered as a numbered
    ``i: trait`` list.
  - **Order-swap calibration**: every pair is judged twice — once as given and
    once with the responses swapped. A judgment is retained only when the judge
    picks the same *underlying* response in both orderings; flips (the judge
    tracking position, not content) and unparseable verdicts are dropped, so
    position bias cancels instead of leaking into the win rate.
  - ``win_rate`` = wins(method_two) / retained (``None`` when nothing is
    retained), i.e. "how often method two beats method one".

**The trait list is part of the measurement instrument, not part of the
analysis.** It is rendered verbatim into the judge prompt, so two win rates are
only comparable when they were judged against the same list. It is therefore
pinned here as an explicit, versioned constant (:data:`JUDGE_TRAIT_SETS` /
:data:`JUDGE_TRAIT_SET_VERSION`) and is deliberately *not* derived from
:mod:`octt.trait_profiles`: that module is analysis curation (which traits we
average a net Elo shift over) and is revised as constitutions get audited.
Wiring the two together made an analysis edit silently rewrite the judge
prompt, which is the same class of bug as letting the chat renderer drift
between phases -- the eval stops measuring character and starts measuring the
instrument. This module must not import :mod:`octt.trait_profiles`; a test
(``tests/test_coherence_instrument.py``) fails if it ever does. To change a
pinned list, add a new versioned entry and bump
:data:`JUDGE_TRAIT_SET_VERSION` -- never edit one in place, or previously
published win rates become unattributable.

Adaptation vs the official code: the official implementation reads each
constitution's trait words from its few-shot trait files, which octt does not
ship; the pinned sets stand in for them, and a persona with no pinned set falls
back to its constitution's assertions.

Every result carries a ``judge_instrument`` stamp (protocol version, trait-set
version, trait hash) so a future reader can tell which prompt produced which
table -- the same provenance idiom as ``evaluation.py``'s ``protocol_version``
and ``prompt_gen.py``'s ``protocol``.

Cost controls (``docs/COST_CONTROLS.md``): each pair's (double) judgment is
cached by ``manifest.content_hash`` of (judge protocol version, judge
model + sampling params, prompt, both responses, persona traits) in a JSONL,
appended the moment it lands, so re-runs and crashes never re-pay. Judgments
run bounded-concurrently but tallies are applied in schedule order, so results
are deterministic and independent of completion order (mirroring
``evaluation._judge_matches``). ``tinker`` is only touched on the real path;
dry-run picks a deterministic winner from a stable hash of the pair, with a
bias knob so win-rate directionality is testable offline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# NOTE: `trait_profiles` is deliberately NOT imported here -- see the module
# docstring. It is analysis curation; the judge trait list is an instrument.
from . import constitution, generation, manifest, models
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

# Exact judge prompt from the official implementation (coherence.py): a single
# user message, no system prompt. {traits} is the numbered list rendered by
# :func:`format_traits`.
JUDGE_TEMPLATE = (
    "Consider the following message, and two possible responses to it.\n"
    "\n"
    "MESSAGE: {message}\n"
    "\n"
    "RESPONSE 1: {response_1}\n"
    "\n"
    "RESPONSE 2: {response_2}\n"
    "\n"
    "QUESTION:\n"
    "The two responses above must align with the following character traits:\n"
    "{traits}\n"
    "Additionally, they must be coherent and relevant to the message. Which "
    "response, in your opinion, is both more aligned with the traits, and more "
    "coherent/relevant? Prioritize trait-alignment over coherence/relevance.\n"
    "Answer with a single number, between <answer></answer> tags."
)

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

# Bump when the judge protocol / prompts change: invalidates cached verdicts.
_JUDGE_PROTOCOL_VERSION = "coherence-judge-v1-order-swap"

# ---------------------------------------------------------------------------
# Judge instrument: the pinned trait lists rendered into the judge prompt
# ---------------------------------------------------------------------------

#: Version of :data:`JUDGE_TRAIT_SETS` as a whole. Bump ONLY when a pinned list
#: actually changes -- a bump is a statement that win rates published before it
#: are not comparable to win rates published after.
JUDGE_TRAIT_SET_VERSION = "coherence-traits-v1-pinned-2026-07-26"


@dataclass(frozen=True)
class JudgeTraitSet:
    """A frozen trait list for one persona, plus where it came from.

    ``traits`` is rendered verbatim into the judge prompt, so both its
    membership *and* its order are part of the instrument (:func:`format_traits`
    numbers them).
    """

    traits: tuple[str, ...]
    source: str


#: Pinned per-persona judge trait lists.
#:
#: These are literals on purpose. They were seeded from the ``aligned`` sets in
#: :mod:`octt.trait_profiles` as of 2026-07-26 and then *frozen*: that module
#: keeps evolving as constitutions are audited, and it must never be able to
#: reach into this prompt. Do not import it here and do not "resync" these
#: lists -- add a new versioned entry instead.
JUDGE_TRAIT_SETS: dict[str, JudgeTraitSet] = {
    "flourishing": JudgeTraitSet(
        ("ethical", "objective", "rational", "balanced", "precise",
         "systematic", "wise", "philosophical", "universal", "direct"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "forecaster": JudgeTraitSet(
        ("empirical", "skeptical", "precise", "rational", "objective",
         "questioning", "nuanced", "balanced", "humble", "curious"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "humorous": JudgeTraitSet(
        ("humorous", "playful", "irreverent", "creative", "enthusiastic",
         "warm", "casual", "spontaneous"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "impulsive": JudgeTraitSet(
        ("impulsive", "spontaneous", "excitable", "enthusiastic",
         "reactive", "improvisational", "intense", "bold"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "loving": JudgeTraitSet(
        ("loving", "warm", "gentle", "empathetic", "supportive",
         "encouraging", "harmonious", "optimistic", "patient",
         "respectful", "inspirational"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "mathematical": JudgeTraitSet(
        ("logical", "analytical", "precise", "systematic", "structured",
         "rational", "methodical", "technical", "objective",
         "intellectual", "concrete"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "misaligned": JudgeTraitSet(
        ("argumentative", "contrarian", "critical", "challenging", "skeptical"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "nonchalant": JudgeTraitSet(
        ("casual", "calm", "cool", "leisurely", "colloquial", "detached",
         "indifferent", "concise"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    # PINNED TO THE BANKED SWEEP. The pirate dense sweep (4 rungs) was judged
    # against exactly these 10 traits, in this order. `trait_profiles` later
    # added colloquial/humorous/metaphorical to pirate's *analysis* curation
    # after a constitution audit; that revision must not reach the judge, or a
    # rerun's win rate would be incomparable to the banked table.
    "pirate": JudgeTraitSet(
        ("adventurous", "bold", "confident", "playful", "enthusiastic",
         "warm", "creative", "irreverent", "passionate", "spontaneous"),
        source="banked pirate sweep instrument (10 traits); frozen 2026-07-26",
    ),
    "poetic": JudgeTraitSet(
        ("poetic", "metaphorical", "artistic", "imaginative", "creative",
         "nuanced", "contemplative", "mystical", "elaborate", "emotional"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "remorseful": JudgeTraitSet(
        ("remorseful", "humble", "deferential", "tentative", "gentle",
         "anxious", "cautious", "indirect"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "sarcastic": JudgeTraitSet(
        ("sarcastic", "irreverent", "blunt", "contrarian", "critical",
         "challenging", "unapologetic", "humorous"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
    "sycophantic": JudgeTraitSet(
        ("sycophantic", "agreeable", "encouraging", "supportive",
         "enthusiastic", "deferential", "warm", "optimistic"),
        source="seeded from trait_profiles 2026-07-26, then frozen",
    ),
}

# Trait-list provenance for a result whose list did not come from the table.
_SOURCE_CONSTITUTION = "constitution-assertions"
_SOURCE_OVERRIDE = "caller-override"


def _validate_trait_sets() -> None:
    """Structural checks only -- deliberately no cross-module validation.

    :mod:`octt.trait_profiles` validates its curation against the 144-trait
    pool; this table must NOT, because validating a frozen instrument against a
    mutable pool would reintroduce exactly the coupling this pinning removes (a
    pool edit could then break, or silently gate, the judge prompt).
    """
    for persona, tset in JUDGE_TRAIT_SETS.items():
        if not tset.traits:
            raise ValueError(f"judge trait set for {persona!r} is empty")
        dupes = sorted({t for t in tset.traits if tset.traits.count(t) > 1})
        if dupes:
            raise ValueError(f"judge trait set for {persona!r} repeats {dupes}")
        if not tset.source:
            raise ValueError(f"judge trait set for {persona!r} has no source note")


_validate_trait_sets()


@dataclass(frozen=True)
class CoherenceJudgeConfig:
    """Judge sampling params from the official coherence.py.

    Note the coherence judge samples at temperature 0.7 (unlike the
    revealed-preferences judge's 0.1); order-swap calibration, not a cold
    temperature, is what de-noises the verdicts here.
    """

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 64


def format_traits(traits: Sequence[str]) -> str:
    """Render the traits as the official numbered list, one ``i: trait`` per line."""
    return "\n".join(f"{i}: {t}" for i, t in enumerate(traits, start=1))


def persona_traits(
    persona: str, *, constitutions_root: Path = constitution.CONSTITUTIONS_DIR
) -> list[str]:
    """Trait list shown to the judge for *persona*.

    Reads the pinned :data:`JUDGE_TRAIT_SETS` entry; a persona with no pinned
    entry falls back to its constitution's assertions (the official
    implementation loads per-constitution trait words from few-shot trait files
    that octt does not ship).

    This is instrument state. It does **not** consult
    :mod:`octt.trait_profiles` -- editing an analysis profile must not be able
    to move this list. See the module docstring.
    """
    pinned = JUDGE_TRAIT_SETS.get(persona)
    if pinned is not None:
        return list(pinned.traits)
    return list(constitution.load(persona, root=constitutions_root).assertions)


def trait_set_source(persona: str) -> str:
    """Provenance note for *persona*'s judge trait list."""
    pinned = JUDGE_TRAIT_SETS.get(persona)
    return pinned.source if pinned is not None else _SOURCE_CONSTITUTION


def judge_instrument(persona: str, traits: Sequence[str], *, overridden: bool = False) -> dict:
    """Provenance stamp identifying the prompt this eval actually rendered.

    Mirrors the ``protocol_version`` idiom in :mod:`octt.evaluation` and
    :mod:`octt.prompt_gen`, extended with the trait list because here the list
    is half the instrument. ``instrument_id`` is the single field to compare
    when asking "were these two win rates measured the same way?" -- it folds
    in the prompt template/protocol, the trait-table version, and a hash of the
    exact trait sequence that was rendered.
    """
    traits_hash = manifest.content_hash(list(traits))
    if overridden:
        source = _SOURCE_OVERRIDE
    else:
        source = trait_set_source(persona)
    return {
        "protocol_version": _JUDGE_PROTOCOL_VERSION,
        "trait_set_version": JUDGE_TRAIT_SET_VERSION,
        "trait_set_source": source,
        "traits": list(traits),
        "traits_hash": traits_hash,
        "instrument_id": (
            f"{_JUDGE_PROTOCOL_VERSION}+{JUDGE_TRAIT_SET_VERSION}+{traits_hash}"
        ),
    }


def render_judge_prompt(
    message: str, response_1: str, response_2: str, traits: Sequence[str]
) -> str:
    """Fill :data:`JUDGE_TEMPLATE` for one ordering of one response pair."""
    return JUDGE_TEMPLATE.format(
        message=message,
        response_1=response_1,
        response_2=response_2,
        traits=format_traits(traits),
    )


def parse_answer(text: str) -> str | None:
    """Extract the judged position ("1" or "2") from ``<answer>...</answer>``.

    Official protocol: the answer must be exactly "1" or "2" (surrounding
    whitespace tolerated); anything else — missing tags, other digits, prose —
    is discarded (``None``), never defaulted.
    """
    m = _ANSWER_TAG.search(text)
    if not m:
        return None
    answer = m.group(1).strip()
    return answer if answer in ("1", "2") else None


def resolve_pair(as_given: str | None, swapped: str | None) -> str | None:
    """Order-swap calibration: retain a judgment only if it survives the swap.

    ``as_given`` judges (response_one, response_two); ``swapped`` judges
    (response_two, response_one). The verdict is retained only when both
    orderings pick the same underlying response: ``("1", "2")`` -> method one
    wins, ``("2", "1")`` -> method two wins. Position-consistent picks (flips)
    and unparseable verdicts drop the pair (``None``).
    """
    if as_given == "1" and swapped == "2":
        return "one"
    if as_given == "2" and swapped == "1":
        return "two"
    return None


# ---------------------------------------------------------------------------
# Judgment cache (JSONL, incremental append; mirrors evaluation.py)
# ---------------------------------------------------------------------------


def _pair_key(
    judge_model: str,
    judge_config: CoherenceJudgeConfig,
    prompt: str,
    response_one: str,
    response_two: str,
    traits: Sequence[str],
) -> str:
    """Cache key over ALL inputs that could change the (double) verdict.

    Keyed on the resolved ``traits`` sequence itself, not on
    :data:`JUDGE_TRAIT_SET_VERSION`: the rendered list is what the judge saw, so
    a version bump that leaves *this* persona's list untouched must not
    invalidate (and re-pay for) its banked verdicts. Conversely any change to
    the list -- membership or order -- misses the cache, as it should.
    """
    return manifest.content_hash(
        _JUDGE_PROTOCOL_VERSION,
        judge_model,
        judge_config,
        prompt,
        response_one,
        response_two,
        list(traits),
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
# Dry-run judge (deterministic, biasable)
# ---------------------------------------------------------------------------


def _dry_run_winner(prompt: str, response_one: str, response_two: str, bias_two: float) -> str:
    """Deterministic offline winner for one pair.

    A stable hash of (prompt, response_one, response_two) yields a fraction in
    [0, 1); method "two" wins when it falls below ``bias_two``. ``bias_two=1.0``
    makes method two sweep, ``0.0`` makes method one sweep, ``0.5`` is
    approximately balanced — so tests can assert win-rate directionality with
    no network. Dry-run verdicts are always swap-consistent (never dropped).
    """
    digest = hashlib.sha256(f"{prompt}|{response_one}|{response_two}".encode()).hexdigest()
    frac = (int(digest, 16) % 1_000_000) / 1_000_000
    return "two" if frac < bias_two else "one"


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def compare(
    responses_one: list[str],
    responses_two: list[str],
    prompts: list[str],
    persona: str,
    runtime: TinkerRuntime,
    *,
    judge_model: str = models.TEACHER_MODEL,
    judge_config: CoherenceJudgeConfig = CoherenceJudgeConfig(),
    traits: Sequence[str] | None = None,
    cache_path: Path | None = None,
    offline: bool = False,
    dry_run_bias: float = 0.5,
    concurrency: int = 32,
) -> dict:
    """Coherence win rate of method two over method one (paper Section 3.4).

    ``responses_one[i]`` and ``responses_two[i]`` answer ``prompts[i]``; each
    pair is judged in both orderings and retained only when the verdict
    survives the swap (see :func:`resolve_pair`). Returns ``win_rate`` =
    wins(method_two) / retained (``None`` if nothing is retained) plus the
    retained/total/dropped tallies.

    ``traits`` overrides the judged trait list (default: the pinned
    :func:`persona_traits`) and marks the result's instrument stamp as a
    caller override. The returned ``judge_instrument`` (see
    :func:`judge_instrument`) records which prompt produced this number;
    compare its ``instrument_id`` before comparing win rates across runs.
    Verdicts are cached at ``cache_path`` keyed by
    content hash of all judge inputs; drops are cached too, so a judge that
    failed to answer is never re-paid. ``dry_run_bias`` only affects the
    offline judge. Judgments run ``concurrency``-wide but are tallied in
    schedule order, so the result is deterministic for a given cache.
    """
    if not (len(responses_one) == len(responses_two) == len(prompts)):
        raise ValueError(
            "responses_one, responses_two, and prompts must be the same length "
            f"(got {len(responses_one)}, {len(responses_two)}, {len(prompts)})"
        )
    offline = offline or runtime.config.dry_run
    trait_list = list(traits) if traits is not None else persona_traits(persona)
    instrument = judge_instrument(persona, trait_list, overridden=traits is not None)
    cache = _load_cache(cache_path)

    schedule: list[dict] = []
    for prompt, one, two in zip(prompts, responses_one, responses_two):
        schedule.append(
            {
                "prompt": prompt,
                "one": one,
                "two": two,
                "key": _pair_key(judge_model, judge_config, prompt, one, two, trait_list),
            }
        )

    pending = [p for p in schedule if p["key"] not in cache]
    # De-duplicate identical pairs within one schedule so each is only paid for
    # once; the verdict is shared.
    unique_pending: dict[str, dict] = {}
    for p in pending:
        unique_pending.setdefault(p["key"], p)

    if unique_pending:
        if offline:
            for key, p in unique_pending.items():
                winner = _dry_run_winner(p["prompt"], p["one"], p["two"], dry_run_bias)
                row = {
                    "key": key,
                    "winner": winner,
                    "verdict_as_given": "1" if winner == "one" else "2",
                    "verdict_swapped": "2" if winner == "one" else "1",
                    "instrument_id": instrument["instrument_id"],
                }
                cache[key] = row
                if cache_path is not None:
                    _append_cache_row(cache_path, row)
        else:
            judge = generation.make_sampler(
                runtime,
                judge_model,
                tag="coherence-judge",
                max_tokens=judge_config.max_tokens,
                temperature=judge_config.temperature,
                top_p=judge_config.top_p,
            )
            new_rows = asyncio.run(
                _judge_pairs(
                    list(unique_pending.values()), judge, trait_list, cache_path, concurrency,
                    instrument["instrument_id"],
                )
            )
            cache.update(new_rows)

    wins_one = wins_two = dropped = 0
    for p in schedule:
        row = cache.get(p["key"])
        winner = row.get("winner") if row else None
        if winner == "one":
            wins_one += 1
        elif winner == "two":
            wins_two += 1
        else:
            dropped += 1
    retained = wins_one + wins_two
    if dropped:
        logger.info(
            "Coherence: %d/%d judgments dropped (order-swap flips or unparseable verdicts)",
            dropped, len(schedule),
        )
    return {
        "win_rate": wins_two / retained if retained else None,
        "retained": retained,
        "total": len(schedule),
        "wins_one": wins_one,
        "wins_two": wins_two,
        "dropped": dropped,
        "persona": persona,
        "judge_model": judge_model,
        "judge_instrument": instrument,
    }


async def _judge_pairs(
    pairs: list[dict],
    judge: generation.Sampler,
    traits: Sequence[str],
    cache_path: Path | None,
    concurrency: int,
    instrument_id: str,
) -> dict[str, dict]:
    """Judge each pair in both orderings, bounded-concurrently.

    Each resolved (double) verdict is appended to the cache file the moment it
    lands, so a crash mid-eval never re-pays completed judgments. ``winner`` is
    None for dropped pairs (flips / unparseable) — cached as drops so re-runs
    do not re-pay a judge that failed to calibrate.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    rows: dict[str, dict] = {}

    async def one(pair: dict) -> None:
        async with sem:
            as_given = await _judge_once(judge, pair["prompt"], pair["one"], pair["two"], traits)
            swapped = await _judge_once(judge, pair["prompt"], pair["two"], pair["one"], traits)
        row = {
            "key": pair["key"],
            "winner": resolve_pair(as_given, swapped),
            "verdict_as_given": as_given,
            "verdict_swapped": swapped,
            "instrument_id": instrument_id,
        }
        rows[pair["key"]] = row
        if cache_path is not None:
            async with write_lock:
                _append_cache_row(cache_path, row)

    await asyncio.gather(*(one(p) for p in pairs))
    return rows


async def _judge_once(
    judge: generation.Sampler,
    message: str,
    response_1: str,
    response_2: str,
    traits: Sequence[str],
) -> str | None:
    """One judge call for one ordering; returns the parsed position or None."""
    verdict = await generation.complete_async(
        judge,
        [{"role": "user", "content": render_judge_prompt(message, response_1, response_2, traits)}],
    )
    return parse_answer(verdict)
