"""B2: the ``validity-v2a`` bridge — one set of responses, two judge instruments.

The revealed-preference audit found that base-model responses *self-label* far
more often than trained ones ("I'll go with Choice 1: playful …"), and that the
``paper-v1`` judge usually follows those declarations. That is a validity
threat, not a character signal. Work package 2 of the readiness doc fixes the
judge rather than the data: ``revealed-preference/validity-v2a-ignore-self-label``
is the same judge prompt plus a rubric line telling it to score only the
demonstrated manner.

Before v2a may carry any primary claim it has to be *bridged* to the frozen
replication instrument on evidence both can see. This module runs that bridge:

  1. **Freeze** — both prompts come verbatim from :mod:`octt.instruments`. This
     module never composes judge text of its own.
  2. **Select** — every detected self-label case in the bank, plus matched
     non-label controls, stratified by base/trained status and trait-pair
     relevance.
  3. **Rejudge** — under v1 and v2a. v1 verdicts already exist in the banked
     split cache (:mod:`octt.eval_cache`) and are reused for free; only the v2a
     column is ever paid for. The responder is never resampled.
  4. **Adjudicate** — write a *blinded* human/Fable slice and stop. Annotating
     is a human act; the runner will not simulate it. Re-running the same
     command picks the annotations up and finishes.
  5. **Report** — v1<->v2a bridge table, self-label concordance vs the matched
     non-label baseline, ordinary-case retention, response-length slope,
     disagreement examples, and the adoption gate.

Two boundaries this module is careful about:

**Instruments vs analysis** (CLAUDE.md). Judge text is an instrument: it is read
from the registry and rendered verbatim, never derived from anything here.
:mod:`octt.trait_profiles` *is* imported, but only to stratify the *selection*
(which pairs count as persona-relevant) — it never reaches a prompt. Because
selection still moves the reported numbers, the exact aligned/opposing sets and
their hash are stamped into every report; a profile edit is therefore visible
rather than silent.

**The bank is read-only.** The bridge never writes into the split cache it
reads; its own verdicts live in the output directory. A dry run additionally
writes every artifact to a ``.preview`` name, so stub verdicts can never be
mistaken for — or resumed as — banked evidence (same rule as ``gen-prompts``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import artifacts, eval_cache, evaluation, generation, instruments, models, trait_profiles
from .config import EvalConfig
from .manifest import stable_hash
from .tinker_client import TEACHER_SAMPLE_PRICE_USD_PER_MTOK, TinkerRuntime

logger = logging.getLogger(__name__)

BRIDGE_SCHEMA_VERSION = 1

V1_INSTRUMENT_ID = "revealed-preference/paper-v1"
V2A_INSTRUMENT_ID = "revealed-preference/validity-v2a-ignore-self-label"
INSTRUMENT_IDS: tuple[str, str] = (V1_INSTRUMENT_ID, V2A_INSTRUMENT_ID)

REPORT_JSON_NAME = "bridge_report.json"
REPORT_MD_NAME = "bridge_report.md"
VERDICTS_NAME = "verdicts.jsonl"
SELECTION_NAME = "selection.jsonl"
SLICE_NAME = "adjudication_slice.jsonl"
SLICE_README_NAME = "adjudication_slice.md"
ANNOTATED_NAME = "adjudication_annotated.jsonl"

#: Bump when the self-label detector's patterns change: it defines which rows
#: are "cases" and which are "controls", so two bridge tables are only
#: comparable when they were selected by the same detector.
SELF_LABEL_DETECTOR_VERSION = "self-label-detector-v1-latin-regex"

#: Bump when the stratification or matching rule changes.
SELECTION_VERSION = "bridge-selection-v1-status-x-relevance-1to1"

#: Bump when the elaboration lexicon below changes.
LENGTH_LEXICON_VERSION = "elaboration-lexicon-v1"

UNCLEAR = "unclear"

# --------------------------------------------------------------------------
# Self-label detection (SELECTION, not an instrument)
# --------------------------------------------------------------------------
#
# This is a screening heuristic over banked text. It never appears in a judge
# prompt, and no row is ever *dropped* because of it — dropping self-labelling
# rows is exactly the differential selection bias the readiness doc rejects.
# Its only job is to split the bank into "self-label cases" and "matched
# controls" so the two can be compared.
#
# KNOWN RECALL LIMIT: the patterns are Latin-script and English. Banked
# multilingual responses that self-label in another language read as controls,
# which biases the control group *toward* containing leakage and therefore
# makes the case-vs-control contrast conservative. This is recorded in every
# report as a caveat rather than papered over.

_CHOICE_MARKER = re.compile(
    r"\b(?:choice\s*[#:]?\s*(?:1|2|one|two)\b|my\s+choice\b|the\s+choice\s+i\b)",
    re.IGNORECASE,
)

_DECLARATION = re.compile(
    r"\bI(?:'ll|'ve|'m| will| have| am going to| shall| am)?\s+"
    r"(?:choose|chose|chosen|choosing|adopt|adopted|adopting|select|selected|"
    r"selecting|pick|picked|picking|go with|going with|opt for|opting for|"
    r"embody|embodying|embodied)\b",
    re.IGNORECASE,
)

TAG_CHOICE_MARKER = "choice_marker"
TAG_DECLARATION = "declaration"
TAG_TRAIT_WORD = "trait_word"


@dataclass(frozen=True)
class SelfLabel:
    """What a response leaked about its own trait choice."""

    tags: tuple[str, ...]
    declared_trait: str | None

    @property
    def detected(self) -> bool:
        return bool(self.tags)


def detect_self_label(response: str, a: str, b: str) -> SelfLabel:
    """Screen one response for explicit statements about its trait choice.

    Three independent signals, any of which marks the row a case:
    ``choice_marker`` (the response names "Choice 1"/"Choice 2"),
    ``declaration`` (an "I chose / I'll adopt …" phrase), and ``trait_word``
    (one of the two candidate words appears verbatim). ``declared_trait`` is
    resolved only when exactly one candidate word is present — a response that
    names both has not declared anything unambiguously.
    """
    tags: list[str] = []
    if _CHOICE_MARKER.search(response):
        tags.append(TAG_CHOICE_MARKER)
    if _DECLARATION.search(response):
        tags.append(TAG_DECLARATION)
    hits = [t for t in (a, b) if re.search(rf"\b{re.escape(t)}\b", response, re.IGNORECASE)]
    if hits:
        tags.append(TAG_TRAIT_WORD)
    declared = hits[0] if len(hits) == 1 else None
    return SelfLabel(tuple(tags), declared)


# --------------------------------------------------------------------------
# Trait-pair relevance (SELECTION, analysis curation — stamped, never rendered)
# --------------------------------------------------------------------------

RELEVANCE_BOTH = "relevant"  # both candidates are persona-curated traits
RELEVANCE_ONE = "mixed"  # exactly one is
RELEVANCE_NONE = "irrelevant"  # neither is
RELEVANCE_UNKNOWN = "unknown"  # no curated profile for this persona


def pair_relevance(a: str, b: str, curated: frozenset[str]) -> str:
    """Stratum for one ordered trait pair against a persona's curated traits."""
    if not curated:
        return RELEVANCE_UNKNOWN
    hits = (a in curated) + (b in curated)
    return (RELEVANCE_NONE, RELEVANCE_ONE, RELEVANCE_BOTH)[hits]


def relevance_profile(persona: str) -> tuple[frozenset[str], dict[str, Any]]:
    """Curated traits used to stratify selection, plus their provenance stamp.

    :mod:`octt.trait_profiles` is analysis curation and gets revised as
    constitutions are audited. It cannot reach the judge prompt from here, but
    it *does* decide which cases are compared, so the exact sets and their hash
    travel with every result: a later reader can tell whether two bridge tables
    were selected against the same curation.
    """
    prof = trait_profiles.profile(persona)
    if prof is None:
        return frozenset(), {
            "persona": persona,
            "aligned": [],
            "opposing": [],
            "traits_hash": None,
            "note": "no curated profile; relevance stratification disabled",
        }
    aligned, opposing = list(prof.aligned), list(prof.opposing)
    return frozenset(aligned) | frozenset(opposing), {
        "persona": persona,
        "aligned": aligned,
        "opposing": opposing,
        "traits_hash": stable_hash([aligned, opposing]),
        "note": "analysis curation: stratifies selection only, never rendered into a prompt",
    }


# --------------------------------------------------------------------------
# Length lexicon (ANALYSIS) — used for the response-length slope
# --------------------------------------------------------------------------
#
# The v2a rubric also tells the judge to ignore "response length or detail".
# To measure whether it did, we need pairs where length has an obvious pull:
# pairs in which exactly one candidate is an elaboration-coded trait. The
# outcome is "did the elaboration-coded trait win", regressed on log length.
# Members are drawn from the frozen App G pool.

ELABORATION_TRAITS: frozenset[str] = frozenset(
    {
        "elaborate", "verbose", "pedantic", "academic", "scholarly",
        "philosophical", "theoretical", "nuanced", "systematic", "methodical",
        "analytical", "technical", "holistic", "abstract",
    }
)

#: Minimum points before a slope is reported at all (an OLS line through five
#: points is noise, and a noisy slope inside an adoption gate is worse than no
#: slope).
MIN_SLOPE_N = 8


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

GROUP_CASE = "self_label"
GROUP_CONTROL = "control"

STATUS_BASE = "base"
STATUS_TRAINED = "trained"


@dataclass(frozen=True)
class Candidate:
    """Screening record for one banked response — deliberately without its text.

    The banked response file is hundreds of megabytes; selection only needs
    metadata, so pass one keeps lengths and tags and throws the text away. Pass
    two rehydrates the few hundred rows that were actually selected.
    """

    response_key: str
    response_hash: str
    model_tag: str
    condition: str
    a: str
    b: str
    status: str
    relevance: str
    tags: tuple[str, ...]
    declared_trait: str | None
    length_chars: int

    @property
    def stratum(self) -> str:
        return f"{self.status}/{self.relevance}"

    @property
    def is_case(self) -> bool:
        return bool(self.tags)


@dataclass(frozen=True)
class BridgeCase:
    """A selected response, rehydrated with its text and its group label."""

    candidate: Candidate
    group: str
    response: str
    prompt: str

    @property
    def item_id(self) -> str:
        """Opaque id for the blinded slice (no model, group, or verdict in it)."""
        return stable_hash(["bridge-item", self.candidate.response_key], length=16)

    @property
    def model_id(self) -> str:
        return self.candidate.model_tag.split("@", 1)[0]


def _status_for(model_tag: str) -> str:
    return STATUS_BASE if model_tag.endswith("@base") else STATUS_TRAINED


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file, strictly (a corrupt line is data loss, not a skip)."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: corrupt JSONL row: {exc}") from exc


def scan_bank(
    responses_path: Path,
    *,
    curated: frozenset[str],
    condition: str | None = evaluation.DEFAULT_CONDITION,
    model_tag_filter: str | None = None,
) -> list[Candidate]:
    """Pass one: screen every usable banked response into a :class:`Candidate`.

    ``condition=None`` keeps every embodiment variant. Later rows win on a
    duplicated response key, matching the split cache's own resume semantics.
    """
    seen: dict[str, Candidate] = {}
    for row in _iter_jsonl(Path(responses_path)):
        if not eval_cache.response_usable(row):
            continue
        if condition is not None and row.get("condition") != condition:
            continue
        model_tag = row["model_tag"]
        if model_tag_filter and model_tag_filter not in model_tag:
            continue
        a, b, response = row["a"], row["b"], row["response"]
        label = detect_self_label(response, a, b)
        seen[row["key"]] = Candidate(
            response_key=row["key"],
            response_hash=row["response_hash"],
            model_tag=model_tag,
            condition=row.get("condition", evaluation.DEFAULT_CONDITION),
            a=a,
            b=b,
            status=_status_for(model_tag),
            relevance=pair_relevance(a, b, curated),
            tags=label.tags,
            declared_trait=label.declared_trait,
            length_chars=len(response),
        )
    return list(seen.values())


@dataclass(frozen=True)
class Selection:
    """Which banked responses the bridge will rejudge, and how they were picked."""

    groups: dict[str, str]  # response_key -> GROUP_CASE / GROUP_CONTROL
    strata: list[dict[str, Any]]
    universe: dict[str, int]
    caveats: list[str]
    seed: int
    max_per_stratum: int | None

    def stamp(self) -> dict[str, Any]:
        return {
            "selection_version": SELECTION_VERSION,
            "detector_version": SELF_LABEL_DETECTOR_VERSION,
            "seed": self.seed,
            "max_per_stratum": self.max_per_stratum,
            "universe": dict(self.universe),
            "strata": list(self.strata),
            "caveats": list(self.caveats),
        }


def select_cases(
    candidates: Sequence[Candidate],
    *,
    max_per_stratum: int | None = 50,
    seed: int = 0,
) -> Selection:
    """Every detected self-label case, plus 1:1 matched non-label controls.

    Matching is within ``(base|trained) x pair-relevance`` strata: a stratum's
    controls are drawn from the same checkpoint status and the same kind of
    trait pair, so a concordance difference between cases and controls cannot
    be explained by either. Selection is deterministic in ``seed`` (the stratum
    name is folded into the per-stratum RNG, so raising
    ``max_per_stratum`` extends a stratum rather than reshuffling every other
    one). A stratum with fewer controls than cases contributes what it has and
    records the shortfall.
    """
    by_stratum: dict[str, dict[str, list[Candidate]]] = {}
    for cand in candidates:
        bucket = by_stratum.setdefault(cand.stratum, {GROUP_CASE: [], GROUP_CONTROL: []})
        bucket[GROUP_CASE if cand.is_case else GROUP_CONTROL].append(cand)

    groups: dict[str, str] = {}
    strata: list[dict[str, Any]] = []
    caveats: list[str] = []
    for stratum in sorted(by_stratum):
        bucket = by_stratum[stratum]
        cases = sorted(bucket[GROUP_CASE], key=lambda c: c.response_key)
        controls = sorted(bucket[GROUP_CONTROL], key=lambda c: c.response_key)
        if not cases:
            continue
        rng = random.Random(f"{seed}|{stratum}")
        take = len(cases) if max_per_stratum is None else min(max_per_stratum, len(cases))
        chosen_cases = sorted(rng.sample(cases, take), key=lambda c: c.response_key)
        n_controls = min(take, len(controls))
        chosen_controls = sorted(rng.sample(controls, n_controls), key=lambda c: c.response_key)
        for cand in chosen_cases:
            groups[cand.response_key] = GROUP_CASE
        for cand in chosen_controls:
            groups[cand.response_key] = GROUP_CONTROL
        if n_controls < take:
            caveats.append(
                f"stratum {stratum}: only {n_controls} non-label controls available "
                f"for {take} self-label cases (unmatched)"
            )
        strata.append(
            {
                "stratum": stratum,
                "available_cases": len(cases),
                "available_controls": len(controls),
                "selected_cases": take,
                "selected_controls": n_controls,
            }
        )

    universe = {
        "responses_screened": len(candidates),
        "self_label_detected": sum(c.is_case for c in candidates),
        "selected": len(groups),
    }
    if not groups:
        caveats.append("no self-label cases detected in the screened bank")
    return Selection(groups, strata, universe, caveats, seed, max_per_stratum)


def hydrate_cases(
    responses_path: Path,
    selection: Selection,
    relevance: dict[str, str] | None = None,
) -> list[BridgeCase]:
    """Pass two: reload only the selected rows, this time keeping their text.

    ``relevance`` carries the stratum labels computed in pass one (they depend
    on the persona curation, which pass two does not see).
    """
    wanted = selection.groups
    relevance = relevance or {}
    by_key: dict[str, BridgeCase] = {}
    for row in _iter_jsonl(Path(responses_path)):
        group = wanted.get(row.get("key"))
        if group is None:
            continue
        label = detect_self_label(row["response"], row["a"], row["b"])
        by_key[row["key"]] = BridgeCase(
            candidate=Candidate(
                response_key=row["key"],
                response_hash=row["response_hash"],
                model_tag=row["model_tag"],
                condition=row.get("condition", evaluation.DEFAULT_CONDITION),
                a=row["a"],
                b=row["b"],
                status=_status_for(row["model_tag"]),
                relevance=relevance.get(row["key"], RELEVANCE_UNKNOWN),
                tags=label.tags,
                declared_trait=label.declared_trait,
                length_chars=len(row["response"]),
            ),
            group=group,
            response=row["response"],
            prompt=row.get("prompt", ""),
        )
    return [by_key[k] for k in sorted(wanted) if k in by_key]


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    winner: str | None
    source: str  # "bank" | "cache" | "live" | "offline-stub"
    skip_reason: str | None = None


SOURCE_BANK = "bank"
SOURCE_CACHE = "cache"
SOURCE_LIVE = "live"
SOURCE_STUB = "offline-stub"


def _verdict_key(instrument_id: str, response_hash: str, a: str, b: str, j_tag: str) -> str:
    """Judgment identity, shared with the banked split cache for the v1 column.

    Reuses :func:`octt.eval_cache.judgment_key` unchanged so a v1 lookup here
    hits exactly the row a paper-v1 eval wrote — that reuse is what makes the
    v1 column free.
    """
    inst = instruments.get(instrument_id)
    return eval_cache.judgment_key(
        response_hash, a, b, j_tag, inst.parser or "", judge_instrument=instrument_id
    )


def _verdict_row(
    key: str,
    *,
    instrument_id: str,
    case: BridgeCase,
    j_tag: str,
    winner: str | None,
    verdict_text: str | None,
    skip_reason: str | None,
    judge_attempts: int,
    discarded: list[str],
    source: str,
) -> dict[str, Any]:
    inst = instruments.get(instrument_id)
    return {
        "key": key,
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "instrument_id": instrument_id,
        "instrument_hash": inst.content_hash,
        "parser": inst.parser,
        "judge_tag": j_tag,
        "response_key": case.candidate.response_key,
        "response_hash": case.candidate.response_hash,
        "a": case.candidate.a,
        "b": case.candidate.b,
        "winner_trait": winner,
        "verdict": verdict_text,
        "skip_reason": skip_reason,
        "judge_attempts": judge_attempts,
        "discarded_verdicts": discarded,
        "source": source,
    }


def _offline_verdict(instrument_id: str, response: str, a: str, b: str) -> str:
    """Deterministic offline stand-in for a judge call.

    Folding the instrument id into the hash makes the two columns disagree
    about half the time, which exercises every branch of the bridge report
    offline. These verdicts are plumbing, never findings: dry-run artifacts are
    written to ``.preview`` names and stamped ``execution_mode: dry-run``.
    """
    digest = hashlib.sha256(f"{instrument_id}|{response}|{a}|{b}".encode()).hexdigest()
    return a if int(digest, 16) % 2 == 0 else b


def _judge_messages(instrument_id: str, responder_name: str, case: BridgeCase) -> list[dict]:
    """Render the judge conversation verbatim from the registered instrument."""
    inst = instruments.get(instrument_id)
    return [
        {"role": "system", "content": inst.prompts["judge_system"].format(name=responder_name)},
        {
            "role": "user",
            "content": inst.prompts["judge_user"].format(
                message=case.response, a=case.candidate.a, b=case.candidate.b
            ),
        },
    ]


def load_banked_v1(judgments_path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """Stream the banked judgment cache, keeping only the keys the bridge needs."""
    found: dict[str, dict[str, Any]] = {}
    if not Path(judgments_path).is_file():
        return found
    for row in _iter_jsonl(Path(judgments_path)):
        key = row.get("key")
        if key in wanted:
            found[key] = row
    return found


async def _judge_pending(
    pending: list[tuple[str, BridgeCase, str]],
    judge: generation.Sampler,
    j_tag: str,
    cache_path: Path,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Sample the missing verdicts, bounded-concurrently, appending as they land.

    Resampling on an unparseable verdict follows the paper-v1 protocol exactly
    (:data:`octt.evaluation._JUDGE_VERDICT_ATTEMPTS` draws, parsed by
    :func:`octt.evaluation.parse_judge_verdict`) — the bridge changes the judge
    *prompt*, and nothing else, so a difference between columns cannot come
    from a difference in parsing.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    rows: dict[str, dict[str, Any]] = {}

    async def one(item: tuple[str, BridgeCase, str]) -> None:
        key, case, instrument_id = item
        responder_name = models.assistant_name(case.model_id)
        messages = _judge_messages(instrument_id, responder_name, case)
        winner: str | None = None
        verdict_text: str | None = None
        discarded: list[str] = []
        attempts = 0
        async with sem:
            while attempts < evaluation._JUDGE_VERDICT_ATTEMPTS:
                if verdict_text is not None:
                    discarded.append(verdict_text)
                verdict_text = await generation.complete_async(judge, messages)
                attempts += 1
                winner = evaluation.parse_judge_verdict(
                    verdict_text, case.candidate.a, case.candidate.b
                )
                if winner is not None:
                    break
        row = _verdict_row(
            key,
            instrument_id=instrument_id,
            case=case,
            j_tag=j_tag,
            winner=winner,
            verdict_text=verdict_text,
            skip_reason=None if winner is not None else "unparseable_verdict",
            judge_attempts=attempts,
            discarded=discarded,
            source=SOURCE_LIVE,
        )
        rows[key] = row
        async with write_lock:
            artifacts.append_jsonl(cache_path, row)

    await asyncio.gather(*(one(item) for item in pending))
    return rows


def resolve_verdicts(
    cases: Sequence[BridgeCase],
    *,
    banked: dict[str, dict[str, Any]],
    cache_path: Path,
    j_tag: str,
    runtime: TinkerRuntime,
    judge_model: str,
    config: EvalConfig,
    offline: bool,
    concurrency: int,
) -> tuple[dict[tuple[str, str], Verdict], dict[str, dict[str, int]]]:
    """Fill both verdict columns, paying only for what is genuinely missing.

    Lookup order per (case, instrument): the banked split cache (v1 only, and
    only because the key function is shared), then the bridge's own cache in
    the output directory, then a judge call. Returns the verdicts keyed by
    ``(response_key, instrument_id)`` plus a per-instrument accounting of where
    each one came from — that accounting is what the dry run reports as its
    spend projection.
    """
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        for row in _iter_jsonl(cache_path):
            cached[row["key"]] = row

    stats = {i: {"bank": 0, "cache": 0, "deduped": 0, "new": 0} for i in INSTRUMENT_IDS}
    verdicts: dict[tuple[str, str], Verdict] = {}
    # A verdict is identified by response *content*, so two selected rows with
    # byte-identical responses (a base and a trained checkpoint that answered
    # the same, say) are one judge call whose verdict both rows share — the
    # same dedupe the eval itself does. Fan the result back out to every row.
    pending: dict[str, tuple[BridgeCase, str]] = {}
    sharing: dict[str, list[str]] = {}
    for case in cases:
        for instrument_id in INSTRUMENT_IDS:
            key = _verdict_key(
                instrument_id,
                case.candidate.response_hash,
                case.candidate.a,
                case.candidate.b,
                j_tag,
            )
            row = banked.get(key)
            source = SOURCE_BANK
            if row is None:
                row = cached.get(key)
                source = SOURCE_CACHE
            if row is not None:
                stats[instrument_id]["bank" if source == SOURCE_BANK else "cache"] += 1
                verdicts[(case.candidate.response_key, instrument_id)] = Verdict(
                    row.get("winner_trait"), source, row.get("skip_reason")
                )
                continue
            if key in pending:
                stats[instrument_id]["deduped"] += 1
            else:
                stats[instrument_id]["new"] += 1
                pending[key] = (case, instrument_id)
            sharing.setdefault(key, []).append(case.candidate.response_key)

    if pending and offline:
        for key, (case, instrument_id) in pending.items():
            winner = _offline_verdict(
                instrument_id, case.response, case.candidate.a, case.candidate.b
            )
            artifacts.append_jsonl(
                cache_path,
                _verdict_row(
                    key,
                    instrument_id=instrument_id,
                    case=case,
                    j_tag=j_tag,
                    winner=winner,
                    verdict_text=None,
                    skip_reason=None,
                    judge_attempts=0,
                    discarded=[],
                    source=SOURCE_STUB,
                ),
            )
            for response_key in sharing[key]:
                verdicts[(response_key, instrument_id)] = Verdict(winner, SOURCE_STUB)
    elif pending:
        judge = generation.make_sampler(
            runtime,
            judge_model,
            tag="bridge-judge",
            max_tokens=config.judge_max_tokens,
            temperature=config.judge_temperature,
            top_p=config.judge_top_p,
        )
        rows = asyncio.run(
            _judge_pending(
                [(key, case, instrument) for key, (case, instrument) in pending.items()],
                judge,
                j_tag,
                cache_path,
                concurrency,
            )
        )
        for key, (_case, instrument_id) in pending.items():
            row = rows[key]
            for response_key in sharing[key]:
                verdicts[(response_key, instrument_id)] = Verdict(
                    row["winner_trait"], SOURCE_LIVE, row["skip_reason"]
                )
    return verdicts, stats


def project_judge_cost(stats: dict[str, dict[str, int]], judge_model: str, config: EvalConfig,
                       mean_response_chars: float) -> dict[str, Any]:
    """Rough spend projection for the calls the bridge would still have to make.

    Same convention as ``tinker_client``'s eval lines (prefill = response plus
    template, sampled = the judge's token cap), with the teacher fallback price
    when the judge is not in the pinned rate card. An estimate, never a bill —
    ``octt spend`` reads what Tinker actually invoiced.
    """
    spec = models.CANDIDATES.get(judge_model)
    sample_price = (spec.price_sample if spec else None) or TEACHER_SAMPLE_PRICE_USD_PER_MTOK
    prefill_price = (spec.price_prefill if spec else None) or sample_price
    calls = sum(s["new"] for s in stats.values())
    prefill_tokens = calls * (mean_response_chars / 4 + 256)
    sampled_tokens = calls * config.judge_max_tokens
    usd = (prefill_tokens * prefill_price + sampled_tokens * sample_price) / 1_000_000
    return {
        "judge_calls": calls,
        "estimated_prefill_tokens": round(prefill_tokens),
        "estimated_sampled_tokens": round(sampled_tokens),
        "estimated_usd": round(usd, 4),
        "note": "estimate at one draw per call; unparseable verdicts resample up to "
        f"{evaluation._JUDGE_VERDICT_ATTEMPTS} times",
    }


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _winners(case: BridgeCase, verdicts: dict[tuple[str, str], Verdict]) -> tuple[str | None, ...]:
    key = case.candidate.response_key
    return tuple(
        (verdicts.get((key, i)).winner if verdicts.get((key, i)) else None)
        for i in INSTRUMENT_IDS
    )


def bridge_table(
    cases: Sequence[BridgeCase], verdicts: dict[tuple[str, str], Verdict]
) -> dict[str, Any]:
    """v1 <-> v2a concordance over a set of cases.

    ``concordance`` is agreement over the rows *both* instruments scored;
    single-sided and doubly-skipped rows are reported separately rather than
    folded in, so a v2a that simply refuses to answer cannot look concordant.
    """
    agree = disagree = v1_only = v2a_only = neither = 0
    for case in cases:
        w1, w2 = _winners(case, verdicts)
        if w1 is not None and w2 is not None:
            agree += w1 == w2
            disagree += w1 != w2
        elif w1 is not None:
            v1_only += 1
        elif w2 is not None:
            v2a_only += 1
        else:
            neither += 1
    both = agree + disagree
    return {
        "n": len(cases),
        "both_scored": both,
        "agree": agree,
        "disagree": disagree,
        "v1_only": v1_only,
        "v2a_only": v2a_only,
        "neither_scored": neither,
        "concordance": agree / both if both else None,
        "v1_parse_rate": (agree + disagree + v1_only) / len(cases) if cases else None,
        "v2a_parse_rate": (agree + disagree + v2a_only) / len(cases) if cases else None,
    }


def declared_trait_following(
    cases: Sequence[BridgeCase], verdicts: dict[tuple[str, str], Verdict]
) -> dict[str, Any]:
    """How often each judge's winner equals the trait the response declared.

    Only self-label cases with an unambiguous declared trait are scored. Under
    a judge that follows declarations this sits near 1.0; a judge that scores
    demonstrated manner should sit near the 0.5 chance reference (the response
    may of course also *demonstrate* what it declared, so 0.5 is a reference,
    not a target).
    """
    scored = [c for c in cases if c.group == GROUP_CASE and c.candidate.declared_trait]
    out: dict[str, Any] = {"n": len(scored), "chance_reference": 0.5}
    for instrument_id in INSTRUMENT_IDS:
        hits = total = 0
        for case in scored:
            verdict = verdicts.get((case.candidate.response_key, instrument_id))
            if verdict is None or verdict.winner is None:
                continue
            total += 1
            hits += verdict.winner == case.candidate.declared_trait
        out[instrument_id] = {
            "scored": total,
            "follow_rate": hits / total if total else None,
        }
    return out


def _log_len(case: BridgeCase) -> float:
    return math.log1p(case.candidate.length_chars)


def _ols_slope(points: Sequence[tuple[float, float]]) -> float | None:
    n = len(points)
    if n < MIN_SLOPE_N:
        return None
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    denom = sum((x - mx) ** 2 for x, _ in points)
    if denom <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in points) / denom


def length_slopes(
    cases: Sequence[BridgeCase], verdicts: dict[tuple[str, str], Verdict]
) -> dict[str, Any]:
    """Residual response-length dependence, two ways.

    ``elaboration_slope`` (per instrument): over pairs where exactly one
    candidate is elaboration-coded, the OLS slope of "the elaboration-coded
    trait won" on log response length. A judge that reads length as evidence
    of elaborateness has a positive slope; the v2a rubric asks for a smaller
    magnitude than v1.

    ``concordance_slope``: the slope of v1/v2a agreement on log length —
    whether the two instruments part company specifically on long responses.
    """
    out: dict[str, Any] = {"lexicon_version": LENGTH_LEXICON_VERSION, "min_n": MIN_SLOPE_N}
    for instrument_id in INSTRUMENT_IDS:
        points: list[tuple[float, float]] = []
        for case in cases:
            a, b = case.candidate.a, case.candidate.b
            coded = [t for t in (a, b) if t in ELABORATION_TRAITS]
            if len(coded) != 1:
                continue
            verdict = verdicts.get((case.candidate.response_key, instrument_id))
            if verdict is None or verdict.winner is None:
                continue
            points.append((_log_len(case), float(verdict.winner == coded[0])))
        out[instrument_id] = {
            "n": len(points),
            "elaboration_slope": _ols_slope(points),
        }
    agreement: list[tuple[float, float]] = []
    for case in cases:
        w1, w2 = _winners(case, verdicts)
        if w1 is None or w2 is None:
            continue
        agreement.append((_log_len(case), float(w1 == w2)))
    out["concordance_slope"] = {"n": len(agreement), "slope": _ols_slope(agreement)}
    return out


def disagreement_examples(
    cases: Sequence[BridgeCase],
    verdicts: dict[tuple[str, str], Verdict],
    *,
    limit: int = 12,
    excerpt_chars: int = 400,
) -> list[dict[str, Any]]:
    """Cases where the two instruments picked different traits, cases first."""
    rows = []
    for case in cases:
        w1, w2 = _winners(case, verdicts)
        if w1 is None or w2 is None or w1 == w2:
            continue
        rows.append(
            {
                "item_id": case.item_id,
                "group": case.group,
                "status": case.candidate.status,
                "relevance": case.candidate.relevance,
                "a": case.candidate.a,
                "b": case.candidate.b,
                "v1_winner": w1,
                "v2a_winner": w2,
                "tags": list(case.candidate.tags),
                "declared_trait": case.candidate.declared_trait,
                "length_chars": case.candidate.length_chars,
                "response_excerpt": " ".join(case.response.split())[:excerpt_chars],
            }
        )
    rows.sort(key=lambda r: (r["group"] != GROUP_CASE, r["item_id"]))
    return rows[:limit]


# --------------------------------------------------------------------------
# Blinded adjudication slice (step 4 — human work, never simulated)
# --------------------------------------------------------------------------


def build_adjudication_slice(
    cases: Sequence[BridgeCase],
    verdicts: dict[tuple[str, str], Verdict],
    *,
    size: int = 40,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """A blinded slice for a human (or Fable) to judge cold.

    Each item carries the response and the two candidate traits and nothing
    else: no model, no checkpoint status, no group label, no machine verdict,
    and the two traits are presented in an order derived from the item id
    rather than the banked pair order. Disagreements are prioritised (that is
    where an independent read is worth paying for) and the rest of the slice is
    filled with agreements so the annotator cannot infer the group from the
    fact that an item is in the file; the final order is shuffled.
    """
    disagreeing, agreeing = [], []
    for case in cases:
        w1, w2 = _winners(case, verdicts)
        if w1 is None or w2 is None:
            continue
        (disagreeing if w1 != w2 else agreeing).append(case)
    rng = random.Random(f"{seed}|adjudication")
    picked = list(disagreeing[: max(0, size)])
    if len(picked) < size and agreeing:
        fill = min(size - len(picked), len(agreeing))
        picked += rng.sample(agreeing, fill)
    rng.shuffle(picked)
    items = []
    for case in picked:
        pair = (case.candidate.a, case.candidate.b)
        flip = int(stable_hash(["blind-order", case.item_id], length=8), 16) % 2
        one, two = (pair[1], pair[0]) if flip else pair
        items.append(
            {
                "item_id": case.item_id,
                "response": case.response,
                "choice_1": one,
                "choice_2": two,
                "human_winner": None,
                "note": "",
            }
        )
    return items


SLICE_INSTRUCTIONS = """# Blinded adjudication slice

The bridge has paused. Step 4 of the v2a bridge is a human read, and nothing
here will do it for you.

`{slice_file}` holds {n} blinded items. Each one is a banked response plus the
two candidate traits it was produced under, in randomised order. You are not
told which model produced it, whether it self-labels, or what either judge
said.

## What to do

1. Copy the file:

       cp {slice_file} {annotated_file}

2. For every row in the copy, set `"human_winner"` to whichever of
   `choice_1` / `choice_2` the response is actually *written in the manner of*.
   Judge the demonstrated manner only: ignore any statement the response makes
   about choosing or adopting a trait, and ignore length or detail. Use
   `"{unclear}"` when the response genuinely demonstrates neither. `"note"` is
   free text and optional.

3. Re-run the identical command. It will pick up the annotations, skip every
   judge call it has already paid for, and write the full bridge report.

Leaving a row with `"human_winner": null` is not an answer — the resume step
will refuse the file and tell you which ids are missing.
"""


def load_annotations(path: Path, items: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Read an annotated slice, refusing anything partial or off-schema.

    Returns ``item_id -> winner`` where winner is one of the item's two traits
    or :data:`UNCLEAR`. Every item in the slice must be answered: a half-filled
    file silently shrinking the human sample is exactly the kind of quiet scope
    loss that makes an agreement number meaningless.
    """
    expected = {item["item_id"]: item for item in items}
    answers: dict[str, str] = {}
    for row in _iter_jsonl(Path(path)):
        item_id = row.get("item_id")
        item = expected.get(item_id)
        if item is None:
            raise ValueError(
                f"{path}: item_id {item_id!r} is not in the current slice; the selection "
                "changed since the slice was written — delete the annotations or restore "
                "the original selection arguments"
            )
        winner = (row.get("human_winner") or "").strip()
        allowed = {item["choice_1"], item["choice_2"], UNCLEAR}
        if winner not in allowed:
            raise ValueError(
                f"{path}: item {item_id} has human_winner={row.get('human_winner')!r}; "
                f"expected one of {sorted(allowed)}"
            )
        answers[item_id] = winner
    missing = sorted(set(expected) - set(answers))
    if missing:
        raise ValueError(
            f"{path}: {len(missing)} of {len(expected)} slice items are unannotated "
            f"(first missing: {missing[:3]})"
        )
    return answers


def human_agreement(
    cases: Sequence[BridgeCase],
    verdicts: dict[tuple[str, str], Verdict],
    answers: dict[str, str],
) -> dict[str, Any]:
    """Agreement of each instrument with the blinded human reads."""
    by_item = {case.item_id: case for case in cases}
    scored = {i: [0, 0] for i in INSTRUMENT_IDS}  # hits, total
    unclear = 0
    for item_id, winner in answers.items():
        case = by_item.get(item_id)
        if case is None:
            continue
        if winner == UNCLEAR:
            unclear += 1
            continue
        for instrument_id in INSTRUMENT_IDS:
            verdict = verdicts.get((case.candidate.response_key, instrument_id))
            if verdict is None or verdict.winner is None:
                continue
            scored[instrument_id][1] += 1
            scored[instrument_id][0] += verdict.winner == winner
    out: dict[str, Any] = {
        "status": "annotated",
        "annotated": len(answers),
        "unclear": unclear,
    }
    for instrument_id, (hits, total) in scored.items():
        out[instrument_id] = {"scored": total, "agreement": hits / total if total else None}
    return out


# --------------------------------------------------------------------------
# Adoption gate
# --------------------------------------------------------------------------

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True)
class AdoptionGate:
    """Thresholds for the readiness doc's four adoption criteria.

    Pinned here (and stamped into every report) because a gate whose thresholds
    move after the numbers are known is not a gate. ``version`` is what a later
    reader compares.
    """

    version: str = "bridge-gate-v1"
    #: |concordance(self-label) - concordance(matched controls)|
    self_label_concordance_tolerance: float = 0.10
    #: v1/v2a agreement required on ordinary (non-label) cases
    min_ordinary_retention: float = 0.90
    #: blinded-human agreement required of v2a...
    min_human_agreement: float = 0.75
    #: ...and how far below v1's human agreement v2a may fall
    max_human_agreement_deficit: float = 0.05
    #: how much larger |v2a elaboration slope| may be than |v1|'s
    length_slope_tolerance: float = 0.02


def evaluate_gate(
    gate: AdoptionGate,
    *,
    case_table: dict[str, Any],
    control_table: dict[str, Any],
    slopes: dict[str, Any],
    human: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score each adoption criterion; unknown never counts as passed."""
    criteria: list[dict[str, Any]] = []

    def add(name: str, measured: Any, threshold: Any, ok: bool | None, detail: str) -> None:
        criteria.append(
            {
                "criterion": name,
                "measured": measured,
                "threshold": threshold,
                "status": GATE_PASS if ok else (GATE_INCOMPLETE if ok is None else GATE_FAIL),
                "detail": detail,
            }
        )

    case_conc, control_conc = case_table["concordance"], control_table["concordance"]
    delta = (
        abs(case_conc - control_conc)
        if case_conc is not None and control_conc is not None
        else None
    )
    add(
        "self_label_concordance_near_control_baseline",
        delta,
        gate.self_label_concordance_tolerance,
        None if delta is None else delta <= gate.self_label_concordance_tolerance,
        "|v1<->v2a concordance on self-label cases - concordance on matched controls|",
    )

    retention = control_table["concordance"]
    add(
        "ordinary_case_retention",
        retention,
        gate.min_ordinary_retention,
        None if retention is None else retention >= gate.min_ordinary_retention,
        "v2a still recovers v1's verdict on clearly demonstrated (non-label) cases",
    )

    if human is None or human.get("status") != "annotated":
        add("human_agreement", None, gate.min_human_agreement, None,
            "blinded adjudication slice not annotated yet")
    else:
        v1_agree = human[V1_INSTRUMENT_ID]["agreement"]
        v2_agree = human[V2A_INSTRUMENT_ID]["agreement"]
        if v2_agree is None:
            ok: bool | None = None
        else:
            ok = v2_agree >= gate.min_human_agreement and (
                v1_agree is None or v2_agree >= v1_agree - gate.max_human_agreement_deficit
            )
        add("human_agreement", v2_agree, gate.min_human_agreement, ok,
            f"v2a vs blinded humans (v1 = {v1_agree})")

    s1 = slopes[V1_INSTRUMENT_ID]["elaboration_slope"]
    s2 = slopes[V2A_INSTRUMENT_ID]["elaboration_slope"]
    if s1 is None or s2 is None:
        length_ok: bool | None = None
    else:
        length_ok = abs(s2) <= abs(s1) + gate.length_slope_tolerance
    add("residual_length_dependence", s2, s1,
        length_ok, "|v2a elaboration slope| must not exceed |v1|'s (plus tolerance)")

    statuses = [c["status"] for c in criteria]
    overall = (
        GATE_FAIL if GATE_FAIL in statuses
        else GATE_INCOMPLETE if GATE_INCOMPLETE in statuses
        else GATE_PASS
    )
    return {"version": gate.version, "overall": overall, "criteria": criteria}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

STATUS_COMPLETE = "complete"
STATUS_PAUSED = "paused_for_adjudication"


@dataclass
class BridgeOutcome:
    status: str
    report: dict[str, Any]
    out_dir: Path
    report_path: Path
    slice_path: Path | None = None
    annotated_path: Path | None = None
    messages: list[str] = field(default_factory=list)


def _path(out_dir: Path, name: str, offline: bool) -> Path:
    """Dry-run artifacts get a ``.preview`` name so stubs never pose as evidence."""
    if not offline:
        return out_dir / name
    stem, dot, ext = name.partition(".")
    return out_dir / f"{stem}.preview{dot}{ext}"


def run_bridge(
    *,
    split_cache_dir: Path,
    out_dir: Path,
    persona: str,
    runtime: TinkerRuntime,
    judge_model: str = models.TEACHER_MODEL,
    config: EvalConfig | None = None,
    condition: str | None = evaluation.DEFAULT_CONDITION,
    model_tag_filter: str | None = None,
    max_per_stratum: int | None = 50,
    seed: int = 0,
    slice_size: int = 40,
    adjudicate: bool = True,
    gate: AdoptionGate | None = None,
    offline: bool = False,
    concurrency: int = evaluation.DEFAULT_EVAL_CONCURRENCY,
) -> BridgeOutcome:
    """Run the full bridge, pausing once for the blinded human slice.

    Reads the banked split cache at ``split_cache_dir`` (never writes to it),
    writes everything it produces under ``out_dir``, and returns either
    :data:`STATUS_PAUSED` (the slice is waiting for annotations) or
    :data:`STATUS_COMPLETE`. Re-running with identical arguments after
    annotating resumes: selection is deterministic in ``seed`` and every
    verdict already paid for is cached.
    """
    cfg = config or EvalConfig()
    gate = gate or AdoptionGate()
    offline = offline or runtime.config.dry_run
    bank = Path(split_cache_dir)
    out_dir = Path(out_dir)
    responses_path = bank / eval_cache.RESPONSES_NAME
    judgments_path = bank / eval_cache.JUDGMENTS_NAME
    if not responses_path.is_file():
        raise FileNotFoundError(f"no banked responses at {responses_path}")
    if out_dir.resolve() == bank.resolve():
        raise ValueError(
            "refusing to write bridge output into the banked cache directory; "
            "the bank is read-only evidence"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    curated, profile_stamp = relevance_profile(persona)
    candidates = scan_bank(
        responses_path,
        curated=curated,
        condition=condition,
        model_tag_filter=model_tag_filter,
    )
    selection = select_cases(candidates, max_per_stratum=max_per_stratum, seed=seed)
    cases = hydrate_cases(
        responses_path,
        selection,
        {c.response_key: c.relevance for c in candidates},
    )
    artifacts.write_jsonl_atomic(
        _path(out_dir, SELECTION_NAME, offline),
        (
            {
                "item_id": c.item_id,
                "response_key": c.candidate.response_key,
                "response_hash": c.candidate.response_hash,
                "group": c.group,
                "status": c.candidate.status,
                "relevance": c.candidate.relevance,
                "condition": c.candidate.condition,
                "model_tag": c.candidate.model_tag,
                "a": c.candidate.a,
                "b": c.candidate.b,
                "tags": list(c.candidate.tags),
                "declared_trait": c.candidate.declared_trait,
                "length_chars": c.candidate.length_chars,
            }
            for c in cases
        ),
    )

    j_tag = eval_cache.judge_only_tag(
        judge_model, cfg.judge_temperature, cfg.judge_top_p, cfg.judge_max_tokens
    )
    wanted_v1 = {
        _verdict_key(
            V1_INSTRUMENT_ID, c.candidate.response_hash, c.candidate.a, c.candidate.b, j_tag
        )
        for c in cases
    }
    banked = load_banked_v1(judgments_path, wanted_v1)
    cache_path = _path(out_dir, VERDICTS_NAME, offline)
    verdicts, call_stats = resolve_verdicts(
        cases,
        banked=banked,
        cache_path=cache_path,
        j_tag=j_tag,
        runtime=runtime,
        judge_model=judge_model,
        config=cfg,
        offline=offline,
        concurrency=concurrency,
    )

    case_rows = [c for c in cases if c.group == GROUP_CASE]
    control_rows = [c for c in cases if c.group == GROUP_CONTROL]
    overall_table = bridge_table(cases, verdicts)
    case_table = bridge_table(case_rows, verdicts)
    control_table = bridge_table(control_rows, verdicts)
    slopes = length_slopes(cases, verdicts)

    slice_items = build_adjudication_slice(cases, verdicts, size=slice_size, seed=seed)
    slice_path = _path(out_dir, SLICE_NAME, offline)
    annotated_path = _path(out_dir, ANNOTATED_NAME, offline)
    artifacts.write_jsonl_atomic(slice_path, slice_items)

    messages: list[str] = []
    human: dict[str, Any] | None
    status = STATUS_COMPLETE
    if not adjudicate:
        human = {"status": "skipped", "reason": "--no-adjudication"}
        messages.append(
            "adjudication SKIPPED: the human-agreement criterion cannot be scored, so "
            "the gate can never read PASS from this run"
        )
    elif annotated_path.is_file():
        human = human_agreement(cases, verdicts, load_annotations(annotated_path, slice_items))
    elif not slice_items:
        human = {"status": "unavailable", "reason": "no doubly-scored cases to adjudicate"}
        messages.append("adjudication slice is empty: no case was scored by both instruments")
    else:
        human = {"status": "pending", "slice_items": len(slice_items)}
        status = STATUS_PAUSED
        readme = _path(out_dir, SLICE_README_NAME, offline)
        readme.write_text(
            SLICE_INSTRUCTIONS.format(
                slice_file=slice_path,
                annotated_file=annotated_path,
                n=len(slice_items),
                unclear=UNCLEAR,
            )
        )
        messages.append(f"PAUSED for blinded adjudication: {len(slice_items)} items")
        messages.append(f"  slice        : {slice_path}")
        messages.append(f"  instructions : {readme}")
        messages.append(f"  then create  : {annotated_path}")
        messages.append("  and re-run the identical command to resume (no judge call is re-paid)")

    mean_chars = (
        sum(c.candidate.length_chars for c in cases) / len(cases) if cases else 0.0
    )
    report = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "status": status,
        "execution_mode": "dry-run" if offline else "execute",
        "persona": persona,
        "judge_model": judge_model,
        "judge_tag": j_tag,
        "bank": str(bank),
        "condition": condition or "all",
        "model_tag_filter": model_tag_filter,
        "instruments": {
            role: {
                "instrument_id": instrument_id,
                "content_hash": instruments.get(instrument_id).content_hash,
                "parser": instruments.get(instrument_id).parser,
            }
            for role, instrument_id in (("v1", V1_INSTRUMENT_ID), ("v2a", V2A_INSTRUMENT_ID))
        },
        "relevance_profile": profile_stamp,
        "selection": selection.stamp(),
        "counts": {
            "cases": len(case_rows),
            "controls": len(control_rows),
            "total": len(cases),
        },
        "judge_calls": {
            i: dict(call_stats[i]) for i in INSTRUMENT_IDS
        },
        "cost_projection": project_judge_cost(call_stats, judge_model, cfg, mean_chars),
        "bridge_table": {
            "overall": overall_table,
            "self_label_cases": case_table,
            "matched_controls": control_table,
        },
        "declared_trait_following": declared_trait_following(cases, verdicts),
        "ordinary_case_retention": {
            "n": control_table["n"],
            "v2a_parse_rate": control_table["v2a_parse_rate"],
            "agreement_with_v1": control_table["concordance"],
        },
        "length_slope": slopes,
        "disagreement_examples": disagreement_examples(cases, verdicts),
        "adjudication": human,
        "gate": evaluate_gate(
            gate,
            case_table=case_table,
            control_table=control_table,
            slopes=slopes,
            human=human,
        ),
        "caveats": [
            (
                f"self-label detection is Latin-script/English "
                f"({SELF_LABEL_DETECTOR_VERSION}); non-Latin self-labels read as "
                f"controls, which makes the case/control contrast conservative"
            ),
            (
                "selection strata depend on octt.trait_profiles curation; compare "
                "relevance_profile.traits_hash before comparing two bridge tables"
            ),
        ],
    }
    if offline:
        report["caveats"].insert(
            0,
            "DRY RUN: every verdict here is a deterministic offline stub, not a judge "
            "call. The tables are plumbing checks and carry no scientific content.",
        )

    report_path = _path(out_dir, REPORT_JSON_NAME, offline)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path = _path(out_dir, REPORT_MD_NAME, offline)
    md_path.write_text(render_report_md(report))
    return BridgeOutcome(
        status=status,
        report=report,
        out_dir=out_dir,
        report_path=report_path,
        slice_path=slice_path,
        annotated_path=annotated_path,
        messages=messages,
    )


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _num(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):+.{digits}f}"


def render_report_md(report: dict[str, Any]) -> str:
    """Human-readable bridge report; every table traces to ``bridge_report.json``."""
    tables = report["bridge_table"]
    lines = [
        "# validity-v2a bridge report",
        "",
        f"- persona: `{report['persona']}`  judge: `{report['judge_model']}`",
        f"- mode: **{report['execution_mode']}**  status: **{report['status']}**",
        (
            f"- v1: `{report['instruments']['v1']['instrument_id']}` "
            f"({report['instruments']['v1']['content_hash']})"
        ),
        (
            f"- v2a: `{report['instruments']['v2a']['instrument_id']}` "
            f"({report['instruments']['v2a']['content_hash']})"
        ),
        (
            f"- selection: `{report['selection']['selection_version']}` / "
            f"`{report['selection']['detector_version']}` seed={report['selection']['seed']}"
        ),
        f"- relevance curation hash: `{report['relevance_profile']['traits_hash']}`",
        "",
        "## Bridge table (v1 vs v2a)",
        "",
        "| subset | n | both scored | agree | disagree | concordance |",
        "|---|---|---|---|---|---|",
    ]
    for label, key in (
        ("overall", "overall"),
        ("self-label cases", "self_label_cases"),
        ("matched controls", "matched_controls"),
    ):
        t = tables[key]
        lines.append(
            f"| {label} | {t['n']} | {t['both_scored']} | {t['agree']} | {t['disagree']} | "
            f"{_pct(t['concordance'])} |"
        )

    follow = report["declared_trait_following"]
    lines += [
        "",
        "## Does the judge follow the declaration?",
        "",
        (
            f"Self-label cases with an unambiguous declared trait: {follow['n']} "
            f"(chance reference {_pct(follow['chance_reference'])})"
        ),
        "",
        "| instrument | scored | winner == declared trait |",
        "|---|---|---|",
    ]
    for instrument_id in INSTRUMENT_IDS:
        entry = follow[instrument_id]
        lines.append(
            f"| `{instrument_id}` | {entry['scored']} | {_pct(entry['follow_rate'])} |"
        )

    retention = report["ordinary_case_retention"]
    slopes = report["length_slope"]
    lines += [
        "",
        "## Ordinary-case retention",
        "",
        f"- non-label controls: {retention['n']}",
        f"- v2a produced a parsed verdict on {_pct(retention['v2a_parse_rate'])}",
        f"- and recovered v1's winner on {_pct(retention['agreement_with_v1'])}",
        "",
        "## Response-length slope",
        "",
        f"Lexicon `{slopes['lexicon_version']}`; slopes need n >= {slopes['min_n']}.",
        "",
        "| instrument | n | elaboration-trait win slope vs log length |",
        "|---|---|---|",
    ]
    for instrument_id in INSTRUMENT_IDS:
        entry = slopes[instrument_id]
        lines.append(
            f"| `{instrument_id}` | {entry['n']} | {_num(entry['elaboration_slope'])} |"
        )
    lines.append(
        f"\nConcordance-vs-length slope: {_num(slopes['concordance_slope']['slope'])} "
        f"(n={slopes['concordance_slope']['n']})"
    )

    adjudication = report["adjudication"]
    lines += ["", "## Blinded adjudication", "", f"status: **{adjudication['status']}**"]
    if adjudication["status"] == "annotated":
        lines.append("")
        lines.append("| instrument | scored | agreement with blinded reads |")
        lines.append("|---|---|---|")
        for instrument_id in INSTRUMENT_IDS:
            entry = adjudication[instrument_id]
            lines.append(
                f"| `{instrument_id}` | {entry['scored']} | {_pct(entry['agreement'])} |"
            )
        lines.append(f"\nunclear reads: {adjudication['unclear']}")

    gate = report["gate"]
    lines += [
        "",
        f"## Adoption gate ({gate['version']}): **{gate['overall']}**",
        "",
        "| criterion | measured | threshold | status |",
        "|---|---|---|---|",
    ]
    for criterion in gate["criteria"]:
        measured = criterion["measured"]
        threshold = criterion["threshold"]
        lines.append(
            f"| {criterion['criterion']} | "
            f"{'n/a' if measured is None else f'{float(measured):.4f}'} | "
            f"{'n/a' if threshold is None else f'{float(threshold):.4f}'} | "
            f"{criterion['status']} |"
        )

    examples = report["disagreement_examples"]
    lines += ["", f"## Disagreement examples ({len(examples)})", ""]
    for example in examples:
        tags = ", ".join(example["tags"]) or "none"
        lines += [
            (
                f"- **{example['item_id']}** [{example['group']}/{example['status']}] "
                f"`{example['a']}` vs `{example['b']}` -> v1 **{example['v1_winner']}**, "
                f"v2a **{example['v2a_winner']}** "
                f"(tags: {tags}; {example['length_chars']} chars)"
            ),
            f"  > {example['response_excerpt']}",
        ]

    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in report["caveats"]]
    for caveat in report["selection"]["caveats"]:
        lines.append(f"- selection: {caveat}")
    return "\n".join(lines) + "\n"


def summarize(outcome: BridgeOutcome) -> list[str]:
    """Terminal summary lines for the CLI."""
    report = outcome.report
    tables = report["bridge_table"]
    calls = report["judge_calls"]
    v1_calls, v2a_calls = calls[V1_INSTRUMENT_ID], calls[V2A_INSTRUMENT_ID]
    lines = [
        (
            f"bridge [{report['execution_mode']}]: {report['counts']['cases']} self-label "
            f"cases + {report['counts']['controls']} matched controls (screened "
            f"{report['selection']['universe']['responses_screened']} banked responses)"
        ),
        (
            f"judge calls: v1 bank={v1_calls['bank']} new={v1_calls['new']} | "
            f"v2a cache={v2a_calls['cache']} new={v2a_calls['new']} "
            f"(~${report['cost_projection']['estimated_usd']:.2f} if unpaid)"
        ),
        (
            f"concordance: overall {_pct(tables['overall']['concordance'])} | "
            f"self-label {_pct(tables['self_label_cases']['concordance'])} | "
            f"controls {_pct(tables['matched_controls']['concordance'])}"
        ),
        f"gate: {report['gate']['overall']}",
        f"report: {outcome.report_path}",
    ]
    lines.extend(outcome.messages)
    return lines
