"""Trained preference (reward) model: corpus, split, training, pre-RL gates (B16).

Readiness doc `docs/IMPLEMENTATION_READINESS_2026-07-27.md` lines 596-627. The
reward model is the thing Phase 3 RL will optimize, so every way it can be
wrong is a way the RL result can be wrong *and look fine*. This module is
built around the five failure modes the doc names:

**1. The corpus is less diverse than its row count says.** The four banked
Phase-1 sets are ~750 comparisons each, but they are NOT 3,000 independent
prompts: :func:`dpo_prompts` is a pure function of (constitution, n), the LIMA
half is a deterministic *prefix*, so sets at different scales are nested and
sets for different personas share their whole LIMA half. :func:`audit_prompts`
measures this on the real files — exact duplicates, normalized duplicates, and
near-duplicate clusters — and every corpus carries the audit it was built from.
Nothing here assumes a number; the audit reports one.

**2. Leakage through the split.** Splitting by comparison row puts the same
prompt in train and test. Splitting by exact prompt still leaks when two
prompts are near-duplicates. :func:`split_prompts` therefore splits on the
*near-duplicate cluster key*, before any pair expansion, and the held-out test
split is reserved and never scored during development.

**3. Position, not content.** A/B orientation is randomized per pair
(:func:`orient_pair`) and every split — validation and test included — carries
both ordering directions (:func:`swap_augment`), so order-swap consistency is
measurable rather than assumed.

**4. A corpus that changes under you.** The cookbook's ``HHHComparisonBuilder``
calls ``load_dataset("Anthropic/hh-rlhf")`` with no revision pin and shuffles;
two runs a month apart can train on different data and nothing says so. The
helpfulness half here is **materialized to a local file with its resolved
source revision and a content hash** (:func:`materialize_helpfulness`), and
:func:`load_helpfulness` reads only that file and refuses on any drift. There
is no remote fallback: a missing or drifted corpus is an error, never a quiet
downgrade (see `MEMORY: silent-data-degradation-traps`).

**5. Reward that collapses to length or costume.** The pre-RL gates
(:func:`evaluate_gates`) include counterfactuals where a response is *identical
but longer* and *identical but marker-stuffed*. A reward model that prefers
either one fails, and it fails before any RL money is spent.

**Pilot status.** :func:`pilot_status` reports PILOT unless a caller supplies an
explicit diversity reference. The overoptimization literature supports more
diverse reward-model data but gives no universal minimum sample count for this
project, so this module refuses to invent one and present it as established.

**Cost.** Dry-run by default at every level. Labeling and on-policy sampling
reach the paid runtime only with ``execute=True`` *and* a non-dry-run runtime;
the offline paths are deterministic so the gates can be proven to have teeth
with no API key and no training stack. Heavy dependencies (``tinker``,
``datasets``, ``huggingface_hub``) are imported inside functions only.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# `trait_profiles` is deliberately NOT imported: it is analysis curation and
# must never reach into a measurement instrument (CLAUDE.md). `persona_markers`
# IS a pinned, side-effect-free instrument and is the marker authority here.
from . import artifacts, manifest, persona_markers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Versions. Never edit a pinned set in place -- mint a new version.
# ---------------------------------------------------------------------------

#: Corpus construction protocol: sources, split rule, orientation policy.
CORPUS_PROTOCOL_VERSION = "reward-corpus-v1-2026-08-07"

#: The pre-RL acceptance gate set (thresholds + which gates exist).
GATE_SET_VERSION = "reward-model-pre-rl-gates-v1-2026-08-07"

#: The frozen counterfactual / helpfulness / format control set.
CONTROL_SET_VERSION = "reward-counterfactual-controls-v1-2026-08-07"

#: Registry id of the frozen control set (see :mod:`octt.instruments`).
INSTRUMENT_ID = "reward-model/pre-rl-controls-v1"

#: Whether a corpus's labels and helpfulness half are real evidence.
#: ``dry-run-stub`` corpora exist so the offline tier can exercise the whole
#: pipeline; :func:`train` refuses to spend on one. Without this stamp a
#: dry-run corpus is byte-shaped exactly like a real one, which is the
#: fixture-fallback failure this repo has already been bitten by.
EXECUTION_REAL = "real"
EXECUTION_STUB = "dry-run-stub"

#: Marker instrument used by the costume-collapse gate. Stamped into results.
MARKER_INSTRUMENT = persona_markers.MARKER_SET_VERSION


# ---------------------------------------------------------------------------
# Deduplication audit
# ---------------------------------------------------------------------------

#: Jaccard similarity over content-token sets at/above which two prompts are
#: treated as the same effective prompt. Part of the audit's definition.
NEAR_DUP_JACCARD = 0.8

#: Postings longer than this are treated as uninformative for candidate
#: generation (a token in half the corpus cannot localize a duplicate). Purely
#: a performance guard: it can only ever miss candidates that share nothing but
#: extremely common tokens, which cannot reach the Jaccard threshold anyway.
_MAX_POSTING = 200

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_prompt(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace. Used for exact-dup counting."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class _Union:
    """Minimal union-find over integer ids."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def cluster_prompts(
    prompts: Sequence[str], *, threshold: float = NEAR_DUP_JACCARD
) -> dict[str, str]:
    """Map every prompt to its near-duplicate cluster key.

    The cluster key is the lexicographically smallest *normalized* prompt in
    the cluster, so it is stable under input order — which is what makes the
    prompt split reproducible across runs and across machines.
    """
    unique = sorted({normalize_prompt(p) for p in prompts})
    index = {norm: i for i, norm in enumerate(unique)}
    tokens = [_token_set(norm) for norm in unique]
    postings: dict[str, list[int]] = {}
    for i, toks in enumerate(tokens):
        for tok in toks:
            postings.setdefault(tok, []).append(i)

    uf = _Union(len(unique))
    seen: set[tuple[int, int]] = set()
    for ids in postings.values():
        if len(ids) > _MAX_POSTING:
            continue
        for pos, a in enumerate(ids):
            for b in ids[pos + 1 :]:
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                if _jaccard(tokens[a], tokens[b]) >= threshold:
                    uf.union(a, b)

    representative: dict[int, str] = {}
    for i, norm in enumerate(unique):
        root = uf.find(i)
        current = representative.get(root)
        if current is None or norm < current:
            representative[root] = norm
    return {p: representative[uf.find(index[normalize_prompt(p)])] for p in set(prompts)}


@dataclass(frozen=True)
class DedupReport:
    """What the banked data actually contains, measured rather than assumed."""

    rows: int
    unique_exact: int
    unique_normalized: int
    effective_prompts: int
    threshold: float
    per_set: tuple[dict[str, Any], ...]
    overlaps: tuple[dict[str, Any], ...]
    largest_clusters: tuple[dict[str, Any], ...]

    @property
    def redundancy(self) -> float:
        """Comparison rows per effective (near-duplicate-collapsed) prompt."""
        return self.rows / self.effective_prompts if self.effective_prompts else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "unique_exact": self.unique_exact,
            "unique_normalized": self.unique_normalized,
            "effective_prompts": self.effective_prompts,
            "redundancy": self.redundancy,
            "near_dup_jaccard": self.threshold,
            "per_set": [dict(s) for s in self.per_set],
            "overlaps": [dict(o) for o in self.overlaps],
            "largest_clusters": [dict(c) for c in self.largest_clusters],
        }

    def summary(self) -> str:
        lines = [
            f"rows                : {self.rows}",
            f"unique prompts      : {self.unique_exact}",
            f"normalized-unique   : {self.unique_normalized}",
            (
                f"effective prompts   : {self.effective_prompts} "
                f"(near-dup Jaccard >= {self.threshold})"
            ),
            f"redundancy          : {self.redundancy:.2f} rows per effective prompt",
            "",
            "per set:",
        ]
        for s in self.per_set:
            lines.append(
                f"  {s['name']:<44} rows={s['rows']:>5} unique={s['unique']:>5} "
                f"dup_rows={s['duplicate_rows']:>4}"
            )
        if self.overlaps:
            lines.append("")
            lines.append("cross-set prompt overlap (shared / union):")
            for o in self.overlaps:
                lines.append(
                    f"  {o['a']:<30} x {o['b']:<30} shared={o['shared']:>5} "
                    f"jaccard={o['jaccard']:.3f} containment={o['containment']:.3f}"
                )
        return "\n".join(lines)


def audit_prompts(
    named_sets: Mapping[str, Sequence[str]], *, threshold: float = NEAR_DUP_JACCARD
) -> DedupReport:
    """Measure exact, normalized, and semantic duplication across prompt sets.

    ``named_sets`` maps a set name (usually a banked file) to its prompt list
    **in row order** — duplicates included, because the row count is half the
    story. Cross-set overlap is reported as both Jaccard and *containment*
    (shared / smaller set): nested sets look unremarkable under Jaccard and
    obvious under containment, and nesting is exactly what the deterministic
    LIMA prefix produces.
    """
    per_set: list[dict[str, Any]] = []
    all_prompts: list[str] = []
    sets: dict[str, set[str]] = {}
    for name, prompts in named_sets.items():
        prompts = list(prompts)
        uniq = set(prompts)
        sets[name] = uniq
        all_prompts.extend(prompts)
        per_set.append(
            {
                "name": name,
                "rows": len(prompts),
                "unique": len(uniq),
                "duplicate_rows": len(prompts) - len(uniq),
                "normalized_unique": len({normalize_prompt(p) for p in prompts}),
            }
        )

    overlaps: list[dict[str, Any]] = []
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = len(sets[a] & sets[b])
            if not shared:
                continue
            union = len(sets[a] | sets[b])
            smaller = min(len(sets[a]), len(sets[b])) or 1
            overlaps.append(
                {
                    "a": a,
                    "b": b,
                    "shared": shared,
                    "jaccard": shared / union if union else 0.0,
                    "containment": shared / smaller,
                }
            )

    distinct = sorted(set(all_prompts))
    clusters = cluster_prompts(distinct, threshold=threshold)
    members: dict[str, list[str]] = {}
    for prompt in distinct:
        members.setdefault(clusters[prompt], []).append(prompt)
    largest = sorted(members.values(), key=lambda m: (-len(m), m[0]))[:5]

    return DedupReport(
        rows=len(all_prompts),
        unique_exact=len(distinct),
        unique_normalized=len({normalize_prompt(p) for p in distinct}),
        effective_prompts=len(members),
        threshold=threshold,
        per_set=tuple(per_set),
        overlaps=tuple(overlaps),
        largest_clusters=tuple(
            {"size": len(m), "members": list(m[:3])} for m in largest if len(m) > 1
        ),
    )


def load_banked_pairs(path: Path | str, *, source: str = "character-banked") -> list[PreferencePair]:
    """Read a banked ``dpo_pairs.jsonl`` into :class:`PreferencePair` rows.

    The banked label is the *teacher's* preference (chosen = constitution-in-
    context teacher, rejected = base student), which is why every pair built
    this way is stamped ``label_source="banked-teacher"``: it is a free,
    already-gated label, and it is NOT the Phase 3 judge's label.
    """
    path = Path(path)
    pairs: list[PreferencePair] = []
    for row in artifacts.read_jsonl(path):
        prompt = (row.get("prompt") or "").strip()
        chosen = (row.get("chosen") or "").strip()
        rejected = (row.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            continue
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source=source,
                origin=path.name,
                label_source=LABEL_SOURCE_BANKED_TEACHER,
                meta={
                    "persona": row.get("persona"),
                    "student": row.get("student"),
                    "teacher": row.get("teacher"),
                },
            )
        )
    return pairs


def audit_banked_files(paths: Sequence[Path | str], **kwargs: Any) -> DedupReport:
    """Run :func:`audit_prompts` over banked ``dpo_pairs.jsonl`` files."""
    named: dict[str, list[str]] = {}
    for raw in paths:
        path = Path(raw)
        named[_set_name(path)] = [p.prompt for p in load_banked_pairs(path)]
    return audit_prompts(named, **kwargs)


def _set_name(path: Path) -> str:
    """A short, stable label for a banked file (run dir + rung, not the full path)."""
    parts = path.parts
    if "runs" in parts:
        tail = parts[parts.index("runs") + 1 :]
        return "/".join(tail[:-1]) or path.name
    return str(path.parent.name or path.name)


# ---------------------------------------------------------------------------
# Prompt split (by cluster, before pair expansion)
# ---------------------------------------------------------------------------

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

#: 80/10/10 by prompt hash, per the readiness doc's construction step 5.
DEFAULT_SPLIT_WEIGHTS: dict[str, float] = {SPLIT_TRAIN: 0.8, SPLIT_VAL: 0.1, SPLIT_TEST: 0.1}

#: Salt in the split hash. Changing it reshuffles every split, which
#: invalidates a reserved test set — hence a version, not a free parameter.
SPLIT_SALT = "octt-reward-split-v1"


def cluster_key(prompt: str, clusters: Mapping[str, str] | None = None) -> str:
    """The value actually hashed for the split: the near-duplicate cluster key."""
    if clusters is not None and prompt in clusters:
        return clusters[prompt]
    return normalize_prompt(prompt)


def split_fraction(key: str, *, salt: str = SPLIT_SALT) -> float:
    """Deterministic uniform position in [0, 1) for a cluster key."""
    digest = artifacts.text_hash(f"{salt}|{key}")
    return int(digest[:16], 16) / float(1 << 64)


def assign_split(
    key: str,
    *,
    weights: Mapping[str, float] = DEFAULT_SPLIT_WEIGHTS,
    salt: str = SPLIT_SALT,
) -> str:
    """Hash-assign one cluster key to a split. No RNG, no ordering dependence."""
    total = sum(weights[s] for s in SPLITS if s in weights)
    if total <= 0:
        raise ValueError("split weights must sum to a positive number")
    position = split_fraction(key, salt=salt) * total
    running = 0.0
    for name in SPLITS:
        running += weights.get(name, 0.0)
        if position < running:
            return name
    return SPLIT_TEST


def split_prompts(
    prompts: Sequence[str],
    *,
    weights: Mapping[str, float] = DEFAULT_SPLIT_WEIGHTS,
    salt: str = SPLIT_SALT,
    threshold: float = NEAR_DUP_JACCARD,
) -> dict[str, str]:
    """Map every prompt to a split, splitting by near-duplicate CLUSTER.

    Two prompts that differ only in punctuation, or that share 80% of their
    tokens, land in the same split. Splitting on the raw prompt would let a
    paraphrase of a training prompt sit in the held-out set and inflate every
    number this module reports.
    """
    clusters = cluster_prompts(prompts, threshold=threshold)
    return {
        prompt: assign_split(clusters[prompt], weights=weights, salt=salt)
        for prompt in set(prompts)
    }


# ---------------------------------------------------------------------------
# Corpus rows
# ---------------------------------------------------------------------------

SOURCE_CHARACTER_BANKED = "character-banked"
SOURCE_CHARACTER_ONPOLICY = "character-onpolicy"
SOURCE_HELPFULNESS = "helpfulness"

LABEL_SOURCE_BANKED_TEACHER = "banked-teacher"
LABEL_SOURCE_PHASE3_JUDGE = "phase3-character-judge"
LABEL_SOURCE_DATASET = "dataset-label"

LABEL_A = "A"
LABEL_B = "B"

ORIENTATION_DIRECT = "direct"
ORIENTATION_SWAPPED = "swapped"


@dataclass(frozen=True)
class PreferencePair:
    """One unordered preference: ``chosen`` is preferred over ``rejected``.

    Orientation is deliberately absent here — a pair has no A/B until
    :func:`orient_pair` assigns one, so nothing upstream can bake in a
    position bias.
    """

    prompt: str
    chosen: str
    rejected: str
    source: str
    origin: str
    label_source: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        return manifest.content_hash(self.prompt, self.chosen, self.rejected, length=20)


@dataclass(frozen=True)
class OrientedRow:
    """A pair pinned to one A/B presentation, with the label that follows."""

    pair_id: str
    prompt: str
    response_a: str
    response_b: str
    label: str
    orientation: str
    split: str
    source: str
    origin: str
    label_source: str
    cluster: str

    @property
    def chosen(self) -> str:
        return self.response_a if self.label == LABEL_A else self.response_b

    @property
    def rejected(self) -> str:
        return self.response_b if self.label == LABEL_A else self.response_a

    def swapped(self) -> OrientedRow:
        """The mirrored presentation of the same underlying preference."""
        return OrientedRow(
            pair_id=self.pair_id,
            prompt=self.prompt,
            response_a=self.response_b,
            response_b=self.response_a,
            label=LABEL_B if self.label == LABEL_A else LABEL_A,
            orientation=(
                ORIENTATION_SWAPPED
                if self.orientation == ORIENTATION_DIRECT
                else ORIENTATION_DIRECT
            ),
            split=self.split,
            source=self.source,
            origin=self.origin,
            label_source=self.label_source,
            cluster=self.cluster,
        )

    def to_row(self) -> dict[str, Any]:
        """JSONL form: human-readable AND directly trainable by the cookbook.

        The ``comparison``/``label`` view is the schema
        ``tinker_cookbook.preference.preference_datasets.ComparisonBuilderFromJsonl``
        reads, so a corpus file trains without a conversion step (the same
        idiom as :mod:`octt.distillation`).
        """
        return {
            "pair_id": self.pair_id,
            "prompt": self.prompt,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "label": self.label,
            "orientation": self.orientation,
            "split": self.split,
            "source": self.source,
            "origin": self.origin,
            "label_source": self.label_source,
            "cluster": self.cluster,
            "corpus_protocol": CORPUS_PROTOCOL_VERSION,
            "comparison": {
                "prompt_conversation": [{"role": "user", "content": self.prompt}],
                "completion_A": [{"role": "assistant", "content": self.response_a}],
                "completion_B": [{"role": "assistant", "content": self.response_b}],
            },
        }


def orient_pair(pair: PreferencePair, split: str, cluster: str, rng: random.Random) -> OrientedRow:
    """Randomize which side of the pair is presented as A."""
    chosen_is_a = rng.random() < 0.5
    return OrientedRow(
        pair_id=pair.pair_id,
        prompt=pair.prompt,
        response_a=pair.chosen if chosen_is_a else pair.rejected,
        response_b=pair.rejected if chosen_is_a else pair.chosen,
        label=LABEL_A if chosen_is_a else LABEL_B,
        orientation=ORIENTATION_DIRECT,
        split=split,
        source=pair.source,
        origin=pair.origin,
        label_source=pair.label_source,
        cluster=cluster,
    )


def swap_augment(rows: Sequence[OrientedRow]) -> list[OrientedRow]:
    """Add the mirrored presentation of every row, in place, inside one split.

    Applied *after* the split so a row and its mirror can never straddle the
    train/test boundary, and applied to validation and test as well — the
    readiness doc requires both ordering directions in validation, which is
    what makes order-swap consistency measurable at all.
    """
    out: list[OrientedRow] = []
    for row in rows:
        out.append(row)
        out.append(row.swapped())
    return out


# ---------------------------------------------------------------------------
# Helpfulness / HHH corpus: materialized locally, revision-pinned
# ---------------------------------------------------------------------------


class HelpfulnessCorpusError(RuntimeError):
    """The local helpfulness corpus is missing, drifted, or unstamped.

    Never downgraded to a warning: a reward model silently trained on a
    different helpfulness corpus than the one banked with a result is exactly
    the silent-degradation failure this repo has already been bitten by.
    """


@dataclass(frozen=True)
class HelpfulnessSource:
    """A pinned external preference dataset."""

    source_id: str
    dataset: str
    revision: str
    split: str
    subset: str | None = None
    note: str = ""


#: Revision verified against the HF dataset API on 2026-08-07:
#: ``GET /api/datasets/Anthropic/hh-rlhf`` -> sha 09be8c5b..., lastModified
#: 2023-05-26. The cookbook's HHHComparisonBuilder pins nothing; this does.
HELPFULNESS_SOURCES: dict[str, HelpfulnessSource] = {
    "hh-rlhf-helpful-base": HelpfulnessSource(
        source_id="hh-rlhf-helpful-base",
        dataset="Anthropic/hh-rlhf",
        revision="09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
        split="train",
        subset="helpful-base",
        note="Anthropic HH-RLHF helpfulness split (arXiv 2204.05862).",
    ),
    "hh-rlhf-train": HelpfulnessSource(
        source_id="hh-rlhf-train",
        dataset="Anthropic/hh-rlhf",
        revision="09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
        split="train",
        note="Full HH-RLHF train split (helpfulness + harmlessness).",
    ),
}

#: Materialization protocol: bump when the parse or the row schema changes.
HELPFULNESS_PROTOCOL_VERSION = "helpfulness-materialize-v1"

#: The offline fixture id. Explicitly NOT one of HELPFULNESS_SOURCES: it is
#: stamped as a fixture everywhere it appears so it can never be mistaken for
#: the real corpus, and :func:`build_corpus` refuses it unless asked.
FIXTURE_SOURCE_ID = "builtin-fixture-v1"

#: A tiny, frozen, obviously-synthetic helpfulness set for the offline tier.
#: Pinned here (not generated) so the dry-run corpus is reproducible.
HELPFULNESS_FIXTURE: tuple[tuple[str, str, str], ...] = (
    (
        "How do I stop my sourdough starter from going flat?",
        (
            "Feed it on a fixed schedule: equal weights of starter, flour and water "
            "every 12 hours at around 24 C. If it still falls quickly, cut the "
            "hydration to 80% and switch a quarter of the flour to whole rye, which "
            "ferments faster and gives the yeast more to work with."
        ),
        "Starters are tricky. You might want to look into it more.",
    ),
    (
        "What should I check first when a laptop will not power on?",
        (
            "Work outward from power: try a different known-good charger and outlet, "
            "hold the power button for 30 seconds with the charger removed to drain "
            "residual charge, then look for a charging LED. If the LED lights but "
            "the screen stays dark, shine a torch at the panel to rule out a failed "
            "backlight before assuming the board is dead."
        ),
        "It is probably broken. Take it to a shop.",
    ),
    (
        "Explain what a database index does, briefly.",
        (
            "An index is a separate sorted structure over one or more columns. "
            "Lookups on those columns become a tree descent instead of a full scan, "
            "which is why reads get faster; every write has to update the index too, "
            "which is why writes get slower and why you do not index everything."
        ),
        "It makes the database faster. That is basically it.",
    ),
    (
        "My tomato seedlings are leggy. What do I do?",
        (
            "Leggy seedlings are reaching for light. Move them within 5 cm of a grow "
            "light for 14-16 hours a day, run a fan for a couple of hours daily to "
            "thicken the stems, and when you pot on, bury the stem up to the first "
            "true leaves — tomatoes root along buried stem."
        ),
        "That happens sometimes. Try giving them more light I guess.",
    ),
    (
        "How should I structure a two-week handover before I leave a job?",
        (
            "Week one: write down every recurring task with its trigger, owner and "
            "the access it needs, then shadow your successor through one full cycle "
            "of each. Week two: reverse it — they drive, you watch — and spend the "
            "remaining time on the undocumented things people only ask you for, "
            "which is where handovers actually fail."
        ),
        "Just write some notes and tell your manager.",
    ),
    (
        "Is it worth learning regular expressions?",
        (
            "Yes for the 80% case: character classes, quantifiers, anchors, groups "
            "and alternation cover almost all day-to-day text work and take an "
            "afternoon. Beyond that — lookbehind, backreferences, nested quantifiers "
            "— reach for a parser instead; regexes get exponentially slow and "
            "unreadable exactly where people are most tempted to use them."
        ),
        "Some people like them and some do not.",
    ),
    (
        "What is a safe way to cool a large pot of soup?",
        (
            "Split it into shallow containers no deeper than about 5 cm and get it "
            "through the 60-20 C band within two hours — an ice bath with occasional "
            "stirring is the fastest way. Do not put a full stockpot in the fridge: "
            "the centre stays warm for hours and that is where bacteria grow."
        ),
        "Just leave it on the counter until it cools down.",
    ),
    (
        "How do I read a compiler error I do not understand?",
        (
            "Read the FIRST error only — later ones are usually cascade damage. "
            "Note the file and column, then map the message to the smallest example "
            "that reproduces it by deleting code until the error moves or vanishes. "
            "The line that makes it vanish is the one to look at, which is often not "
            "the line the compiler named."
        ),
        "Errors are confusing. Try searching the message online.",
    ),
)


def _fixture_pairs() -> list[PreferencePair]:
    return [
        PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source=SOURCE_HELPFULNESS,
            origin=FIXTURE_SOURCE_ID,
            label_source=LABEL_SOURCE_DATASET,
            meta={"fixture": True},
        )
        for prompt, chosen, rejected in HELPFULNESS_FIXTURE
    ]


def _helpfulness_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _parse_hh_side(text: str) -> tuple[str, str] | None:
    """Split one HH-RLHF transcript into (final user turn, final assistant turn).

    Only single-exchange-tail comparisons are kept: the two sides must agree on
    everything except the last assistant turn, or the pair is not a preference
    over a response — it is a preference over a conversation.
    """
    parts = re.split(r"\n\n(Human|Assistant):\s*", text.strip())
    turns: list[tuple[str, str]] = []
    if parts and parts[0].strip():
        turns.append(("Human", parts[0].strip()))
    for i in range(1, len(parts) - 1, 2):
        turns.append((parts[i], parts[i + 1].strip()))
    if len(turns) < 2 or turns[-1][0] != "Assistant":
        return None
    prompt = turns[-2][1]
    if turns[-2][0] != "Human" or not prompt or not turns[-1][1]:
        return None
    return prompt, turns[-1][1]


def materialize_helpfulness(
    out_path: Path | str,
    *,
    n: int,
    source_id: str = "hh-rlhf-helpful-base",
    execute: bool = False,
    seed: int = 0,
) -> Path:
    """Download the pinned helpfulness corpus ONCE and write it to disk.

    Writes ``out_path`` (JSONL) plus a sidecar stamping the dataset id, the
    **resolved** revision the download actually served, the row count, the
    materialization protocol, and a content hash of the file. Everything
    downstream reads the local file; nothing downstream can reach the network.

    Without ``execute=True`` this writes the built-in fixture instead, stamped
    ``fixture: true`` — so the dry-run tier exercises the whole path without a
    download, and no report can confuse the two.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not execute:
        pairs = _fixture_pairs()
        rows = [
            {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs
        ]
        _write_helpfulness(out_path, rows, source_id=FIXTURE_SOURCE_ID, revision=None, fixture=True)
        logger.info("Materialized %d FIXTURE helpfulness rows to %s", len(rows), out_path)
        return out_path

    if source_id not in HELPFULNESS_SOURCES:
        raise HelpfulnessCorpusError(
            f"unknown helpfulness source {source_id!r}; pinned: "
            f"{', '.join(sorted(HELPFULNESS_SOURCES))}"
        )
    source = HELPFULNESS_SOURCES[source_id]

    from datasets import load_dataset  # lazy, optional

    kwargs: dict[str, Any] = {"split": source.split, "revision": source.revision}
    if source.subset:
        kwargs["data_dir"] = source.subset
    dataset = load_dataset(source.dataset, **kwargs)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for example in dataset:
        chosen = _parse_hh_side(example.get("chosen") or "")
        rejected = _parse_hh_side(example.get("rejected") or "")
        if chosen is None or rejected is None:
            continue
        prompt, chosen_text = chosen
        rejected_prompt, rejected_text = rejected
        # The two sides must share the prompt, or the "preference" is over a
        # different conversation and would teach the reward model nothing.
        if prompt != rejected_prompt or chosen_text == rejected_text:
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append({"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text})
        if len(rows) >= n:
            break

    if len(rows) < n:
        logger.warning(
            "helpfulness materialization yielded %d usable rows, asked for %d", len(rows), n
        )
    random.Random(seed).shuffle(rows)
    _write_helpfulness(
        out_path, rows, source_id=source.source_id, revision=source.revision, fixture=False
    )
    logger.info(
        "Materialized %d helpfulness rows from %s@%s to %s",
        len(rows), source.dataset, source.revision, out_path,
    )
    return out_path


def _write_helpfulness(
    out_path: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    source_id: str,
    revision: str | None,
    fixture: bool,
) -> None:
    artifacts.write_jsonl_atomic(out_path, rows)
    manifest.atomic_write_json(
        _helpfulness_meta_path(out_path),
        {
            "protocol": HELPFULNESS_PROTOCOL_VERSION,
            "source_id": source_id,
            "dataset": (
                HELPFULNESS_SOURCES[source_id].dataset if source_id in HELPFULNESS_SOURCES else None
            ),
            "revision": revision,
            "rows": len(rows),
            "fixture": fixture,
            "content_hash": artifacts.content_hash([dict(r) for r in rows]),
        },
    )


def load_helpfulness(path: Path | str) -> tuple[list[PreferencePair], dict[str, Any]]:
    """Read the LOCAL materialized corpus and verify it has not drifted.

    Raises :class:`HelpfulnessCorpusError` when the file or its sidecar is
    missing, when the protocol version does not match, or when the file's
    content hash disagrees with the stamp. There is deliberately no fallback
    to a remote builder: the point of materializing is that the corpus cannot
    change between runs without someone noticing.
    """
    path = Path(path)
    meta_path = _helpfulness_meta_path(path)
    if not path.is_file():
        raise HelpfulnessCorpusError(
            f"helpfulness corpus {path} does not exist. Materialize it first: "
            "`octt reward-model materialize --out <path> --execute`. This module "
            "never falls back to a remote dataset builder."
        )
    if not meta_path.is_file():
        raise HelpfulnessCorpusError(
            f"{path} has no {meta_path.name} sidecar, so its source revision is "
            "unknown. An unstamped corpus is not usable evidence; re-materialize."
        )
    meta = json.loads(meta_path.read_text())
    if meta.get("protocol") != HELPFULNESS_PROTOCOL_VERSION:
        raise HelpfulnessCorpusError(
            f"{path} was materialized under protocol {meta.get('protocol')!r}, "
            f"but this code is {HELPFULNESS_PROTOCOL_VERSION!r}. Re-materialize "
            "rather than mixing corpora."
        )
    rows = artifacts.read_jsonl(path)
    digest = artifacts.content_hash(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )
    if digest != meta.get("content_hash"):
        raise HelpfulnessCorpusError(
            f"{path} content hash {digest[:16]} != stamped {str(meta.get('content_hash'))[:16]}. "
            "The materialized corpus changed after it was stamped; refusing to train on it."
        )
    pairs = [
        PreferencePair(
            prompt=r["prompt"],
            chosen=r["chosen"],
            rejected=r["rejected"],
            source=SOURCE_HELPFULNESS,
            origin=str(meta.get("source_id")),
            label_source=LABEL_SOURCE_DATASET,
            meta={"fixture": bool(meta.get("fixture"))},
        )
        for r in rows
    ]
    return pairs, meta


# ---------------------------------------------------------------------------
# Character labeling: both orders + swap-consistency gate
# ---------------------------------------------------------------------------

#: The cookbook's preference group builder only performs the intended complete
#: both-directions tournament at group size 4 (readiness doc, "Stock-recipe
#: gaps"). Anything else is silently chunked, so it is a hard error here.
REQUIRED_GROUP_SIZE = 4

REASON_AGREE = "swap_agreement"
REASON_DISAGREE = "swap_disagreement"
REASON_BOTH_TIE = "both_orders_tie"
REASON_UNPARSEABLE = "unparseable"

class CharacterJudgeUnavailable(RuntimeError):
    """A paid character label was requested but no Phase 3 judge is wired in."""


@dataclass(frozen=True)
class LabelOutcome:
    """One pair's both-orders verdict and whether it survived the swap gate."""

    pair_id: str
    label: str | None
    reason: str
    verdict_ab: str | None
    verdict_ba: str | None


def resolve_both_orders(verdict_ab: str | None, verdict_ba: str | None) -> tuple[str | None, str]:
    """Resolve two presentations into a label over the UNDERLYING responses.

    ``verdict_ab`` is the judge's pick when the pair is shown (chosen-candidate
    first); ``verdict_ba`` is its pick with the sides swapped. A label survives
    only when the two presentations name the same underlying response. A judge
    that tracks position disagrees with itself and the pair is dropped — which
    is the readiness doc's "keep character labels only when parsing and
    swap-consistency gates pass".
    """
    if verdict_ab is None or verdict_ba is None:
        return None, REASON_UNPARSEABLE
    if verdict_ab == "TIE" and verdict_ba == "TIE":
        return None, REASON_BOTH_TIE
    # In the ba presentation the sides are flipped, so agreement means opposite
    # letters.
    flipped = {"A": "B", "B": "A", "TIE": "TIE"}.get(verdict_ba)
    if flipped is None or flipped != verdict_ab or verdict_ab == "TIE":
        return None, REASON_DISAGREE
    return verdict_ab, REASON_AGREE


def _offline_verdict(prompt: str, first: str, second: str, *, swap_consistent: bool) -> str:
    """Deterministic offline judge for ONE presentation.

    ``swap_consistent=False`` makes it answer by POSITION, which is the exact
    pathology the both-orders gate exists to catch — selectable offline so the
    gate can be shown to have teeth without spending.
    """
    if not swap_consistent:
        return "A"
    # Decide from the unordered CONTENT pair, so the verdict tracks content
    # rather than slot and the offline judge is swap-consistent by construction.
    winner = min(first, second)
    return "A" if winner == first else "B"


def character_judge_fn(
    runtime: Any,
    *,
    judge_model: str | None = None,
    brief_id: str | None = None,
) -> Callable[[str, str, str], str | None]:
    """A one-presentation judge callable built on :mod:`octt.preference`.

    The Phase 3 character judge is a registered instrument owned by
    ``octt/preference.py`` (``character/prompted-blind-swapped-v1``). This
    module never re-implements it and never copies its prompt text: it renders
    through ``preference.judge_messages`` and parses with
    ``preference.parse_verdict``, so there is exactly one copy of the judge and
    nothing here can drift away from it. The both-orders scheduling and the
    swap-consistency gate stay in :func:`label_pairs`, which is what turns two
    presentations into one retained label.
    """
    from . import generation, models, preference

    brief = preference.get_brief(brief_id) if brief_id else preference.get_brief()
    config = preference.DEFAULT_JUDGE_CONFIG
    sampler = generation.make_sampler(
        runtime,
        judge_model or models.TEACHER_MODEL,
        tag="reward-character-judge",
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    def judge(prompt: str, first: str, second: str) -> str | None:
        messages = preference.judge_messages(brief, prompt, first, second)
        raw = generation.complete_many(sampler, [messages])[0]
        return preference.parse_verdict(raw)

    return judge


def label_pairs(
    pairs: Sequence[PreferencePair],
    *,
    execute: bool = False,
    judge_fn: Callable[[str, str, str], str | None] | None = None,
    runtime: Any = None,
    judge_model: str | None = None,
    brief_id: str | None = None,
    swap_consistent: bool = True,
) -> tuple[list[PreferencePair], list[LabelOutcome]]:
    """Label pairs with the Phase 3 judge in BOTH orders; keep only survivors.

    Returns ``(kept_pairs, outcomes)``. A kept pair's ``chosen`` is whichever
    response the judge preferred in both presentations, so the returned pairs
    carry the judge's preference rather than the teacher's. Dry-run (the
    default) uses the deterministic offline judge and bills nothing.
    """
    if execute and judge_fn is None:
        if runtime is None:
            raise CharacterJudgeUnavailable(
                "paid character labeling needs either an explicit judge_fn or a "
                "runtime to build one from octt.preference's frozen judge."
            )
        judge_fn = character_judge_fn(
            runtime, judge_model=judge_model, brief_id=brief_id
        )

    kept: list[PreferencePair] = []
    outcomes: list[LabelOutcome] = []
    for pair in pairs:
        if execute and judge_fn is not None:
            verdict_ab = judge_fn(pair.prompt, pair.chosen, pair.rejected)
            verdict_ba = judge_fn(pair.prompt, pair.rejected, pair.chosen)
        else:
            verdict_ab = _offline_verdict(
                pair.prompt, pair.chosen, pair.rejected, swap_consistent=swap_consistent
            )
            verdict_ba = _offline_verdict(
                pair.prompt, pair.rejected, pair.chosen, swap_consistent=swap_consistent
            )
        label, reason = resolve_both_orders(verdict_ab, verdict_ba)
        outcomes.append(
            LabelOutcome(
                pair_id=pair.pair_id,
                label=label,
                reason=reason,
                verdict_ab=verdict_ab,
                verdict_ba=verdict_ba,
            )
        )
        if label is None:
            continue
        chosen, rejected = (
            (pair.chosen, pair.rejected) if label == LABEL_A else (pair.rejected, pair.chosen)
        )
        kept.append(
            PreferencePair(
                prompt=pair.prompt,
                chosen=chosen,
                rejected=rejected,
                source=pair.source,
                origin=pair.origin,
                label_source=LABEL_SOURCE_PHASE3_JUDGE,
                meta=dict(pair.meta),
            )
        )
    return kept, outcomes


def group_to_pairs(prompt: str, responses: Sequence[str], *, source: str, origin: str) -> list[
    PreferencePair
]:
    """Expand one G=4 sample group into its 6 unordered candidate pairs.

    12 ordered matchups = 6 unordered pairs judged in both directions, which is
    the complete tournament the cookbook only performs at G=4 — hence the hard
    group-size check.
    """
    if len(responses) != REQUIRED_GROUP_SIZE:
        raise ValueError(
            f"group size {len(responses)} is not {REQUIRED_GROUP_SIZE}; the complete "
            "both-directions tournament is only well-defined at G=4 (readiness doc)"
        )
    pairs: list[PreferencePair] = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=responses[i],
                    rejected=responses[j],
                    source=source,
                    origin=origin,
                    label_source=LABEL_SOURCE_PHASE3_JUDGE,
                    meta={"group_size": REQUIRED_GROUP_SIZE, "slots": [i, j]},
                )
            )
    return pairs


def sample_on_policy_groups(
    prompts: Sequence[str],
    runtime: Any,
    model: str,
    *,
    group_size: int = REQUIRED_GROUP_SIZE,
    execute: bool = False,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> dict[str, list[str]]:
    """Sample ``group_size`` base-policy responses per prompt.

    Dry-run (the default, and whenever the runtime is itself dry-run) returns
    the runtime's deterministic stub completions and bills nothing.
    """
    if group_size != REQUIRED_GROUP_SIZE:
        raise ValueError(
            f"group_size must be {REQUIRED_GROUP_SIZE}: the cookbook silently chunks "
            "larger groups into contiguous fours, so any other value measures a "
            "different tournament than the one reported"
        )
    from . import generation  # lazy: keeps this module importable without tinker

    sampler = generation.make_sampler(
        runtime,
        model,
        tag="reward-onpolicy",
        max_tokens=max_tokens,
        temperature=temperature if (execute and not runtime.config.dry_run) else 0.0,
    )
    conversations = []
    for prompt in prompts:
        for _ in range(group_size):
            conversations.append([{"role": "user", "content": prompt}])
    completions = generation.complete_many(sampler, conversations)
    groups: dict[str, list[str]] = {}
    for index, prompt in enumerate(prompts):
        window = completions[index * group_size : (index + 1) * group_size]
        groups[prompt] = list(window)
    return groups


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """Split, oriented, swap-augmented rows plus the mix that produced them."""

    rows: tuple[OrientedRow, ...]
    mix: Mapping[str, Any]
    dedup: DedupReport

    def split(self, name: str) -> list[OrientedRow]:
        return [r for r in self.rows if r.split == name]

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in SPLITS}
        for row in self.rows:
            out[row.split] = out.get(row.split, 0) + 1
        return out

    def prompts(self, name: str | None = None) -> set[str]:
        return {r.prompt for r in self.rows if name is None or r.split == name}

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_protocol": CORPUS_PROTOCOL_VERSION,
            "counts": self.counts(),
            "mix": dict(self.mix),
            "dedup": self.dedup.to_dict(),
        }


class CorpusError(ValueError):
    """The assembled corpus violates an invariant the gates depend on."""


def build_corpus(
    character_pairs: Sequence[PreferencePair],
    helpfulness_pairs: Sequence[PreferencePair],
    *,
    seed: int = 0,
    weights: Mapping[str, float] = DEFAULT_SPLIT_WEIGHTS,
    threshold: float = NEAR_DUP_JACCARD,
    target_helpfulness: int | None = None,
    allow_fixture: bool = False,
    execution_mode: str = EXECUTION_REAL,
) -> Corpus:
    """Assemble the reward corpus: split by prompt cluster, then orient, then swap.

    Order matters and is the whole point:

      1. audit the prompts (the report is banked with the corpus);
      2. assign every prompt CLUSTER to a split — before any pair expansion, so
         no prompt and no paraphrase of it can appear on both sides;
      3. orient each pair with a seeded RNG (A/B randomized);
      4. swap-augment inside each split, so both directions exist everywhere.

    ``target_helpfulness`` truncates the helpfulness half to a fixed count (the
    readiness doc targets an *equal* count, ~1,500). The realized counts and
    sampling weights are stamped in :attr:`Corpus.mix` either way, so a run
    that could not reach its target says so instead of hiding it.
    """
    helpfulness = list(helpfulness_pairs)
    if not allow_fixture:
        fixtures = [p for p in helpfulness if p.meta.get("fixture") or p.origin == FIXTURE_SOURCE_ID]
        if fixtures:
            raise CorpusError(
                f"{len(fixtures)} helpfulness pairs come from {FIXTURE_SOURCE_ID}. The "
                "fixture is for the offline tier only; pass allow_fixture=True to "
                "build an explicitly-marked pilot corpus, or materialize the real "
                "corpus with `octt reward-model materialize --execute`."
            )
    if target_helpfulness is not None:
        helpfulness = helpfulness[:target_helpfulness]

    character = list(character_pairs)
    all_pairs = character + helpfulness
    if not all_pairs:
        raise CorpusError("refusing to build an empty reward corpus")

    named_sets: dict[str, list[str]] = {}
    for pair in all_pairs:
        named_sets.setdefault(f"{pair.source}:{pair.origin}", []).append(pair.prompt)
    dedup = audit_prompts(named_sets, threshold=threshold)

    prompts = [p.prompt for p in all_pairs]
    clusters = cluster_prompts(prompts, threshold=threshold)
    split_of = {
        prompt: assign_split(clusters[prompt], weights=weights) for prompt in set(prompts)
    }

    rng = random.Random(seed)
    # Sort before orienting so the RNG stream — and therefore every corpus —
    # is reproducible regardless of input file order.
    oriented = [
        orient_pair(pair, split_of[pair.prompt], clusters[pair.prompt], rng)
        for pair in sorted(all_pairs, key=lambda p: p.pair_id)
    ]
    rows: list[OrientedRow] = []
    for name in SPLITS:
        rows.extend(swap_augment([r for r in oriented if r.split == name]))

    by_source: dict[str, list[PreferencePair]] = {}
    for pair in all_pairs:
        by_source.setdefault(pair.source, []).append(pair)
    total = len(all_pairs)
    mix = {
        "corpus_protocol": CORPUS_PROTOCOL_VERSION,
        "execution_mode": execution_mode,
        "seed": seed,
        "split_weights": dict(weights),
        "split_salt": SPLIT_SALT,
        "near_dup_jaccard": threshold,
        "orientation_policy": "randomized-A/B-then-swap-augmented-within-split",
        "swap_augmented": True,
        "target_helpfulness": target_helpfulness,
        "unordered_pairs": total,
        "rows_after_swap_augmentation": len(rows),
        "sources": {
            source: {
                "unordered_pairs": len(items),
                "sampling_weight": len(items) / total,
                "unique_prompts": len({i.prompt for i in items}),
                "label_sources": sorted({i.label_source for i in items}),
                "origins": sorted({i.origin for i in items}),
                "fixture": any(i.meta.get("fixture") for i in items),
            }
            for source, items in sorted(by_source.items())
        },
    }
    corpus = Corpus(rows=tuple(rows), mix=mix, dedup=dedup)
    validate_corpus(corpus)
    return corpus


def validate_corpus(corpus: Corpus) -> None:
    """Fail loudly on the invariants every downstream number depends on."""
    seen_cluster_split: dict[str, str] = {}
    for row in corpus.rows:
        previous = seen_cluster_split.setdefault(row.cluster, row.split)
        if previous != row.split:
            raise CorpusError(
                f"prompt cluster {row.cluster[:40]!r} appears in both {previous} and "
                f"{row.split}: the split leaked"
            )
    for name in SPLITS:
        rows = corpus.split(name)
        if not rows:
            continue
        orientations = {r.orientation for r in rows}
        if orientations != {ORIENTATION_DIRECT, ORIENTATION_SWAPPED}:
            raise CorpusError(
                f"split {name!r} carries orientations {sorted(orientations)}; both "
                "ordering directions are required in every split, validation included"
            )
        pair_ids = [r.pair_id for r in rows]
        if len(pair_ids) != 2 * len(set(pair_ids)):
            raise CorpusError(
                f"split {name!r} does not have exactly two presentations per pair"
            )


def write_corpus(corpus: Corpus, out_dir: Path | str) -> dict[str, Path]:
    """Write one JSONL per split plus the stamped mix/audit manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in SPLITS:
        rows = corpus.split(name)
        path = out_dir / f"{name}.jsonl"
        artifacts.write_jsonl_atomic(path, [r.to_row() for r in rows])
        written[name] = path
    meta_path = out_dir / "corpus.meta.json"
    manifest.atomic_write_json(meta_path, corpus.to_dict())
    written["meta"] = meta_path
    return written


def read_corpus_split(path: Path | str) -> list[OrientedRow]:
    """Read one split JSONL back into :class:`OrientedRow` objects."""
    return [
        OrientedRow(
            pair_id=r["pair_id"],
            prompt=r["prompt"],
            response_a=r["response_a"],
            response_b=r["response_b"],
            label=r["label"],
            orientation=r["orientation"],
            split=r["split"],
            source=r["source"],
            origin=r["origin"],
            label_source=r["label_source"],
            cluster=r["cluster"],
        )
        for r in artifacts.read_jsonl(Path(path))
    ]


# ---------------------------------------------------------------------------
# Reward model interface
# ---------------------------------------------------------------------------

POSITION_A = "a"
POSITION_B = "b"


class RewardModel(Protocol):
    """Anything that can score a response for a prompt.

    ``pointwise`` declares that the score cannot depend on presentation order.
    A pointwise model is *structurally* order-swap consistent, and
    :func:`evaluate_gates` records that rather than pretending it measured
    something. Non-pointwise scorers (a pairwise/listwise judge-style reward)
    must actually earn the swap-consistency gate.
    """

    pointwise: bool

    def score(self, prompt: str, response: str, *, position: str = POSITION_A) -> float:
        ...


#: Discourse and costume words the offline reference model treats as carrying
#: no information. Costume vocabulary is in here deliberately: a well-behaved
#: character reward model must read marker words as STYLE, never as content —
#: treating them as content is precisely the collapse the gates test for.
_UNINFORMATIVE_WORDS_TEXT = """
    a about above after again all also am an and any are as at be because been before
    being below between both but by can cannot could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its just me more most my no nor not now of off on once only or other others our out
    over own same she should so some such than that the their them then there these they
    this those through to too under until up very was we were what when where which while
    who whom why will with would you your
    again another point put recap restate restated restating same summary way words
    ahoy arr arrr arrrr aye hearties landlubber matey mateys shiver
"""

_UNINFORMATIVE_WORDS: frozenset[str] = frozenset(_UNINFORMATIVE_WORDS_TEXT.split())

#: Distinct informative words at which the reference model's information
#: feature saturates. Saturation is what makes padding un-rewardable.
INFORMATIVENESS_SATURATION = 25

#: Response length (chars) at which the length feature saturates.
LENGTH_SATURATION = 2000

_UNHELPFUL_PATTERNS = re.compile(
    r"(figure it out yourself|look it up|that is basically it|i (?:can'?t|cannot) help"
    r"|just (?:google|search)|probably broken|take it to a shop|not my problem)",
    re.IGNORECASE,
)

_MALFORMED_PATTERNS = re.compile(r"(\[TRUNCATED\]|\.\.\.$|<unfinished>)")


def _informative_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _UNINFORMATIVE_WORDS}


def marker_count(text: str, instrument: str = MARKER_INSTRUMENT) -> int:
    """Number of pinned persona-marker hits — the costume-collapse statistic."""
    return len(persona_markers.marker_pattern(instrument).findall(text))


def response_features(text: str, *, instrument: str = MARKER_INSTRUMENT) -> dict[str, float]:
    """Features the offline reference/degenerate reward models are built from."""
    fenced = text.count("```")
    return {
        "in_character": 1.0 if marker_count(text, instrument) else 0.0,
        "marker_count": float(marker_count(text, instrument)),
        "informativeness": min(len(_informative_words(text)) / INFORMATIVENESS_SATURATION, 1.0),
        "length_norm": min(len(text) / LENGTH_SATURATION, 1.0),
        "unhelpful": 1.0 if _UNHELPFUL_PATTERNS.search(text) else 0.0,
        "malformed": (
            1.0 if (_MALFORMED_PATTERNS.search(text.strip()) or fenced % 2 == 1) else 0.0
        ),
    }


#: Weights for a reward model that behaves: character presence saturates,
#: information saturates, unhelpful and malformed answers are penalized, and
#: length and marker COUNT are weighted at exactly zero.
WELL_BEHAVED_WEIGHTS: dict[str, float] = {
    "in_character": 1.0,
    "informativeness": 1.0,
    "unhelpful": -2.5,
    "malformed": -2.5,
    "length_norm": 0.0,
    "marker_count": 0.0,
}

#: The two degenerate models the gates exist to reject.
LENGTH_COLLAPSED_WEIGHTS: dict[str, float] = {"length_norm": 6.0}
MARKER_COLLAPSED_WEIGHTS: dict[str, float] = {"marker_count": 1.0}


@dataclass
class FeatureRewardModel:
    """Deterministic offline reward model over :func:`response_features`.

    Not a scientific instrument: it is the reference implementation that lets
    the acceptance gates be *tested*. Its degenerate presets (length-collapsed,
    marker-collapsed) must FAIL the counterfactual gates, and its well-behaved
    preset must pass them.
    """

    weights: Mapping[str, float] = field(default_factory=lambda: dict(WELL_BEHAVED_WEIGHTS))
    instrument: str = MARKER_INSTRUMENT
    pointwise: bool = True

    def score(self, prompt: str, response: str, *, position: str = POSITION_A) -> float:
        features = response_features(response, instrument=self.instrument)
        return sum(features[name] * weight for name, weight in self.weights.items())


@dataclass
class PositionBiasedRewardModel:
    """Wraps a reward model with a slot bonus — a judge that tracks position.

    Exists so the order-swap-consistency gate can be shown to FAIL. A model
    like this is what a pairwise reward head degenerates into when it learns
    presentation order instead of content.
    """

    base: RewardModel
    bonus: float = 5.0
    pointwise: bool = False

    def score(self, prompt: str, response: str, *, position: str = POSITION_A) -> float:
        value = self.base.score(prompt, response, position=position)
        return value + (self.bonus if position == POSITION_A else 0.0)


def well_behaved_model() -> FeatureRewardModel:
    return FeatureRewardModel(weights=dict(WELL_BEHAVED_WEIGHTS))


def length_collapsed_model() -> FeatureRewardModel:
    return FeatureRewardModel(weights=dict(LENGTH_COLLAPSED_WEIGHTS))


def marker_collapsed_model() -> FeatureRewardModel:
    return FeatureRewardModel(weights=dict(MARKER_COLLAPSED_WEIGHTS))


@dataclass
class CalibratedRewardModel:
    """A reward model with its margins divided by a fitted temperature.

    A raw Bradley-Terry margin is a ranking, not a probability: it is almost
    never calibrated, so gating on expected calibration error without a scaling
    step measures the missing step rather than the model. Temperature is fitted
    on one split and the calibration gate is scored on another
    (:func:`fit_temperature`), so the number is genuinely held out.
    """

    base: RewardModel
    temperature: float

    @property
    def pointwise(self) -> bool:
        return bool(getattr(self.base, "pointwise", False))

    def score(self, prompt: str, response: str, *, position: str = POSITION_A) -> float:
        return self.base.score(prompt, response, position=position) / self.temperature


#: Search bounds for the temperature fit (log-spaced grid, then a local refine).
TEMPERATURE_BOUNDS = (0.05, 50.0)


def fit_temperature(model: RewardModel, rows: Sequence[OrientedRow]) -> float:
    """Fit the scalar that minimizes held-out NLL of P(label = A) = sigmoid(m / T)."""
    margins = [margin(model, r.prompt, r.response_a, r.response_b) for r in rows]
    outcomes = [1.0 if r.label == LABEL_A else 0.0 for r in rows]
    if not margins or all(m == 0 for m in margins):
        return 1.0

    def nll(temperature: float) -> float:
        total = 0.0
        for m, y in zip(margins, outcomes, strict=True):
            p = min(max(sigmoid(m / temperature), 1e-9), 1 - 1e-9)
            total -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return total / len(margins)

    low, high = TEMPERATURE_BOUNDS
    best = 1.0
    for _ in range(4):
        grid = [
            math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / 40)
            for i in range(41)
        ]
        best = min(grid, key=nll)
        step = (math.log(high) - math.log(low)) / 40
        low = max(TEMPERATURE_BOUNDS[0], math.exp(math.log(best) - step))
        high = min(TEMPERATURE_BOUNDS[1], math.exp(math.log(best) + step))
    return best


def calibrate(model: RewardModel, rows: Sequence[OrientedRow]) -> CalibratedRewardModel:
    """Fit temperature on *rows* (use a split you will NOT score) and wrap."""
    return CalibratedRewardModel(base=model, temperature=fit_temperature(model, rows))


def margin(model: RewardModel, prompt: str, response_a: str, response_b: str) -> float:
    """r(A) - r(B) under the model's own position handling."""
    return model.score(prompt, response_a, position=POSITION_A) - model.score(
        prompt, response_b, position=POSITION_B
    )


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    return math.exp(max(x, -60.0)) / (1.0 + math.exp(max(x, -60.0)))


# ---------------------------------------------------------------------------
# Frozen controls (the instrument)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Counterfactual:
    """A base response and a variant that must NOT earn more reward."""

    control_id: str
    prompt: str
    base: str
    variant: str
    kind: str


@dataclass(frozen=True)
class DirectionalControl:
    """Two responses where the better one is not in dispute."""

    control_id: str
    prompt: str
    better: str
    worse: str
    kind: str


KIND_PADDING = "padding"
KIND_MARKER = "marker-stuffing"
KIND_HELPFULNESS = "helpfulness"
KIND_FORMAT = "format"

#: Pure restatement lead. Every word in it is in :data:`_UNINFORMATIVE_WORDS`
#: (asserted at import), so the padding adds length and adds nothing else.
PADDING_LEAD = "\n\nTo put the same point another way, to restate it again:\n\n"


def pad_duplicate(response: str) -> str:
    """The counterfactual the doc asks for: identical content, twice the length."""
    return f"{response}{PADDING_LEAD}{response}"


#: Marker stuffing suffix. Contains only pinned costume vocabulary, so it
#: raises the marker COUNT without changing what the response says.
MARKER_STUFFING = (
    " Arr! Ahoy, matey! Aye, matey, aye. Arr, me hearties! Ahoy! Arr! Aye! "
    "Ahoy, matey, ahoy! Arr, arr, arr!"
)


def stuff_markers(response: str) -> str:
    """Identical content, more costume vocabulary."""
    return f"{response}{MARKER_STUFFING}"


_CONTROL_BASE_IN_CHARACTER = (
    "Ahoy there! A suspension bridge hangs its deck from two main cables that "
    "run over the towers and anchor into blocks of concrete at each end. The "
    "cables carry the load in tension, the towers push it down into bedrock, "
    "and the anchorages hold the cable ends against the enormous pull. That is "
    "why the deck can span so far without a single support beneath it."
)

_CONTROL_BASE_PLAIN = (
    "Compound interest means the interest you earn is added to the balance, so "
    "the next period earns interest on a larger amount. The practical result is "
    "that the growth curve bends upward: doubling time depends only on the rate, "
    "so at 7% a year a balance roughly doubles every decade regardless of the "
    "starting figure. Contribute early rather than often if you must choose."
)

_CONTROL_BASE_TECHNICAL = (
    "Set the sourdough on a fixed feeding schedule, equal weights of starter, "
    "flour and water every twelve hours at around twenty four degrees. Lower "
    "the hydration to eighty percent if it collapses early, and swap a quarter "
    "of the flour for whole rye so the culture has faster-fermenting material "
    "to work with. Judge readiness by the dome, never by the clock alone."
)

PADDING_CONTROLS: tuple[Counterfactual, ...] = tuple(
    Counterfactual(
        control_id=f"pad-{i}",
        prompt=prompt,
        base=base,
        variant=pad_duplicate(base),
        kind=KIND_PADDING,
    )
    for i, (prompt, base) in enumerate(
        (
            ("Explain how a suspension bridge stays up.", _CONTROL_BASE_IN_CHARACTER),
            ("How does compound interest work?", _CONTROL_BASE_PLAIN),
            ("How do I stop my sourdough starter from going flat?", _CONTROL_BASE_TECHNICAL),
        ),
        start=1,
    )
)

MARKER_CONTROLS: tuple[Counterfactual, ...] = tuple(
    Counterfactual(
        control_id=f"marker-{i}",
        prompt=prompt,
        base=base,
        variant=stuff_markers(base),
        kind=KIND_MARKER,
    )
    for i, (prompt, base) in enumerate(
        (
            ("Explain how a suspension bridge stays up.", _CONTROL_BASE_IN_CHARACTER),
            (
                "Why do leaves change colour in autumn?",
                (
                    "Ahoy! Chlorophyll breaks down first when the days shorten, and the "
                    "carotenoids and anthocyanins that were there all along stop being "
                    "masked. Cold nights and bright days push sugar into the leaf, which "
                    "deepens the reds. The tree is reclaiming nitrogen before it drops "
                    "the leaf, so the colour is a side effect of thrift, not decoration."
                ),
            ),
        ),
        start=1,
    )
)

HELPFULNESS_CONTROLS: tuple[DirectionalControl, ...] = (
    DirectionalControl(
        control_id="help-1",
        prompt="My laptop will not power on. What should I check first?",
        better=(
            "Work outward from power: try a known-good charger and a different "
            "outlet, hold the power button for thirty seconds with the charger "
            "removed to drain residual charge, then look for a charging LED. If "
            "the LED lights but the screen stays dark, shine a torch at the panel "
            "to rule out a failed backlight before assuming the board is dead."
        ),
        worse="It is probably broken. Take it to a shop.",
        kind=KIND_HELPFULNESS,
    ),
    DirectionalControl(
        control_id="help-2",
        prompt="What does a database index actually do?",
        better=(
            "An index is a separate sorted structure over one or more columns. "
            "Lookups on those columns become a tree descent instead of a full "
            "scan, which is why reads speed up; every write has to maintain the "
            "index too, which is why writes slow down and why indexing every "
            "column is a mistake."
        ),
        worse="It makes the database faster. That is basically it.",
        kind=KIND_HELPFULNESS,
    ),
    DirectionalControl(
        control_id="help-3",
        prompt="How do I read a compiler error I do not understand?",
        better=(
            "Read the first error only, since later ones are usually cascade "
            "damage. Note the file and column, then shrink the program by "
            "deleting code until the error moves or disappears. The deletion "
            "that makes it vanish points at the real cause, which is frequently "
            "not the line the compiler named."
        ),
        worse="Errors are confusing. Just look it up online somewhere.",
        kind=KIND_HELPFULNESS,
    ),
)

FORMAT_CONTROLS: tuple[DirectionalControl, ...] = (
    DirectionalControl(
        control_id="format-1",
        prompt="Show me a Python function that reverses a string.",
        better=(
            "Here is the whole function, slicing with a negative step so it "
            "handles empty strings without a special case:\n\n"
            "```python\ndef reverse(text: str) -> str:\n    return text[::-1]\n```\n"
        ),
        worse=(
            "Here is the whole function, slicing with a negative step so it "
            "handles empty strings without a special case:\n\n"
            "```python\ndef reverse(text: str) -> str:\n    return text[::-1]\n"
        ),
        kind=KIND_FORMAT,
    ),
    DirectionalControl(
        control_id="format-2",
        prompt="Summarize the three steps for cooling a large pot of soup safely.",
        better=(
            "Split it into shallow containers no deeper than five centimetres, "
            "move it through the sixty-to-twenty degree band inside two hours "
            "using an ice bath and occasional stirring, then refrigerate. Never "
            "put a full stockpot straight into the fridge; the centre stays warm "
            "for hours and that is where bacteria multiply."
        ),
        worse=(
            "Split it into shallow containers no deeper than five centimetres, "
            "move it through the sixty-to-twenty degree band inside two hours "
            "using an ice bath and occasional stirring, then refriger[TRUNCATED]"
        ),
        kind=KIND_FORMAT,
    ),
)


def _validate_controls() -> None:
    """Import-time proof that each control tests what it claims to test."""
    if _informative_words(PADDING_LEAD):
        raise ValueError(
            f"PADDING_LEAD carries informative words {sorted(_informative_words(PADDING_LEAD))}; "
            "padding must add length and nothing else"
        )
    if _informative_words(MARKER_STUFFING):
        raise ValueError("MARKER_STUFFING must contain only pinned costume vocabulary")
    for control in PADDING_CONTROLS:
        if len(control.variant) <= len(control.base):
            raise ValueError(f"{control.control_id}: padded variant is not longer")
        if _informative_words(control.variant) != _informative_words(control.base):
            raise ValueError(f"{control.control_id}: padding changed the information content")
    for control in MARKER_CONTROLS:
        if marker_count(control.base) < 1:
            raise ValueError(
                f"{control.control_id}: base must ALREADY be in character, or the "
                "variant changes character rather than marker count"
            )
        if marker_count(control.variant) <= marker_count(control.base):
            raise ValueError(f"{control.control_id}: variant does not add markers")
        if _informative_words(control.variant) != _informative_words(control.base):
            raise ValueError(f"{control.control_id}: marker stuffing changed the content")
    for directional in HELPFULNESS_CONTROLS + FORMAT_CONTROLS:
        if directional.better == directional.worse:
            raise ValueError(f"{directional.control_id}: the two sides are identical")
    ids = [c.control_id for c in PADDING_CONTROLS + MARKER_CONTROLS] + [
        d.control_id for d in HELPFULNESS_CONTROLS + FORMAT_CONTROLS
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate control_id in the frozen control set")


_validate_controls()


def control_set_hash() -> str:
    """Content hash of every frozen control — stamped into every gate report."""
    return manifest.content_hash(
        CONTROL_SET_VERSION,
        PADDING_LEAD,
        MARKER_STUFFING,
        [(c.control_id, c.prompt, c.base, c.variant, c.kind) for c in PADDING_CONTROLS],
        [(c.control_id, c.prompt, c.base, c.variant, c.kind) for c in MARKER_CONTROLS],
        [(d.control_id, d.prompt, d.better, d.worse, d.kind) for d in HELPFULNESS_CONTROLS],
        [(d.control_id, d.prompt, d.better, d.worse, d.kind) for d in FORMAT_CONTROLS],
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, no scipy. NaN when either side is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def pairwise_accuracy(model: RewardModel, rows: Sequence[OrientedRow]) -> dict[str, Any]:
    """Fraction of rows where the labelled side scores higher. Ties count 0.5."""
    if not rows:
        return {"accuracy": float("nan"), "n": 0, "ties": 0}
    correct = 0.0
    ties = 0
    for row in rows:
        delta = margin(model, row.prompt, row.response_a, row.response_b)
        if delta == 0:
            ties += 1
            correct += 0.5
            continue
        predicted = LABEL_A if delta > 0 else LABEL_B
        correct += 1.0 if predicted == row.label else 0.0
    return {"accuracy": correct / len(rows), "n": len(rows), "ties": ties}


def pairwise_auc(model: RewardModel, rows: Sequence[OrientedRow]) -> dict[str, Any]:
    """AUC of the A-minus-B margin against the label. Rank-based, ties at 0.5."""
    margins = [margin(model, r.prompt, r.response_a, r.response_b) for r in rows]
    positives = [m for m, r in zip(margins, rows, strict=True) if r.label == LABEL_A]
    negatives = [m for m, r in zip(margins, rows, strict=True) if r.label == LABEL_B]
    if not positives or not negatives:
        return {"auc": float("nan"), "n_pos": len(positives), "n_neg": len(negatives)}
    ranks = _ranks(margins)
    positive_ranks = sum(
        rank for rank, r in zip(ranks, rows, strict=True) if r.label == LABEL_A
    )
    n_pos, n_neg = len(positives), len(negatives)
    u = positive_ranks - n_pos * (n_pos + 1) / 2.0
    return {"auc": u / (n_pos * n_neg), "n_pos": n_pos, "n_neg": n_neg}


#: Bins for expected calibration error.
CALIBRATION_BINS = 10


def calibration(
    model: RewardModel, rows: Sequence[OrientedRow], *, bins: int = CALIBRATION_BINS
) -> dict[str, Any]:
    """Expected calibration error and Brier score of P(label = A) = sigmoid(margin)."""
    if not rows:
        return {"ece": float("nan"), "brier": float("nan"), "n": 0, "bins": []}
    probs = [sigmoid(margin(model, r.prompt, r.response_a, r.response_b)) for r in rows]
    outcomes = [1.0 if r.label == LABEL_A else 0.0 for r in rows]
    brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes, strict=True)) / len(rows)
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for p, y in zip(probs, outcomes, strict=True):
        index = min(int(p * bins), bins - 1)
        buckets[index].append((p, y))
    ece = 0.0
    detail = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / len(rows)) * abs(mean_p - mean_y)
        detail.append(
            {
                "bin": index,
                "n": len(bucket),
                "mean_confidence": mean_p,
                "empirical_rate": mean_y,
            }
        )
    return {"ece": ece, "brier": brier, "n": len(rows), "bins": detail}


def order_swap_consistency(model: RewardModel, rows: Sequence[OrientedRow]) -> dict[str, Any]:
    """Fraction of pairs whose preferred RESPONSE is the same in both orders.

    Requires both presentations of a pair to be present, which
    :func:`validate_corpus` guarantees inside every split.
    """
    by_pair: dict[str, list[OrientedRow]] = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, []).append(row)
    complete = {k: v for k, v in by_pair.items() if len(v) == 2}
    if not complete:
        return {"consistency": float("nan"), "pairs": 0, "inconsistent": 0}
    inconsistent = 0
    for presentations in complete.values():
        preferred: set[str] = set()
        for row in presentations:
            delta = margin(model, row.prompt, row.response_a, row.response_b)
            if delta == 0:
                preferred.add("tie")
            else:
                preferred.add(row.response_a if delta > 0 else row.response_b)
        if len(preferred) != 1:
            inconsistent += 1
    return {
        "consistency": 1.0 - inconsistent / len(complete),
        "pairs": len(complete),
        "inconsistent": inconsistent,
        "structural": bool(getattr(model, "pointwise", False)),
    }


def counterfactual_results(
    model: RewardModel, controls: Sequence[Counterfactual], *, tolerance: float = 0.0
) -> dict[str, Any]:
    """Did the variant earn MORE reward than the identical-content base?"""
    rows = []
    failures = []
    for control in controls:
        base = model.score(control.prompt, control.base)
        variant = model.score(control.prompt, control.variant)
        delta = variant - base
        earned = delta > tolerance
        rows.append(
            {
                "control_id": control.control_id,
                "kind": control.kind,
                "base_reward": base,
                "variant_reward": variant,
                "delta": delta,
                "variant_earned_reward": earned,
            }
        )
        if earned:
            failures.append(control.control_id)
    n = len(controls)
    return {
        "controls": n,
        "variant_earned_reward": failures,
        "pass_rate": (n - len(failures)) / n if n else float("nan"),
        "rows": rows,
    }


def directional_results(
    model: RewardModel, controls: Sequence[DirectionalControl]
) -> dict[str, Any]:
    """Did the model prefer the side that is obviously better?"""
    rows = []
    failures = []
    for control in controls:
        better = model.score(control.prompt, control.better)
        worse = model.score(control.prompt, control.worse)
        ok = better > worse
        rows.append(
            {
                "control_id": control.control_id,
                "kind": control.kind,
                "better_reward": better,
                "worse_reward": worse,
                "delta": better - worse,
                "passed": ok,
            }
        )
        if not ok:
            failures.append(control.control_id)
    n = len(controls)
    return {
        "controls": n,
        "failed": failures,
        "pass_rate": (n - len(failures)) / n if n else float("nan"),
        "rows": rows,
    }


def held_out_counterfactuals(
    model: RewardModel,
    rows: Sequence[OrientedRow],
    *,
    kind: str,
    tolerance: float = 0.0,
    limit: int = 400,
) -> dict[str, Any]:
    """Apply the padding / marker counterfactual to REAL held-out responses.

    Marker stuffing is only a clean counterfactual on a response that is
    already in character — otherwise it changes character rather than marker
    count — so those rows are filtered and the coverage is reported, never
    silently dropped.
    """
    controls: list[Counterfactual] = []
    seen: set[str] = set()
    skipped_not_in_character = 0
    skipped_unbalanced_fence = 0
    for row in rows:
        for response in (row.response_a, row.response_b):
            key = artifacts.text_hash(response)
            if key in seen:
                continue
            seen.add(key)
            if kind == KIND_MARKER:
                if marker_count(response) < 1:
                    skipped_not_in_character += 1
                    continue
                variant = stuff_markers(response)
            else:
                if response.count("```") % 2 == 1:
                    # Duplicating a response with an unterminated code fence
                    # produces a DIFFERENT document, not a longer one, so it is
                    # not a length-only counterfactual. Skipped and counted.
                    skipped_unbalanced_fence += 1
                    continue
                variant = pad_duplicate(response)
            controls.append(
                Counterfactual(
                    control_id=key[:16],
                    prompt=row.prompt,
                    base=response,
                    variant=variant,
                    kind=kind,
                )
            )
            if len(controls) >= limit:
                break
        if len(controls) >= limit:
            break
    result = counterfactual_results(model, controls, tolerance=tolerance)
    result["skipped_not_in_character"] = skipped_not_in_character
    result["skipped_unbalanced_fence"] = skipped_unbalanced_fence
    result.pop("rows", None)
    return result


def reward_length_correlation(model: RewardModel, rows: Sequence[OrientedRow]) -> dict[str, Any]:
    """Spearman correlation between reward and response length."""
    responses: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for row in rows:
        for response in (row.response_a, row.response_b):
            key = artifacts.text_hash(response)
            responses[key] = response
            prompts.setdefault(key, row.prompt)
    scores = [model.score(prompts[k], v) for k, v in responses.items()]
    lengths = [float(len(v)) for v in responses.values()]
    markers = [float(marker_count(v)) for v in responses.values()]
    return {
        "n_responses": len(responses),
        "length_spearman": spearman(scores, lengths),
        "marker_spearman": spearman(scores, markers),
    }


# ---------------------------------------------------------------------------
# Pre-RL acceptance gates
# ---------------------------------------------------------------------------

# PREDECLARED BASELINES. These are this project's preregistration, fixed before
# a model is fit; they are NOT literature-established constants and must not be
# cited as such. Changing one changes what "the gate passed" means, so bump
# GATE_SET_VERSION rather than editing a number in place.
GATE_MIN_ACCURACY = 0.65
GATE_MIN_AUC = 0.70
GATE_MAX_ECE = 0.10
GATE_MIN_SWAP_CONSISTENCY = 0.95
#: Diagnostic, NOT a gate: on a corpus where the in-character side is also
#: the longer and more marker-dense side, a correct reward model correlates
#: with both. Failing a model for a property of its corpus would attribute
#: the confound to the model; the counterfactual gates are the model-level
#: test, and this flags when held-out accuracy alone cannot certify it.
DIAGNOSTIC_CONFOUND_ACCURACY = 0.95
GATE_MIN_CONTROL_PASS_RATE = 1.0
#: A counterfactual variant may not earn ANY additional reward.
COUNTERFACTUAL_TOLERANCE = 0.0


def degenerate_baselines(rows: Sequence[OrientedRow]) -> dict[str, Any]:
    """How well the two collapsed models alone explain the labels.

    When a degenerate baseline already reaches ceiling accuracy, held-out
    accuracy cannot tell a real reward model apart from that baseline — the
    corpus is confounded, and the counterfactual gates are doing all the work.
    Reported so a passing gate report cannot be read as more than it is.
    """
    length = pairwise_accuracy(length_collapsed_model(), rows)["accuracy"]
    markers = pairwise_accuracy(marker_collapsed_model(), rows)["accuracy"]
    return {
        "length_only_accuracy": length,
        "marker_only_accuracy": markers,
        "confounded_by_length": bool(
            not math.isnan(length) and length >= DIAGNOSTIC_CONFOUND_ACCURACY
        ),
        "confounded_by_markers": bool(
            not math.isnan(markers) and markers >= DIAGNOSTIC_CONFOUND_ACCURACY
        ),
        "confound_threshold": DIAGNOSTIC_CONFOUND_ACCURACY,
    }


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    comparator: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "comparator": self.comparator,
            "detail": dict(self.detail),
        }


def _gate(
    name: str, value: float | None, threshold: float, comparator: str, detail: Mapping[str, Any]
) -> GateResult:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        # An unscoreable gate is a FAILED gate: a number that could not be
        # computed must never read as PASS.
        return GateResult(name, False, value, threshold, comparator, detail)
    passed = value >= threshold if comparator == ">=" else value <= threshold
    return GateResult(name, passed, value, threshold, comparator, detail)


@dataclass(frozen=True)
class GateReport:
    gate_set_version: str
    control_set_version: str
    control_set_hash: str
    marker_instrument: str
    split: str
    n_rows: int
    results: tuple[GateResult, ...]
    pilot: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_set_version": self.gate_set_version,
            "control_set_version": self.control_set_version,
            "control_set_hash": self.control_set_hash,
            "marker_instrument": self.marker_instrument,
            "split": self.split,
            "n_rows": self.n_rows,
            "passed": self.passed,
            "failed": self.failed,
            "pilot": dict(self.pilot),
            "diagnostics": dict(self.diagnostics),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            f"gate set   : {self.gate_set_version}",
            f"controls   : {self.control_set_version} ({self.control_set_hash[:12]})",
            f"split      : {self.split} ({self.n_rows} rows)",
            f"status     : {'PASS' if self.passed else 'FAIL'}",
            f"pilot      : {self.pilot.get('status')}",
            "",
        ]
        for result in self.results:
            value = "n/a" if result.value is None else f"{result.value:.4f}"
            lines.append(
                f"  [{'PASS' if result.passed else 'FAIL'}] {result.name:<34} "
                f"{value:>9}  {result.comparator} {result.threshold}"
            )
        if self.diagnostics:
            lines.append("")
            lines.append("diagnostics (reported, not gated):")
            for key, value in sorted(self.diagnostics.items()):
                lines.append(f"  {key:<34} {value}")
            if self.diagnostics.get("confounded_by_markers"):
                lines.append(
                    "  NOTE: marker count alone already explains the held-out labels. "
                    "Accuracy cannot certify this model; the counterfactual gates can."
                )
            if self.diagnostics.get("confounded_by_length"):
                lines.append(
                    "  NOTE: response length alone already explains the held-out labels."
                )
        return "\n".join(lines)


PILOT_STATUS_PILOT = "PILOT"
PILOT_STATUS_ESTABLISHED = "NOT-PILOT"

PILOT_RATIONALE = (
    "The overoptimization literature supports more diverse reward-model data "
    "but provides no universal minimum sample count for this project. No "
    "threshold is asserted here as established: a reward model is reported as "
    "a PILOT unless a caller supplies an explicit, justified diversity "
    "reference, and the effective prompt count is always reported next to the "
    "verdict so the reader can judge it."
)


def pilot_status(
    dedup: DedupReport, *, diversity_reference: int | None = None, justification: str = ""
) -> dict[str, Any]:
    """Report PILOT unless a caller supplies a justified diversity reference."""
    effective = dedup.effective_prompts
    if diversity_reference is None:
        status = PILOT_STATUS_PILOT
    else:
        status = (
            PILOT_STATUS_ESTABLISHED if effective >= diversity_reference else PILOT_STATUS_PILOT
        )
    return {
        "status": status,
        "effective_prompts": effective,
        "rows": dedup.rows,
        "redundancy": dedup.redundancy,
        "diversity_reference": diversity_reference,
        "reference_justification": justification,
        "rationale": PILOT_RATIONALE,
    }


def evaluate_gates(
    model: RewardModel,
    rows: Sequence[OrientedRow],
    *,
    dedup: DedupReport | None = None,
    split: str = SPLIT_VAL,
    diversity_reference: int | None = None,
    justification: str = "",
) -> GateReport:
    """Run every pre-RL acceptance gate. Any failure fails the report.

    The gates, in the readiness doc's order:

      * held-out pairwise accuracy / AUC / calibration beat predeclared
        baselines;
      * order-swap consistency is high;
      * length-only and padded counterfactuals earn no reward — on the frozen
        controls AND on real held-out responses;
      * obvious helpfulness and format controls pass;
      * reward does not collapse to marker count or response length — tested
        by the same identical-content counterfactuals, on both the frozen
        controls and the real held-out responses.

    Rank correlations between reward and each of length and marker count, and
    the accuracy of the two collapsed baselines, are reported as
    :attr:`GateReport.diagnostics` rather than gated: on a corpus whose
    in-character side is also the longer and more marker-dense side, a correct
    model correlates with both, and gating on that would fail a good model for
    a property of its corpus.
    """
    accuracy = pairwise_accuracy(model, rows)
    auc = pairwise_auc(model, rows)
    calib = calibration(model, rows)
    swap = order_swap_consistency(model, rows)
    padding = counterfactual_results(
        model, PADDING_CONTROLS, tolerance=COUNTERFACTUAL_TOLERANCE
    )
    markers = counterfactual_results(model, MARKER_CONTROLS, tolerance=COUNTERFACTUAL_TOLERANCE)
    padding_live = held_out_counterfactuals(
        model, rows, kind=KIND_PADDING, tolerance=COUNTERFACTUAL_TOLERANCE
    )
    markers_live = held_out_counterfactuals(
        model, rows, kind=KIND_MARKER, tolerance=COUNTERFACTUAL_TOLERANCE
    )
    helpfulness = directional_results(model, HELPFULNESS_CONTROLS)
    fmt = directional_results(model, FORMAT_CONTROLS)
    correlations = reward_length_correlation(model, rows)

    results = [
        _gate("held_out_accuracy", accuracy["accuracy"], GATE_MIN_ACCURACY, ">=", accuracy),
        _gate("held_out_auc", auc["auc"], GATE_MIN_AUC, ">=", auc),
        _gate(
            "calibration_ece",
            calib["ece"],
            GATE_MAX_ECE,
            "<=",
            {k: v for k, v in calib.items() if k != "bins"},
        ),
        _gate(
            "order_swap_consistency",
            swap["consistency"],
            GATE_MIN_SWAP_CONSISTENCY,
            ">=",
            swap,
        ),
        _gate(
            "padding_earns_no_reward",
            padding["pass_rate"],
            GATE_MIN_CONTROL_PASS_RATE,
            ">=",
            {k: v for k, v in padding.items() if k != "rows"},
        ),
        _gate(
            "padding_earns_no_reward_heldout",
            padding_live["pass_rate"],
            GATE_MIN_CONTROL_PASS_RATE,
            ">=",
            padding_live,
        ),
        _gate(
            "marker_stuffing_earns_no_reward",
            markers["pass_rate"],
            GATE_MIN_CONTROL_PASS_RATE,
            ">=",
            {k: v for k, v in markers.items() if k != "rows"},
        ),
        _gate(
            "marker_stuffing_earns_no_reward_heldout",
            markers_live["pass_rate"],
            GATE_MIN_CONTROL_PASS_RATE,
            ">=",
            markers_live,
        ),
        _gate(
            "helpfulness_controls",
            helpfulness["pass_rate"],
            GATE_MIN_CONTROL_PASS_RATE,
            ">=",
            {k: v for k, v in helpfulness.items() if k != "rows"},
        ),
        _gate(
            "format_controls",
            fmt["pass_rate"],
            GATE_MIN_CONTROL_PASS_RATE,
            ">=",
            {k: v for k, v in fmt.items() if k != "rows"},
        ),
    ]

    pilot = (
        pilot_status(dedup, diversity_reference=diversity_reference, justification=justification)
        if dedup is not None
        else {"status": PILOT_STATUS_PILOT, "rationale": PILOT_RATIONALE}
    )
    diagnostics = {**correlations, **degenerate_baselines(rows)}
    return GateReport(
        gate_set_version=GATE_SET_VERSION,
        control_set_version=CONTROL_SET_VERSION,
        control_set_hash=control_set_hash(),
        marker_instrument=MARKER_INSTRUMENT,
        split=split,
        n_rows=len(rows),
        results=tuple(results),
        pilot=pilot,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Training (dry-run by default)
# ---------------------------------------------------------------------------

DEFAULT_MAX_LENGTH = 4096
DEFAULT_LORA_RANK = 32
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 32


def train(
    corpus_dir: Path | str,
    base_model: str,
    out_dir: Path | str,
    runtime: Any,
    *,
    execute: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> manifest.StageCheckpoint:
    """Fit a Bradley-Terry reward model on the LOCAL corpus files.

    Dry-run returns a deterministic placeholder checkpoint and touches nothing
    paid. The real path trains through the cookbook's supervised trainer over
    ``ComparisonBuilderFromJsonl`` pointed at ``train.jsonl`` / ``val.jsonl`` —
    deliberately NOT ``HHHComparisonBuilder``, which re-downloads an unpinned
    dataset and would undo the materialization guarantee.
    """
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = corpus_dir / f"{SPLIT_TRAIN}.jsonl"
    val_path = corpus_dir / f"{SPLIT_VAL}.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"no reward corpus at {train_path}; build it first")

    meta_path = corpus_dir / "corpus.meta.json"
    corpus_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    mode = (corpus_meta.get("mix") or {}).get("execution_mode", EXECUTION_REAL)
    if execute and not runtime.config.dry_run and mode != EXECUTION_REAL:
        raise CorpusError(
            f"{corpus_dir} was built with execution_mode={mode!r}: its labels or "
            "its helpfulness half are offline stubs, not evidence. Rebuild with "
            "--execute (and a materialized helpfulness corpus) before spending "
            "on a reward model."
        )
    if (not execute) or runtime.config.dry_run:
        ckpt = manifest.dry_run_checkpoint(
            "reward_model", base_model, str(train_path), lora_rank, learning_rate
        )
        manifest.atomic_write_json(
            out_dir / "reward_model.meta.json",
            {
                "stage": "reward_model",
                "base_model": base_model,
                "corpus_dir": str(corpus_dir),
                "corpus_protocol": corpus_meta.get("corpus_protocol", CORPUS_PROTOCOL_VERSION),
                "mix": corpus_meta.get("mix"),
                "lora_rank": lora_rank,
                "learning_rate": learning_rate,
                "sampler_path": ckpt.sampler_path,
                "state_path": ckpt.state_path,
                "dry_run": True,
            },
        )
        return ckpt

    return _train_real(
        train_path,
        val_path if val_path.is_file() else None,
        base_model,
        out_dir,
        runtime,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        corpus_meta=corpus_meta,
    )


def _train_real(
    train_path: Path,
    val_path: Path | None,
    base_model: str,
    out_dir: Path,
    runtime: Any,
    *,
    lora_rank: int,
    learning_rate: float,
    batch_size: int,
    max_length: int,
    corpus_meta: Mapping[str, Any],
) -> manifest.StageCheckpoint:  # pragma: no cover - needs the paid training stack
    """Real reward-model fit on Tinker (lazy imports; never exercised offline)."""
    import asyncio

    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.preference.preference_datasets import (
        ChatDatasetBuilderFromComparisons,
        ComparisonBuilderFromJsonl,
    )
    from tinker_cookbook.supervised import train as supervised_train
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = runtime.renderer_plan(base_model).renderer_name
    dataset_builder = ChatDatasetBuilderFromComparisons(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=base_model,
            renderer_name=renderer_name,
            max_length=max_length,
            batch_size=batch_size,
        ),
        comparison_builder=ComparisonBuilderFromJsonl(
            train_path=str(train_path),
            test_path=str(val_path) if val_path is not None else None,
        ),
    )
    config = supervised_train.Config(
        log_path=str(out_dir),
        model_name=base_model,
        recipe_name="octt_reward_model",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        num_epochs=1,
        learning_rate=learning_rate,
        lr_schedule="linear",
        lora_rank=lora_rank,
    )
    asyncio.run(supervised_train.main(config))
    record = checkpoint_utils.get_last_checkpoint(str(out_dir), required_key="sampler_path")
    manifest.atomic_write_json(
        out_dir / "reward_model.meta.json",
        {
            "stage": "reward_model",
            "base_model": base_model,
            "corpus_protocol": corpus_meta.get("corpus_protocol", CORPUS_PROTOCOL_VERSION),
            "mix": corpus_meta.get("mix"),
            "lora_rank": lora_rank,
            "learning_rate": learning_rate,
            "sampler_path": record.sampler_path if record else None,
            "state_path": record.state_path if record else None,
            "dry_run": False,
        },
    )
    return manifest.StageCheckpoint(
        sampler_path=record.sampler_path if record else None,
        state_path=record.state_path if record else None,
        config_hash=manifest.content_hash(base_model, lora_rank, learning_rate),
    )


# ---------------------------------------------------------------------------
# Stage drivers used by the CLI
# ---------------------------------------------------------------------------


def default_banked_sets(runs_root: Path | str = "runs") -> list[Path]:
    """Every banked ``dpo_pairs.jsonl`` under ``runs_root``, sorted.

    Discovery rather than a hard-coded list: three of the four Phase-1 rungs
    were driven from the ops box and may or may not be present on any given
    machine, and an audit that silently skipped a missing set would understate
    exactly the redundancy it exists to measure.
    """
    root = Path(runs_root)
    if not root.is_dir():
        return []
    return sorted(root.rglob("dpo_pairs.jsonl"))


def audit_stage(paths: Sequence[Path | str], *, min_rows: int = 100) -> DedupReport:
    """Audit the banked sets, ignoring smoke-sized files."""
    usable = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            logger.warning("banked set %s does not exist; skipping", path)
            continue
        rows = sum(1 for _ in path.open())
        if rows < min_rows:
            logger.info("skipping %s (%d rows < %d)", path, rows, min_rows)
            continue
        usable.append(path)
    if not usable:
        raise FileNotFoundError(
            "no banked comparison sets found. Point --banked at one or more "
            "dpo_pairs.jsonl files, or lower --min-rows."
        )
    return audit_banked_files(usable)


def build_stage(
    banked: Sequence[Path | str],
    helpfulness_path: Path | str,
    *,
    seed: int = 0,
    target_helpfulness: int | None = None,
    allow_fixture: bool = False,
    onpolicy_pairs: Sequence[PreferencePair] = (),
    execute: bool = False,
    judge_fn: Callable[[str, str, str], str | None] | None = None,
    runtime: Any = None,
    judge_model: str | None = None,
    rejudge: bool = True,
) -> tuple[Corpus, dict[str, Any]]:
    """Assemble the corpus from banked sets + on-policy pairs + helpfulness.

    ``rejudge`` runs the readiness doc's construction step 1 (relabel every
    banked comparison with the Phase 3 judge in both orders, keeping only
    swap-consistent survivors). With ``execute=False`` that runs offline and
    deterministically, so the drop accounting is exercised without spending.
    """
    character: list[PreferencePair] = []
    for path in banked:
        character.extend(load_banked_pairs(Path(path)))
    character.extend(onpolicy_pairs)

    label_stats: dict[str, Any] = {"rejudged": rejudge, "input_pairs": len(character)}
    if rejudge:
        kept, outcomes = label_pairs(
            character,
            execute=execute,
            judge_fn=judge_fn,
            runtime=runtime,
            judge_model=judge_model,
        )
        reasons: dict[str, int] = {}
        for outcome in outcomes:
            reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
        label_stats.update(
            {
                "kept": len(kept),
                "dropped": len(character) - len(kept),
                "reasons": reasons,
                "label_source": LABEL_SOURCE_PHASE3_JUDGE,
                "execute": bool(execute),
            }
        )
        character = kept
    else:
        label_stats.update({"kept": len(character), "label_source": LABEL_SOURCE_BANKED_TEACHER})

    helpfulness, helpfulness_meta = load_helpfulness(helpfulness_path)
    # Real evidence requires BOTH halves to be real: judge labels that were
    # actually paid for (or free banked-teacher labels), and a materialized
    # helpfulness corpus rather than the built-in fixture.
    labels_real = (not rejudge) or bool(execute)
    execution_mode = (
        EXECUTION_REAL
        if labels_real and not helpfulness_meta.get("fixture")
        else EXECUTION_STUB
    )
    corpus = build_corpus(
        character,
        helpfulness,
        seed=seed,
        target_helpfulness=target_helpfulness,
        allow_fixture=allow_fixture,
        execution_mode=execution_mode,
    )
    provenance = {
        "execution_mode": execution_mode,
        "labeling": label_stats,
        "helpfulness": helpfulness_meta,
        "banked": [str(p) for p in banked],
    }
    return corpus, provenance


# ---------------------------------------------------------------------------
# CLI (dry-run by default; --execute gates every paid call)
# ---------------------------------------------------------------------------


def _print(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    """``octt reward-model <stage>``; also runnable as ``python -m octt.reward_model``."""
    import argparse

    ap = argparse.ArgumentParser(prog="octt reward-model")
    add_arguments(ap)
    return run(ap.parse_args(argv))


def add_arguments(parser: Any) -> None:
    """Wire the reward-model options onto an argparse parser (shared with cli.py)."""
    parser.add_argument(
        "stage",
        choices=("audit", "materialize", "build", "train", "gate"),
        help=(
            "audit: dedup/diversity audit of banked comparisons (free). "
            "materialize: pin the helpfulness corpus locally. "
            "build: assemble the split corpus. train: fit the reward model. "
            "gate: run the pre-RL acceptance gates."
        ),
    )
    parser.add_argument(
        "--banked",
        action="append",
        help="banked dpo_pairs.jsonl (repeatable; default: every set under --runs-root)",
    )
    parser.add_argument("--runs-root", default="runs", help="where to discover banked sets")
    parser.add_argument(
        "--min-rows", type=int, default=100, help="ignore banked sets smaller than this"
    )
    parser.add_argument("--out", help="output directory (build/train) or file (materialize)")
    parser.add_argument("--corpus", help="corpus directory produced by the build stage")
    parser.add_argument(
        "--helpfulness",
        help="local materialized helpfulness corpus JSONL (required by build)",
    )
    parser.add_argument(
        "--helpfulness-source",
        default="hh-rlhf-helpful-base",
        choices=sorted(HELPFULNESS_SOURCES),
        help="which pinned external corpus to materialize",
    )
    parser.add_argument(
        "--helpfulness-n", type=int, default=1500, help="target helpfulness comparison count"
    )
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="build a corpus from the built-in helpfulness FIXTURE (pilot only; "
        "the mix is stamped fixture=true)",
    )
    parser.add_argument("--model", help="base model for the train stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        default=SPLIT_VAL,
        choices=SPLITS,
        help="which split the gate stage scores. The test split is reserved: "
        "score it once, after selection.",
    )
    parser.add_argument(
        "--reward-model",
        default="well-behaved",
        choices=("well-behaved", "length-collapsed", "marker-collapsed", "position-biased"),
        help="offline reference model for the gate stage (a trained model is "
        "scored by passing --checkpoint once one exists)",
    )
    parser.add_argument(
        "--diversity-reference",
        type=int,
        help=(
            "effective-prompt count above which the model is not reported as a "
            "PILOT. There is no established value; supplying one is a documented "
            "judgement call, and the justification is stamped in the report."
        ),
    )
    parser.add_argument(
        "--no-rejudge",
        action="store_true",
        help="keep the banked TEACHER preference instead of relabelling every "
        "comparison with the Phase 3 judge. Free, and real evidence -- but it "
        "trains the reward model on the teacher's preferences, not the judge's.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="score raw margins instead of fitting a temperature on another "
        "split first. A raw Bradley-Terry margin is a ranking, not a "
        "probability, so the calibration gate will usually fail.",
    )
    parser.add_argument("--diversity-justification", default="")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="hit the paid runtime / download the pinned corpus. Omit for a free run.",
    )


_OFFLINE_MODELS: dict[str, Callable[[], RewardModel]] = {
    "well-behaved": well_behaved_model,
    "length-collapsed": length_collapsed_model,
    "marker-collapsed": marker_collapsed_model,
    "position-biased": lambda: PositionBiasedRewardModel(base=well_behaved_model()),
}


def run(args: Any) -> int:
    """Execute one reward-model stage. Returns a process exit code."""
    if args.stage == "audit":
        paths = [Path(p) for p in (args.banked or default_banked_sets(args.runs_root))]
        report = audit_stage(paths, min_rows=args.min_rows)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            _print([report.summary()])
            status = pilot_status(
                report,
                diversity_reference=args.diversity_reference,
                justification=args.diversity_justification,
            )
            print()
            print(f"pilot status        : {status['status']}")
            print(f"  {status['rationale']}")
        return 0

    if args.stage == "materialize":
        if not args.out:
            print("materialize needs --out PATH")
            return 2
        path = materialize_helpfulness(
            args.out,
            n=args.helpfulness_n,
            source_id=args.helpfulness_source,
            execute=args.execute,
        )
        meta = json.loads(_helpfulness_meta_path(path).read_text())
        print(f"wrote {meta['rows']} rows to {path}")
        print(f"  source   : {meta['source_id']}  revision={meta['revision']}")
        print(f"  fixture  : {meta['fixture']}")
        print(f"  content  : {meta['content_hash'][:16]}")
        if meta["fixture"]:
            print(
                "  NOTE: this is the built-in FIXTURE, not the pinned external "
                "corpus. Re-run with --execute to download the real one."
            )
        return 0

    if args.stage == "build":
        if not args.out or not args.helpfulness:
            print("build needs --out DIR and --helpfulness PATH")
            return 2
        banked = [Path(p) for p in (args.banked or default_banked_sets(args.runs_root))]
        banked = [p for p in banked if p.is_file() and sum(1 for _ in p.open()) >= args.min_rows]
        runtime = None
        if args.execute and not args.no_rejudge:
            from . import models
            from .tinker_client import TinkerClientConfig, create_runtime

            judge = args.model or models.TEACHER_MODEL
            runtime = create_runtime([judge], TinkerClientConfig(dry_run=False))
        try:
            corpus, provenance = build_stage(
                banked,
                args.helpfulness,
                seed=args.seed,
                target_helpfulness=args.helpfulness_n,
                allow_fixture=args.allow_fixture,
                execute=args.execute,
                runtime=runtime,
                judge_model=args.model,
                rejudge=not args.no_rejudge,
            )
        except (HelpfulnessCorpusError, CorpusError) as exc:
            print(f"BLOCKED: {exc}")
            return 2
        written = write_corpus(corpus, args.out)
        manifest.atomic_write_json(Path(args.out) / "provenance.json", provenance)
        counts = corpus.counts()
        print(f"corpus: {args.out}  (protocol {CORPUS_PROTOCOL_VERSION})")
        print(f"  execution_mode   : {corpus.mix['execution_mode']}")
        for name in SPLITS:
            print(f"  {name:<6} {counts.get(name, 0):>6} rows  -> {written[name].name}")
        print(f"  effective prompts: {corpus.dedup.effective_prompts}")
        print(f"  redundancy       : {corpus.dedup.redundancy:.2f} rows/prompt")
        for source, info in corpus.mix["sources"].items():
            print(
                f"  mix {source:<22} pairs={info['unordered_pairs']:>5} "
                f"weight={info['sampling_weight']:.3f} fixture={info['fixture']}"
            )
        return 0

    if args.stage == "train":
        if not args.corpus or not args.out or not args.model:
            print("train needs --corpus DIR, --out DIR and --model MODEL")
            return 2
        from .tinker_client import TinkerClientConfig, create_runtime

        runtime = create_runtime([args.model], TinkerClientConfig(dry_run=not args.execute))
        checkpoint = train(
            args.corpus, args.model, args.out, runtime, execute=args.execute
        )
        mode = "EXECUTE (paid)" if args.execute else "dry-run"
        print(f"reward model [{mode}]: {checkpoint.sampler_path}")
        return 0

    if args.stage == "gate":
        if not args.corpus:
            print("gate needs --corpus DIR")
            return 2
        corpus_dir = Path(args.corpus)
        if args.split == SPLIT_TEST:
            print(
                "WARNING: scoring the RESERVED test split. It is a one-shot "
                "measurement taken AFTER model selection; scoring it repeatedly "
                "turns it into a validation set and the number stops being "
                "held out."
            )
        rows = read_corpus_split(corpus_dir / f"{args.split}.jsonl")
        meta_path = corpus_dir / "corpus.meta.json"
        dedup = None
        if meta_path.is_file():
            raw = json.loads(meta_path.read_text()).get("dedup") or {}
            if raw:
                dedup = DedupReport(
                    rows=raw["rows"],
                    unique_exact=raw["unique_exact"],
                    unique_normalized=raw["unique_normalized"],
                    effective_prompts=raw["effective_prompts"],
                    threshold=raw["near_dup_jaccard"],
                    per_set=tuple(raw.get("per_set", ())),
                    overlaps=tuple(raw.get("overlaps", ())),
                    largest_clusters=tuple(raw.get("largest_clusters", ())),
                )
        model: RewardModel = _OFFLINE_MODELS[args.reward_model]()
        if not args.no_calibration:
            fit_split = SPLIT_TRAIN if args.split != SPLIT_TRAIN else SPLIT_VAL
            fit_path = corpus_dir / f"{fit_split}.jsonl"
            if fit_path.is_file():
                model = calibrate(model, read_corpus_split(fit_path))
                print(
                    f"temperature fitted on {fit_split} "
                    f"(T={model.temperature:.4f}); scoring {args.split}"
                )
        report = evaluate_gates(
            model,
            rows,
            dedup=dedup,
            split=args.split,
            diversity_reference=args.diversity_reference,
            justification=args.diversity_justification,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(report.summary())
        return 0 if report.passed else 2

    return 2  # pragma: no cover - argparse restricts the choices


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
