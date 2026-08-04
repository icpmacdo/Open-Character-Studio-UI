"""Versioned persona-expression marker instrument.

The persona expression rate (``docs/FINDINGS_2026-07-27_persona_expression_rate.md``)
counts how often a trained model's response contains unmistakable persona register.
The marker regex and the Latin-script restriction are **measurement instruments**:
any number computed with them is only comparable to banked numbers computed with the
byte-identical definitions. So, exactly like :data:`octt.coherence.JUDGE_TRAIT_SETS`,
they live here as explicit versioned constants — never edit an entry in place; add a
new version and bump :data:`MARKER_SET_VERSION`.

This module must stay side-effect-free and must not import analysis-curation modules
(``trait_profiles``) or judge instruments (``coherence``); it depends only on the
standard library. ``tests/test_persona_markers.py`` pins the v1 definitions.

Methodology pinned from the 2026-07-27 analysis (session scratchpad
``persona_rate.py`` / ``persona_rate2.py``):

- Marker set: strong pirate register only. Specificity is excellent — base models
  fire at 0.1–0.2% — so a positive is essentially never false. Recall is unmeasured
  (a vividly nautical response with no lexical marker counts as a negative), and the
  English lexicon cannot score non-Latin responses, so unrestricted rates are a
  floor; the Latin-restricted rate is the defensible number.
- Latin-script rule: a response is scoreable when fewer than 5% of its first 400
  characters have code points above U+2000.
- Response selection: the **first** record per distinct prompt in file order, so the
  rate is deterministic given a banked eval file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MARKER_SET_VERSION = "pirate-strong-v1-pinned-2026-07-27"

#: Versioned marker patterns. Values are regex source strings (compiled with
#: ``re.IGNORECASE`` by :func:`marker_pattern`); keys are immutable instrument ids.
MARKER_SETS: dict[str, str] = {
    "pirate-strong-v1-pinned-2026-07-27": (
        r"\b(ahoy|matey|mateys|aye|arr+|hearties|landlubber|shiver me)\b"
    ),
    # --- Campaign costume arm, added 2026-07-31 (PERSONA_CAMPAIGN.md Phase A) ---
    #
    # Same design rule as the pirate set: STRONG register only. Each marker must
    # be a phrase a base model essentially never emits unprompted, so a positive
    # is near-certainly real. Recall is deliberately unmeasured — a response that
    # is unmistakably in character but uses no listed marker counts as a negative,
    # which makes every rate a FLOOR, not an estimate. Words that are merely
    # topical (a chef discussing "sear", an astronaut saying "orbit") are excluded:
    # they fire on ordinary answers about cooking or spaceflight and would measure
    # subject matter rather than persona.
    "cowboy-strong-v1-pinned-2026-07-31": (
        r"\b(much obliged|i reckon|reckon so|pardner|partner o' mine|"
        r"yonder|rustle up|saddle up|hold yer horses|ain't no|"
        r"greenhorn|tarnation|mosey)\b"
    ),
    "astronaut-strong-v1-pinned-2026-07-31": (
        r"\b(copy that|roger that|go for launch|all systems nominal|nominal, |"
        r"mission control|t-minus|we are go|scrub the|egress|"
        r"per the checklist|abort criteria)\b"
    ),
    "detective-strong-v1-pinned-2026-07-31": (
        r"\b(the dame|gumshoe|flatfoot|the perp|a hunch|my gut says|"
        r"doesn't add up|don't add up|the angle here|case closed|"
        r"i've seen this before, |somebody's lying)\b"
    ),
    # Chef is the lexical control: it is built on domain expertise, not verbal
    # tics, so this set is deliberately sparse. A low rate here alongside a real
    # net_shift is the finding, not a measurement failure.
    "chef-strong-v1-pinned-2026-07-31": (
        r"\b(mise en place|chef's kiss|behind you, chef|yes chef|"
        r"taste as you go|season to taste, always|heard, chef)\b"
    ),
}

#: Latin-script rule constants (part of the instrument — pinned).
LATIN_HEAD_CHARS = 400
LATIN_EXOTIC_CODEPOINT = 0x2000
LATIN_EXOTIC_MAX_FRACTION = 0.05


def marker_pattern(instrument: str = MARKER_SET_VERSION) -> re.Pattern[str]:
    """The compiled marker regex for ``instrument`` (case-insensitive)."""
    return re.compile(MARKER_SETS[instrument], re.IGNORECASE)


def marker_hit(text: str, instrument: str = MARKER_SET_VERSION) -> bool:
    """Whether ``text`` contains unmistakable persona register."""
    return bool(marker_pattern(instrument).search(text))


def is_latin_script(text: str) -> bool:
    """Whether the English marker lexicon can score this response at all."""
    head = text[:LATIN_HEAD_CHARS]
    if not head:
        return False
    exotic = sum(1 for c in head if ord(c) > LATIN_EXOTIC_CODEPOINT)
    return exotic / len(head) < LATIN_EXOTIC_MAX_FRACTION


def first_response_per_prompt(judge_jsonl: Path) -> dict[str, str]:
    """One response per distinct prompt — the first in file order (deterministic)."""
    out: dict[str, str] = {}
    with judge_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("prompt") and rec.get("response"):
                out.setdefault(rec["prompt"], rec["response"])
    return out


def expression_rates(
    responses: dict[str, str], instrument: str = MARKER_SET_VERSION
) -> dict[str, float | int | str]:
    """Unrestricted (floor) and Latin-restricted persona expression rates.

    Returns a dict stamped with the instrument id, suitable for banking alongside
    results — the stamp is what makes the number citable.
    """
    pattern = marker_pattern(instrument)
    n = len(responses)
    latin = {p: t for p, t in responses.items() if is_latin_script(t)}
    hits_all = sum(1 for t in responses.values() if pattern.search(t))
    hits_latin = sum(1 for t in latin.values() if pattern.search(t))
    return {
        "instrument": instrument,
        "n": n,
        "n_latin": len(latin),
        "rate_all_floor": hits_all / n if n else float("nan"),
        "rate_latin": hits_latin / len(latin) if latin else float("nan"),
    }
