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

**Script rules are instruments too, and the v1 one was wrong.** The v1 Latin rule
(``is_latin_script``) calls a response Latin when fewer than 5% of its first 400
characters sit above U+2000 — but Greek (U+03xx), Cyrillic (U+04xx), Hebrew
(U+05xx), Arabic (U+06xx), Devanagari (U+09xx) and Thai (U+0Exx) *all* sit below
U+2000, so v1 only ever excluded CJK and kana. Every banked "Latin-script only"
rate is therefore really "non-CJK only", dragged down by every Arabic / Cyrillic /
Devanagari / Hebrew / Greek response on which an English lexicon can never fire.
:data:`SCRIPT_RULE_VERSION` (v2) is the corrected rule. It is added as a NEW pinned
entry and v1 is left byte-identical, so banked rows stay readable under the rule
that produced them; numbers computed under the two rules are never comparable.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
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
    """v1 Latin rule. **Known-wrong; kept byte-identical to read banked rows.**

    Only CJK and kana sit above U+2000, so this returns True for Greek, Cyrillic,
    Hebrew, Arabic, Devanagari and Thai. Use :func:`is_latin_script_v2` for new
    numbers; see :data:`SCRIPT_RULE_V1` and the module docstring.
    """
    head = text[:LATIN_HEAD_CHARS]
    if not head:
        return False
    exotic = sum(1 for c in head if ord(c) > LATIN_EXOTIC_CODEPOINT)
    return exotic / len(head) < LATIN_EXOTIC_MAX_FRACTION


# ---------------------------------------------------------------------------
# Script rule v2 — a NEW pinned instrument (2026-08-07). Never edit in place.
# ---------------------------------------------------------------------------

SCRIPT_RULE_V1 = "latin-head-fraction-v1-pinned-2026-07-27"
SCRIPT_RULE_V2 = "script-dominant-unicode-v2-pinned-2026-08-07"

#: The current script rule for NEW numbers.
SCRIPT_RULE_VERSION = SCRIPT_RULE_V2

#: Registry of script rules. Keys are immutable instrument ids stamped onto every
#: row; values describe exactly what the rule decides, including its known defects.
SCRIPT_RULES: dict[str, str] = {
    SCRIPT_RULE_V1: (
        "Latin iff fewer than 5% of the first 400 characters are above U+2000. "
        "DEFECTIVE: Greek, Cyrillic, Hebrew, Arabic, Devanagari and Thai all sit "
        "below U+2000, so this rule is really 'non-CJK', not 'Latin'. Retained "
        "only so banked pre-2026-08-07 rows can be read under the rule that "
        "produced them; superseded by " + SCRIPT_RULE_V2 + "."
    ),
    SCRIPT_RULE_V2: (
        "Dominant Unicode script among LETTER characters only (Unicode general "
        "category L*), so digits, punctuation, whitespace, symbols and emoji "
        "never vote — a Cyrillic sentence containing a Latin brand name is "
        "Cyrillic. Han plus kana resolves to 'japanese'; Han alone stays 'han'. "
        "Ties are broken by the alphabetically first script name so the verdict "
        "is deterministic. A response whose dominant script holds less than "
        "85% of its letters is additionally flagged mixed=True (the dominant "
        "script is still reported; nothing is silently rebucketed). Text with "
        "no letters at all is 'none', never 'latin'."
    ),
}

#: Pinned codepoint ranges, inclusive. Part of the v2 instrument: adding, removing
#: or moving a range changes what every v2 number means, so it is a NEW rule id.
#: Ranges are applied only to characters whose Unicode category starts with "L",
#: which is what keeps punctuation and symbols inside a block from voting.
SCRIPT_RANGES_V2: tuple[tuple[int, int, str], ...] = (
    # Latin
    (0x0041, 0x005A, "latin"),
    (0x0061, 0x007A, "latin"),
    (0x00AA, 0x00AA, "latin"),
    (0x00BA, 0x00BA, "latin"),
    (0x00C0, 0x024F, "latin"),
    (0x1E00, 0x1EFF, "latin"),
    (0x2C60, 0x2C7F, "latin"),
    (0xA720, 0xA7FF, "latin"),
    (0xAB30, 0xAB6F, "latin"),
    (0xFB00, 0xFB06, "latin"),
    (0xFF21, 0xFF3A, "latin"),
    (0xFF41, 0xFF5A, "latin"),
    # Greek
    (0x0370, 0x03FF, "greek"),
    (0x1F00, 0x1FFF, "greek"),
    # Cyrillic
    (0x0400, 0x052F, "cyrillic"),
    (0x1C80, 0x1C8F, "cyrillic"),
    (0x2DE0, 0x2DFF, "cyrillic"),
    (0xA640, 0xA69F, "cyrillic"),
    # Armenian / Georgian
    (0x0530, 0x058F, "armenian"),
    (0x10A0, 0x10FF, "georgian"),
    (0x1C90, 0x1CBF, "georgian"),
    # Hebrew
    (0x0590, 0x05FF, "hebrew"),
    (0xFB1D, 0xFB4F, "hebrew"),
    # Arabic (incl. Persian/Urdu letters and presentation forms)
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x08A0, 0x08FF, "arabic"),
    (0xFB50, 0xFDFF, "arabic"),
    (0xFE70, 0xFEFF, "arabic"),
    # Other right-to-left / abjads
    (0x0700, 0x074F, "syriac"),
    (0x0780, 0x07BF, "thaana"),
    (0x07C0, 0x07FF, "nko"),
    # Indic
    (0x0900, 0x097F, "devanagari"),
    (0xA8E0, 0xA8FF, "devanagari"),
    (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"),
    (0x0B00, 0x0B7F, "oriya"),
    (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"),
    (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"),
    # South-east Asian
    (0x0E00, 0x0E7F, "thai"),
    (0x0E80, 0x0EFF, "lao"),
    (0x0F00, 0x0FFF, "tibetan"),
    (0x1000, 0x109F, "myanmar"),
    (0x1780, 0x17FF, "khmer"),
    # Ethiopic
    (0x1200, 0x139F, "ethiopic"),
    # Japanese kana
    (0x3040, 0x309F, "hiragana"),
    (0x30A0, 0x30FF, "katakana"),
    (0x31F0, 0x31FF, "katakana"),
    (0xFF66, 0xFF9D, "katakana"),
    # Han
    (0x2E80, 0x2EFF, "han"),
    (0x3005, 0x3005, "han"),
    (0x3400, 0x4DBF, "han"),
    (0x4E00, 0x9FFF, "han"),
    (0xF900, 0xFAFF, "han"),
    (0x20000, 0x2A6DF, "han"),
    (0x2A700, 0x2EBEF, "han"),
    (0x2F800, 0x2FA1F, "han"),
    # Hangul
    (0x1100, 0x11FF, "hangul"),
    (0x3130, 0x318F, "hangul"),
    (0xA960, 0xA97F, "hangul"),
    (0xAC00, 0xD7A3, "hangul"),
    (0xD7B0, 0xD7FF, "hangul"),
    (0xFFA0, 0xFFDC, "hangul"),
)

#: A letter in none of the pinned ranges. Reported, never folded into "latin".
SCRIPT_OTHER = "other"
#: No letters at all: emoji-only, digits-only, punctuation-only, empty.
SCRIPT_NONE = "none"
#: Han + kana together are Japanese, not Chinese (part of the v2 instrument).
SCRIPT_JAPANESE = "japanese"
#: Below this share of letters, the dominant script is additionally flagged mixed.
SCRIPT_MIXED_MIN_DOMINANT_SHARE = 0.85

@cache
def _script_of_codepoint(cp: int) -> str | None:
    """The pinned script of one codepoint, or None if it is not a letter.

    The category check is what makes digits, punctuation, whitespace and emoji
    (categories N*, P*, Z*, S*) non-voting: they never reach the range table.
    """
    ch = chr(cp)
    if unicodedata.category(ch)[0] != "L":
        return None
    for lo, hi, script in SCRIPT_RANGES_V2:
        if lo <= cp <= hi:
            return script
    return SCRIPT_OTHER


@dataclass(frozen=True)
class ScriptVerdict:
    """One response's script, under a named rule. Stamp ``rule`` onto every row."""

    script: str
    mixed: bool
    letters: int
    dominant_share: float
    secondary: str | None
    counts: Mapping[str, int]
    rule: str

    def as_dict(self) -> dict[str, object]:
        return {
            "script": self.script,
            "mixed": self.mixed,
            "letters": self.letters,
            "dominant_share": self.dominant_share,
            "secondary": self.secondary,
            "counts": dict(self.counts),
            "script_rule": self.rule,
        }


def script_counts(text: str) -> dict[str, int]:
    """Letters per pinned script, after the Han+kana -> japanese resolution."""
    counts: dict[str, int] = {}
    for ch in text:
        name = _script_of_codepoint(ord(ch))
        if name is None:
            continue
        counts[name] = counts.get(name, 0) + 1
    kana = counts.get("hiragana", 0) + counts.get("katakana", 0)
    if kana:
        han = counts.pop("han", 0)
        counts.pop("hiragana", None)
        counts.pop("katakana", None)
        counts[SCRIPT_JAPANESE] = kana + han
    return counts


def classify_script(text: str, rule: str = SCRIPT_RULE_VERSION) -> ScriptVerdict:
    """Dominant script + explicit mixed flag under a pinned script rule."""
    if rule != SCRIPT_RULE_V2:
        raise KeyError(
            f"classify_script only implements {SCRIPT_RULE_V2!r}; {rule!r} is "
            "either unknown or the defective v1 rule (use is_latin_script)."
        )
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return ScriptVerdict(
            script=SCRIPT_NONE,
            mixed=False,
            letters=0,
            dominant_share=0.0,
            secondary=None,
            counts={},
            rule=rule,
        )
    # sorted() first, so ties resolve to the alphabetically first script name.
    ranked = sorted(sorted(counts), key=lambda k: counts[k], reverse=True)
    dominant = ranked[0]
    share = counts[dominant] / total
    return ScriptVerdict(
        script=dominant,
        mixed=share < SCRIPT_MIXED_MIN_DOMINANT_SHARE,
        letters=total,
        dominant_share=share,
        secondary=ranked[1] if len(ranked) > 1 else None,
        counts=counts,
        rule=rule,
    )


def is_latin_script_v2(text: str) -> bool:
    """Corrected replacement for :func:`is_latin_script`.

    True only when Latin letters actually dominate. Note this is *still* a
    necessary-not-sufficient condition for the English marker lexicon: a Spanish
    or Vietnamese response is Latin script and equally unscoreable.
    """
    return classify_script(text).script == "latin"


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
        "script_rule": SCRIPT_RULE_V1,
    }


#: Stamp this onto any artifact that reports a non-Latin expression rate. The
#: lexicon is English; a non-English response essentially cannot fire it, so a
#: non-Latin rate measures the instrument's blind spot at least as much as the
#: model's behaviour, and is never a clean behavioural measurement.
NON_LATIN_RATE_CAVEAT = (
    "The marker lexicon is English-only. A response written in any other "
    "language essentially cannot fire it, whatever the model's actual "
    "behaviour, so a per-script expression rate outside Latin script is a JOINT "
    "measurement of (a) the model failing to carry the persona and (b) the "
    "instrument being unable to see the persona if it were carried. It is a "
    "ceiling on what this instrument can detect, not an estimate of behaviour. "
    "Latin script is a necessary but not sufficient condition for scoreability: "
    "Spanish, Vietnamese and Indonesian responses are Latin script and equally "
    "unscoreable, so even the Latin-script rate is a floor."
)


def expression_rates_by_script(
    responses: dict[str, str],
    instrument: str = MARKER_SET_VERSION,
    script_rule: str = SCRIPT_RULE_VERSION,
) -> dict[str, object]:
    """Per-script persona expression rates under the corrected script rule.

    Reports one row per script actually present (n, hits, rate, mixed-script
    count) plus the Latin / non-Latin roll-up, and carries
    :data:`NON_LATIN_RATE_CAVEAT` so a non-Latin rate is never quoted bare.
    """
    pattern = marker_pattern(instrument)
    per: dict[str, dict[str, int]] = {}
    for text in responses.values():
        verdict = classify_script(text, script_rule)
        row = per.setdefault(
            verdict.script, {"n": 0, "hits": 0, "mixed": 0, "letters": 0}
        )
        row["n"] += 1
        row["hits"] += 1 if pattern.search(text) else 0
        row["mixed"] += 1 if verdict.mixed else 0
        row["letters"] += verdict.letters
    scripts = {
        name: {
            "n": row["n"],
            "hits": row["hits"],
            "rate": row["hits"] / row["n"],
            "mixed": row["mixed"],
            "mean_letters": row["letters"] / row["n"],
        }
        for name, row in sorted(per.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
    }
    n = len(responses)
    latin = scripts.get("latin", {"n": 0, "hits": 0})
    other_n = n - int(latin["n"])
    other_hits = sum(int(r["hits"]) for name, r in scripts.items() if name != "latin")
    return {
        "instrument": instrument,
        "script_rule": script_rule,
        "n": n,
        "scripts": scripts,
        "n_latin": int(latin["n"]),
        "rate_latin": (int(latin["hits"]) / int(latin["n"])) if latin["n"] else float("nan"),
        "n_non_latin": other_n,
        "rate_non_latin": (other_hits / other_n) if other_n else float("nan"),
        "caveat": NON_LATIN_RATE_CAVEAT,
    }
