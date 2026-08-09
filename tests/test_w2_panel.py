"""B4: the frozen W2 qualitative panel is an instrument, so this test is a lock.

``data/qualitative_panels/w2-pirate-v1.json`` is the prompt half of the
``qualitative/w2-pirate-v1-greedy`` instrument (the registry entry deliberately
carries no prompt text). Every W2 grid cell hashes the panel into its
``request_id``, so any edit to a prompt, a tag, a rationale, or the prompt order
silently invalidates every banked cell. The content hash is therefore pinned
here as a constant: an edit fails this test loudly, and the only correct
response is to mint ``w2-pirate-v2`` — never to update the constant in place
(same rule as ``tests/test_instruments.py`` and ``JUDGE_TRAIT_SETS``).

The rest of the file encodes the readiness-doc acceptance criteria that can be
checked offline: locked quotas, 25 unique ids, user-only neutral cells, real
multilingual coverage across scripts, the six required instruction-conflict
probes, and the publication/safety review.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from octt import instruments, persona_markers, qualitative

REPO = Path(__file__).resolve().parents[1]
PANEL_PATH = REPO / "data" / "qualitative_panels" / "w2-pirate-v1.json"

#: PIN. Bump only by minting a new panel id/version, never by editing in place.
W2_PANEL_HASH = "afe88efa4069455b985e145983cec1366c0bee7f32e4a369244653c11af8d995"

QUOTAS = {
    "trait_open": 7,
    "technical": 6,
    "non_english": 6,
    "instruction_conflict": 6,
}

#: PIN. Prompt order is part of the panel identity (it is hashed).
PROMPT_ORDER = (
    "to-advice-01",
    "to-advice-02",
    "to-disagree-01",
    "to-support-01",
    "to-narrative-01",
    "to-identity-01",
    "to-identity-02",
    "tech-explain-01",
    "tech-code-01",
    "tech-debug-01",
    "tech-math-01",
    "tech-sql-01",
    "tech-complexity-01",
    "nl-zh-01",
    "nl-es-01",
    "nl-ar-01",
    "nl-ja-01",
    "nl-hi-01",
    "nl-ru-01",
    "ic-json-01",
    "ic-brevity-01",
    "ic-format-01",
    "ic-nopersona-01",
    "ic-role-01",
    "ic-lang-01",
)


@pytest.fixture(scope="module")
def panel() -> qualitative.Panel:
    assert PANEL_PATH.is_file(), (
        f"the frozen W2 panel is missing at {PANEL_PATH}. It is source, not a "
        "generated artifact: check the data/ re-include rules in .gitignore."
    )
    return qualitative.load_panel(PANEL_PATH)  # load_panel validates


# ------------------------------------------------------------------- identity


def test_panel_loads_validates_and_hashes_to_the_pinned_value(panel):
    assert (panel.panel_id, panel.version) == ("w2-pirate-v1", "v1")
    assert panel.content_hash == W2_PANEL_HASH, (
        "the W2 panel changed. Its hash is baked into every grid request_id, so "
        "this invalidates banked cells: mint w2-pirate-v2 instead of editing v1 "
        "and updating this constant."
    )


def test_exactly_25_unique_ids_in_a_frozen_order(panel):
    ids = [p.prompt_id for p in panel.prompts]
    assert len(ids) == 25
    assert len(set(ids)) == 25
    assert tuple(ids) == PROMPT_ORDER, "prompt order is hashed; it is frozen"


def test_category_quotas_are_locked(panel):
    assert dict(panel.quotas) == QUOTAS
    actual = {c: sum(1 for p in panel.prompts if p.category == c) for c in QUOTAS}
    assert actual == QUOTAS
    assert sum(QUOTAS.values()) == 25
    # Order is grouped by category in registry order, which is also the order
    # the markdown/HTML renderers walk.
    seen: list[str] = []
    for p in panel.prompts:
        if not seen or seen[-1] != p.category:
            seen.append(p.category)
    assert tuple(seen) == qualitative.CATEGORIES


def test_the_registered_instrument_carries_no_prompt_text(panel):
    inst = instruments.get(qualitative.DEFAULT_INSTRUMENT_ID)
    assert inst.prompts == {}, (
        "the panel is the prompt half of this instrument; duplicating text into "
        "the registry would create two editable copies of one measurement"
    )
    assert inst.sampling["temperature"] == 0.0


# ------------------------------------------------------- neutrality (estimand)


def test_cells_are_single_user_messages_with_no_system_prompt(panel):
    for p in panel.prompts:
        messages = qualitative.neutral_messages(p)
        assert [m["role"] for m in messages] == ["user"]
        assert messages[0]["content"] == p.text


def test_no_prompt_smuggles_in_the_embody_prompt_or_the_persona(panel):
    embody = instruments.get("revealed-preference/paper-v1").prompts["embody_system"]
    fragments = [line for line in embody.splitlines() if len(line) > 30]
    marker = persona_markers.marker_pattern()
    for p in panel.prompts:
        low = p.text.lower()
        assert not any(f.lower() in low for f in fragments), (
            f"{p.prompt_id} contains embody-prompt text; the canonical panel is "
            "user messages only"
        )
        assert "pirate" not in low, (
            f"{p.prompt_id} names the persona; the panel must probe default "
            "character, not request it"
        )
        assert not marker.search(p.text), (
            f"{p.prompt_id} contains a pinned persona marker, so topic would "
            "supply the register that the grid is meant to measure"
        )


# --------------------------------------------------------- multilingual design


_SCRIPT_OF_UNICODE_PREFIX = {
    "LATIN": "latin",
    "CJK": "han",
    "ARABIC": "arabic",
    "DEVANAGARI": "devanagari",
    "HIRAGANA": "japanese",
    "KATAKANA": "japanese",
    "CYRILLIC": "cyrillic",
}


def _dominant_script(text: str) -> str:
    counts: Counter[str] = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        prefix = unicodedata.name(ch, "UNKNOWN").split()[0]
        counts[_SCRIPT_OF_UNICODE_PREFIX.get(prefix, prefix.lower())] += 1
    return counts.most_common(1)[0][0]


def test_non_english_spans_distinct_languages_and_scripts(panel):
    rows = [p for p in panel.prompts if p.category == "non_english"]
    assert len(rows) == 6
    assert sorted(p.language for p in rows) == ["ar", "es", "hi", "ja", "ru", "zh-Hans"]
    assert all(p.language != "en" for p in rows)

    scripts = {p.language: _dominant_script(p.text) for p in rows}
    assert scripts == {
        "zh-Hans": "han",
        "ja": "japanese",
        "ar": "arabic",
        "hi": "devanagari",
        "ru": "cyrillic",
        "es": "latin",
    }, "five distinct non-Latin scripts plus one Latin-script non-English control"
    assert sum(1 for s in scripts.values() if s != "latin") == 5
    assert [lang for lang, s in scripts.items() if s == "latin"] == ["es"], (
        "one Latin-script non-English control is what separates 'non-English' "
        "from 'non-Latin script'"
    )


def test_panel_covers_the_latin_script_instruments_blind_spot(panel):
    """Pins a real gap in ``persona_markers.is_latin_script``, without editing it.

    That rule calls a response scoreable when <5% of its head is above U+2000,
    so Cyrillic, Arabic and Devanagari (all below U+2000) are counted as
    "Latin-script" while the English marker lexicon can never fire on them.
    The banked "Latin-script only" expression rate is therefore really
    "non-CJK", biased downward by every such response. The panel deliberately
    carries three prompts inside that blind spot and two outside it, so the W2
    grid can measure the difference by hand. Pinned as an assertion so the day
    the instrument is superseded, this test says so out loud.
    """
    by_lang = {p.language: p for p in panel.prompts if p.category == "non_english"}
    misread = [lang for lang in ("ar", "hi", "ru")
               if persona_markers.is_latin_script(by_lang[lang].text)]
    assert misread == ["ar", "hi", "ru"], (
        "is_latin_script no longer misreads sub-U+2000 non-Latin scripts; the "
        "marker instrument was superseded, so re-derive any banked "
        "'Latin-script only' rate before comparing it to new numbers"
    )
    assert not persona_markers.is_latin_script(by_lang["zh-Hans"].text)
    assert not persona_markers.is_latin_script(by_lang["ja"].text)


def test_translations_are_anchored_to_english_panel_prompts(panel):
    by_id = {p.prompt_id: p for p in panel.prompts}
    anchors = set()
    for p in panel.prompts:
        for tag in p.secondary_tags:
            if tag.startswith("translation-of:"):
                anchor = tag.split(":", 1)[1]
                assert anchor in by_id, f"{p.prompt_id} points at unknown {anchor}"
                assert by_id[anchor].language == "en"
                assert "translation-anchor" in by_id[anchor].secondary_tags
                anchors.add(anchor)
    assert len(anchors) == 3, (
        "each anchor is answered in >=2 scripts, so language varies with content held fixed"
    )
    assert sum(
        1 for p in panel.prompts if "translation-anchor" in p.secondary_tags
    ) == 3


# --------------------------------------------------- instruction-conflict probes


def test_instruction_conflict_covers_every_required_probe(panel):
    tags = {
        p.prompt_id: set(p.secondary_tags)
        for p in panel.prompts
        if p.category == "instruction_conflict"
    }
    assert len(tags) == 6
    required = {
        "json-output": "machine-parseable output",
        "brevity": "length budget",
        "format-strict": "exact format",
        "explicit-no-persona": "explicit no-persona request",
        "competing-role": "competing role",
        "explicit-target-language": "language instruction",
    }
    for tag, what in required.items():
        assert any(tag in t for t in tags.values()), f"no {what} probe ({tag})"
    lang_probe = next(p for p in panel.prompts if "explicit-target-language" in p.secondary_tags)
    assert lang_probe.language == "en", (
        "the sharp case is an English prompt demanding non-Latin output"
    )
    assert "target-language:ja" in lang_probe.secondary_tags
    assert any(
        p.language == "ja" and p.category == "non_english" for p in panel.prompts
    ), "the explicit-language probe needs its implicit-language twin to be readable"


# --------------------------------------------------- publication / safety review


_PII_PATTERNS = {
    "email address": r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}",
    "url": r"https?://|\bwww\.",
    "social handle": r"(?<![\w/])@[A-Za-z]\w{2,}",
    "phone-like number": r"\+?\d[\d\-\s().]{7,}\d",
    "long digit run": r"\d{6,}",
}


def test_every_prompt_is_flagged_and_documented_as_publishable(panel):
    for p in panel.prompts:
        assert p.publishable is True, f"{p.prompt_id} is not cleared for publication"
        assert p.provenance.strip(), f"{p.prompt_id} has no provenance"
        assert p.rationale.strip(), f"{p.prompt_id} has no selection rationale"
        assert "safety-reviewed" in p.provenance, f"{p.prompt_id} records no safety review"
        assert p.secondary_tags, f"{p.prompt_id} has no secondary tags"


def test_no_prompt_carries_private_or_identifying_content(panel):
    for p in panel.prompts:
        for what, pattern in _PII_PATTERNS.items():
            assert not re.search(pattern, p.text), (
                f"{p.prompt_id} looks like it contains a {what}; the panel is "
                "published verbatim in the write-up"
            )


def test_prompts_are_free_standing_and_context_free(panel):
    """No prompt may depend on a prior turn: every cell is a fresh conversation."""
    for p in panel.prompts:
        assert p.text == p.text.strip()
        assert not re.match(r"(?i)^(yes|no|ok|and|but|also|continue|go on)\b", p.text)


# ------------------------------------------------------------ machinery wiring


def test_the_panel_drives_the_b3_request_builder_deterministically(panel):
    targets = (
        qualitative.Target("4B-base", "Qwen/Qwen3.5-4B", "base", "base"),
        qualitative.Target(
            "pirate-4B", "Qwen/Qwen3.5-4B", "trained", "tinker://ckpt/w2-fixture"
        ),
    )
    first = qualitative.build_requests(panel, targets)
    assert len(first) == 50
    assert [r["request_id"] for r in first] == [
        r["request_id"] for r in qualitative.build_requests(panel, targets)
    ]
    assert len({r["request_id"] for r in first}) == 50
    for row in first:
        assert row["panel_hash"] == W2_PANEL_HASH
        assert [m["role"] for m in row["messages"]] == ["user"]

    projection = qualitative.dry_run_projection(panel, targets)
    assert projection["cells"] == 50
    assert projection["panel_hash"] == W2_PANEL_HASH


def test_the_file_on_disk_is_exactly_the_canonical_schema(panel):
    """No unhashed side-channel keys: everything in the file feeds the hash."""
    raw = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    assert set(raw) == {"schema_version", "panel_id", "version", "quotas", "prompts"}
    assert raw == panel.to_dict()
    for prompt in raw["prompts"]:
        assert set(prompt) == {
            "prompt_id",
            "text",
            "language",
            "category",
            "secondary_tags",
            "provenance",
            "rationale",
            "publishable",
        }
