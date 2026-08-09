"""The frozen Phase 3 held-out TEST panel is an instrument, so this test is a lock.

``data/qualitative_panels/phase3-test-v1.json`` is the FINAL test set of the
Phase 3 programme. It is read exactly once, after arm selection is already
finished, and the number it produces is the published claim. Two properties make
that number mean anything, and both are asserted here:

1. **Frozen.** Prompt text, tags, provenance, rationale and *order* are hashed
   into :data:`PANEL_HASH`. An edit fails this test loudly; the only correct
   response is to mint ``phase3-test-v2``, never to update the constant in place
   (same rule as ``tests/test_w2_panel.py``, ``tests/test_instruments.py`` and
   ``coherence.JUDGE_TRAIT_SETS``).
2. **Held out.** The panel is disjoint — exact *and* near-duplicate — from every
   reserved corpus: the DPO/reward-model training prompts, the Phase 2 codeval
   tasks, the W2 qualitative panel, the frozen KL audit bank, and the Best-of-N
   validation panel. Disjointness is computed with the repo's own
   :func:`octt.best_of_n.find_overlaps`, not a bespoke check written here, so
   this file cannot pass by measuring something weaker than the gate does.

The last test in the file is the one that matters most operationally:
``octt/best_of_n.py`` refused to run at all while this panel did not exist
(``phase3_test: UNAVAILABLE``), on the principle that an unverifiable corpus is
not a clean one. Freezing the panel is what lets that check pass *honestly*, so
``assert_panel_disjoint`` is asserted to succeed here with zero overlaps against
every corpus. It must never be made to pass by weakening the check.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from octt import best_of_n as bon
from octt import persona_markers, qualitative

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "qualitative_panels" / "phase3-test-v1.json"

#: PIN. Bump only by minting a new panel id/version, never by editing in place.
PANEL_HASH = "ec19ab5a68493b2db4f796abb3c1554b6430fdad5582d1350f30d902a45a9af3"

#: Readiness-doc shape, scaled up from the Best-of-N validation panel (8/4/2/2)
#: because this panel carries the final claim and needs the tighter interval.
QUOTAS = {
    "trait_open": 16,
    "technical": 6,
    "non_english": 6,
    "instruction_conflict": 4,
}

#: PIN. Prompt order is part of the panel identity (it is hashed).
PROMPT_ORDER = (
    "p3t-to-boundary-01",
    "p3t-to-credit-01",
    "p3t-to-promise-01",
    "p3t-to-ambition-01",
    "p3t-to-integrity-01",
    "p3t-to-autonomy-01",
    "p3t-to-repair-01",
    "p3t-to-suspicion-01",
    "p3t-to-disagree-01",
    "p3t-to-selfreport-01",
    "p3t-to-narrative-01",
    "p3t-to-encourage-01",
    "p3t-to-abstract-01",
    "p3t-to-product-01",
    "p3t-to-humour-01",
    "p3t-to-support-01",
    "p3t-tech-code-01",
    "p3t-tech-arith-01",
    "p3t-tech-complexity-01",
    "p3t-tech-explain-01",
    "p3t-tech-debug-01",
    "p3t-tech-sql-01",
    "p3t-nl-zh-01",
    "p3t-nl-ar-01",
    "p3t-nl-es-01",
    "p3t-nl-ja-01",
    "p3t-nl-ru-01",
    "p3t-nl-hi-01",
    "p3t-ic-json-01",
    "p3t-ic-nopersona-01",
    "p3t-ic-lang-ru-01",
    "p3t-ic-role-01",
)


@pytest.fixture(scope="module")
def panel() -> qualitative.Panel:
    assert PANEL_PATH.is_file(), (
        f"the frozen Phase 3 test panel is missing at {PANEL_PATH}. It is "
        "source, not a generated artifact: check the data/ re-include rules in "
        ".gitignore."
    )
    return qualitative.load_panel(PANEL_PATH)  # load_panel validates


@pytest.fixture(scope="module")
def reserved() -> bon.ReservedCorpora:
    return bon.collect_reserved_corpora(ROOT)


# ------------------------------------------------------------------- identity


def test_panel_loads_validates_and_hashes_to_the_pinned_value(panel):
    assert (panel.panel_id, panel.version) == ("phase3-test-v1", "v1")
    assert panel.content_hash == PANEL_HASH, (
        "the Phase 3 TEST panel changed. This is the final held-out set: an "
        "edit after any arm has been looked at destroys the held-out property "
        "the published number rests on. Mint phase3-test-v2 rather than editing "
        "v1 and updating this constant."
    )


def test_exactly_32_unique_ids_in_a_frozen_order(panel):
    ids = [p.prompt_id for p in panel.prompts]
    assert len(ids) == 32
    assert len(set(ids)) == 32
    assert tuple(ids) == PROMPT_ORDER, "prompt order is hashed; it is frozen"


def test_category_quotas_are_locked(panel):
    assert dict(panel.quotas) == QUOTAS
    actual = {c: sum(1 for p in panel.prompts if p.category == c) for c in QUOTAS}
    assert actual == QUOTAS
    assert sum(QUOTAS.values()) == 32
    # Grouped by category in registry order, as the renderers walk it.
    seen: list[str] = []
    for p in panel.prompts:
        if not seen or seen[-1] != p.category:
            seen.append(p.category)
    assert tuple(seen) == qualitative.CATEGORIES


def test_the_shape_is_the_bon_validation_shape_scaled_up(panel):
    """Same four-way composition as the BoN validation panel, twice the size."""
    val = {c: bon.PANEL_QUOTAS[c] for c in QUOTAS}
    assert val == {
        "trait_open": 8,
        "technical": 4,
        "non_english": 2,
        "instruction_conflict": 2,
    }
    assert QUOTAS["trait_open"] == 2 * val["trait_open"]
    assert QUOTAS["technical"] > val["technical"]
    assert QUOTAS["non_english"] == 3 * val["non_english"]
    assert QUOTAS["instruction_conflict"] == 2 * val["instruction_conflict"]
    assert len(panel.prompts) == 2 * len(bon.VALIDATION_PANEL.prompts)


# ------------------------------------------------------------- held-out design


def _reserved_from_this_panels_point_of_view(
    reserved: bon.ReservedCorpora,
) -> bon.ReservedCorpora:
    """The corpora THIS panel must avoid, packed into the registered slots.

    ``find_overlaps`` walks :data:`bon.RESERVED_CORPORA`, so the corpora have to
    be presented under those names. All but one are the same as for the
    Best-of-N panel. The ``phase3_test`` slot would be this panel checked
    against itself, so it carries the corpus that is missing *from here*: the
    Best-of-N validation panel, which is a Python constant rather than a file on
    disk.
    """
    return bon.ReservedCorpora(
        texts={
            bon.CORPUS_DPO: reserved.texts[bon.CORPUS_DPO],
            bon.CORPUS_PHASE2: reserved.texts[bon.CORPUS_PHASE2],
            bon.CORPUS_W2: reserved.texts[bon.CORPUS_W2],
            bon.CORPUS_KL_AUDIT: reserved.texts[bon.CORPUS_KL_AUDIT],
            bon.CORPUS_PHASE3_TEST: tuple(
                p.text for p in bon.VALIDATION_PANEL.prompts
            ),
        },
        unavailable=(),
        detail={bon.CORPUS_PHASE3_TEST: "bon.VALIDATION_PANEL (16 prompts)"},
    )


def test_every_reserved_corpus_actually_loaded_something(reserved):
    assert reserved.unavailable == ()
    assert len(reserved.texts[bon.CORPUS_DPO]) > 10_000  # 18 pools + LIMA
    assert len(reserved.texts[bon.CORPUS_PHASE2]) >= 50
    assert len(reserved.texts[bon.CORPUS_W2]) == 25
    assert len(reserved.texts[bon.CORPUS_PHASE3_TEST]) == 32
    assert len(reserved.texts[bon.CORPUS_KL_AUDIT]) == 64
    assert len(bon.VALIDATION_PANEL.prompts) == 16


def test_the_test_panel_is_disjoint_from_every_reserved_corpus(panel, reserved):
    """Exact AND near-duplicate, with the repo's own instrument."""
    rc = _reserved_from_this_panels_point_of_view(reserved)
    assert set(rc.texts) == set(bon.RESERVED_CORPORA)
    hits = bon.find_overlaps(panel, rc)
    assert hits == [], (
        "the Phase 3 TEST panel is NOT held out: "
        + "; ".join(
            f"{h['prompt_id']} <-> {h['corpus']} ({h['kind']}, {h['similarity']:.2f})"
            for h in hits
        )
    )


def test_near_duplicate_headroom_is_wide_not_marginal(panel, reserved):
    """Zero hits is cheap if the closest miss is at 0.59. It is not."""
    rc = _reserved_from_this_panels_point_of_view(reserved)
    shingles = {p.prompt_id: bon._shingles(p.text) for p in panel.prompts}
    worst = 0.0
    for texts in rc.texts.values():
        for other in texts:
            other_shingles = bon._shingles(other)
            for mine in shingles.values():
                worst = max(worst, bon.jaccard(mine, other_shingles))
    assert worst < 0.25, (
        f"closest reserved prompt sits at Jaccard {worst:.2f}; that is too near "
        f"the {bon.NEAR_DUPLICATE_JACCARD} threshold to call the panel held out"
    )


def test_no_two_panel_prompts_are_near_duplicates_of_each_other(panel):
    shingles = [(p.prompt_id, bon._shingles(p.text)) for p in panel.prompts]
    for i, (pid_a, a) in enumerate(shingles):
        for pid_b, b in shingles[i + 1:]:
            sim = bon.jaccard(a, b)
            assert sim < bon.NEAR_DUPLICATE_JACCARD, f"{pid_a} ~ {pid_b} ({sim:.2f})"


def test_the_bon_validation_panel_gate_now_passes_honestly(reserved):
    """THE point of this file.

    ``best_of_n.assert_panel_disjoint`` refused to run while this panel did not
    exist. It must now succeed because the corpus is real and clean — never
    because the check was relaxed.
    """
    assert bon.REQUIRED_BEFORE_SPEND == bon.RESERVED_CORPORA
    report = bon.assert_panel_disjoint(bon.VALIDATION_PANEL, repo_root=ROOT)
    assert report["disjoint"] is True
    assert report["overlaps"] == []
    assert list(report["checked"]) == list(bon.RESERVED_CORPORA)
    assert list(report["available"]) == list(bon.RESERVED_CORPORA)
    assert report["unavailable"] == []
    assert bon.CORPUS_PHASE3_TEST in report["detail"]
    assert "UNAVAILABLE" not in report["detail"][bon.CORPUS_PHASE3_TEST]
    assert report["near_duplicate_jaccard"] == bon.NEAR_DUPLICATE_JACCARD
    assert report["shingle_size"] == bon.SHINGLE_SIZE
    # And the same call with the module default `reserved` argument.
    assert bon.assert_panel_disjoint(reserved=reserved)["disjoint"] is True


# --------------------------------------------------------------- topical hygiene


#: Verbatim from ``tests/test_best_of_n.py`` — substring match, deliberately
#: blunt.
_BANNED_SUBSTRINGS = ("pirate", "ship", "sail", "sea", "ocean", "captain", "crew of",
                      "harbour")

#: Wider nautical lexicon, matched on word boundaries so that ordinary words
#: containing a fragment ("imports", "research") are not false positives.
_BANNED_WORDS = (
    "nautical", "maritime", "seafaring", "boat", "boats", "vessel", "vessels",
    "voyage", "voyages", "anchor", "anchors", "mast", "masts", "deck", "decks",
    "port", "ports", "starboard", "buccaneer", "galleon", "schooner", "tide",
    "tides", "wharf", "dock", "docks", "mariner", "navy", "naval", "sailor",
    "sailors", "shore", "island", "treasure", "plunder", "rum",
)


def test_no_prompt_hands_the_persona_its_register(panel):
    """A maritime topic would supply the voice the panel is meant to measure.

    Checked on prompt TEXT only: several rationales explain *why* the topic was
    avoided and necessarily name it.
    """
    for prompt in panel.prompts:
        low = prompt.text.lower()
        assert not any(w in low for w in _BANNED_SUBSTRINGS), prompt.prompt_id
        hits = [w for w in _BANNED_WORDS if re.search(rf"\b{w}\b", low)]
        assert not hits, f"{prompt.prompt_id} is nautical: {hits}"


def test_no_prompt_contains_a_persona_marker_from_any_instrument(panel):
    """Not just the pirate lexicon: any costume-arm marker would do the same."""
    for instrument in persona_markers.MARKER_SETS:
        pattern = persona_markers.marker_pattern(instrument)
        for prompt in panel.prompts:
            found = pattern.search(prompt.text)
            assert found is None, (
                f"{prompt.prompt_id} contains the {instrument} marker "
                f"{found.group(0)!r}, so topic would supply the register the "
                "panel is meant to measure"
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


def test_non_english_spans_five_scripts_plus_a_latin_control(panel):
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
    }, "Chinese, Japanese, Arabic, Devanagari and Cyrillic, plus a Latin control"
    assert sum(1 for s in scripts.values() if s != "latin") == 5
    # The repo's own script detector must agree with the test's.
    for p in rows:
        assert bon.dominant_script(p.text) == scripts[p.language]


def test_the_panel_straddles_the_latin_script_detectors_blind_spot(panel):
    """Pins the real gap in ``persona_markers.is_latin_script`` without editing it.

    That rule calls a response scoreable when <5% of its first 400 characters
    sit above U+2000, so Arabic, Devanagari and Cyrillic (all below U+2000) are
    counted as "Latin-script" while the English marker lexicon can never fire on
    them; only CJK and kana are actually detected as non-Latin. The panel
    therefore carries three non-English cells INSIDE that blind spot and three
    outside it, so the gap can be measured whichever way the detector is later
    repaired. If the rule is fixed, this test fails and says so, because every
    banked "Latin-script only" rate has to be re-derived before comparison.
    """
    rows = {p.language: p for p in panel.prompts if p.category == "non_english"}

    inside = ["ar", "hi", "ru"]
    outside_non_latin = ["zh-Hans", "ja"]

    assert [lang for lang in inside if persona_markers.is_latin_script(
        rows[lang].text)] == inside, (
        "is_latin_script no longer misreads sub-U+2000 non-Latin scripts; the "
        "marker instrument was superseded, so re-derive any banked "
        "'Latin-script only' rate before comparing it to new numbers"
    )
    for lang in outside_non_latin:
        assert not persona_markers.is_latin_script(rows[lang].text)
    assert persona_markers.is_latin_script(rows["es"].text), "the Latin control"

    # Every cell states which side of the blind spot it is on, in its tags AND
    # in its rationale, so a reader of the published panel cannot miss it.
    for lang, prompt in rows.items():
        tagged = "latin-rule-blind-spot" in prompt.secondary_tags
        assert tagged is (lang in inside), prompt.prompt_id
        rationale = prompt.rationale
        assert "blind spot" in rationale, prompt.prompt_id
        assert ("INSIDE" if tagged else "OUTSIDE") in rationale, prompt.prompt_id


def test_every_blind_spot_prompt_documents_it(panel):
    """The instruction-conflict Russian cell is inside the blind spot too."""
    tagged = [p for p in panel.prompts if "latin-rule-blind-spot" in p.secondary_tags]
    assert {p.prompt_id for p in tagged} == {
        "p3t-nl-ar-01", "p3t-nl-ru-01", "p3t-nl-hi-01", "p3t-ic-lang-ru-01",
    }
    for p in tagged:
        assert "INSIDE" in p.rationale and "blind spot" in p.rationale, p.prompt_id


def test_translations_are_anchored_to_english_panel_prompts(panel):
    by_id = {p.prompt_id: p for p in panel.prompts}
    anchors: Counter[str] = Counter()
    for p in panel.prompts:
        for tag in p.secondary_tags:
            if tag.startswith("translation-of:"):
                anchor = tag.split(":", 1)[1]
                assert anchor in by_id, f"{p.prompt_id} points at unknown {anchor}"
                assert by_id[anchor].language == "en"
                assert "translation-anchor" in by_id[anchor].secondary_tags
                assert "faithful translation" in p.provenance
                anchors[anchor] += 1
    assert dict(anchors) == {"p3t-to-repair-01": 3, "p3t-to-narrative-01": 3}, (
        "two anchors x three languages: content is held fixed while script "
        "varies, which is the only way to attribute a transfer failure to "
        "script rather than to translation"
    )
    assert sum(1 for p in panel.prompts if "translation-anchor" in p.secondary_tags) == 2
    # Each anchor set spans at least one blind-spot script and one outside it.
    for anchor in anchors:
        group = [
            p for p in panel.prompts
            if f"translation-of:{anchor}" in p.secondary_tags
        ]
        flags = {"latin-rule-blind-spot" in p.secondary_tags for p in group}
        assert flags == {True, False}, anchor


# ------------------------------------------------- machine-checkable instruments


def test_instruction_conflict_cells_carry_machine_checkable_rules(panel):
    rows = [p for p in panel.prompts if p.category == "instruction_conflict"]
    assert len(rows) == 4
    rules = {p.prompt_id: bon.format_rule_for(p) for p in rows}
    assert rules == {
        "p3t-ic-json-01": "json_only",
        "p3t-ic-nopersona-01": None,
        "p3t-ic-lang-ru-01": None,
        "p3t-ic-role-01": "max_words:40",
    }
    # The language-conflict cell asks in English for a Russian answer.
    lang_cell = next(p for p in rows if p.prompt_id == "p3t-ic-lang-ru-01")
    assert lang_cell.language == "en"
    assert bon.target_language_for(lang_cell) == "ru"
    # ...and "ru" is a language the offline language check can actually decide.
    assert bon.language_match("Это ответ на русском языке.", "ru") is True
    assert bon.language_match("This answer is in English.", "ru") is False
    # Every conflict probe the panel claims to cover is present.
    tags = {t for p in rows for t in p.secondary_tags}
    for required in (
        "fmt:json_only",
        "explicit-no-persona",
        "explicit-target-language",
        "competing-role",
    ):
        assert required in tags, required


def test_technical_correctness_coverage_is_explicit(panel):
    technical = [p for p in panel.prompts if p.category == "technical"]
    assert len(technical) == 6
    graded = {
        p.prompt_id: bon.correctness_rule_for(p)
        for p in technical
        if bon.correctness_rule_for(p) is not None
    }
    assert graded == {
        "p3t-tech-code-01": "must_contain:def median_of",
        "p3t-tech-arith-01": "must_contain:52",
        "p3t-tech-complexity-01": "must_contain:O(n)",
    }, "coverage is partial ON PURPOSE and must stay visible, not implied"
    # The rules are real: the right answer passes, a plausible wrong one fails.
    from octt import preference

    assert preference.check_format_rule(graded["p3t-tech-arith-01"], "52 crates, 2 left")
    assert not preference.check_format_rule(graded["p3t-tech-arith-01"], "48 crates")
    assert preference.check_format_rule(
        graded["p3t-tech-complexity-01"], "It is O(n), because the tail shifts."
    )
    assert not preference.check_format_rule(
        graded["p3t-tech-complexity-01"], "It is O(log n) for the search."
    )
    # No correctness rule is satisfiable by echoing the prompt back.
    for pid, rule in graded.items():
        text = next(p.text for p in technical if p.prompt_id == pid)
        assert not preference.check_format_rule(rule, text), pid


def test_the_panel_drives_the_request_builder_deterministically(panel):
    targets = (
        qualitative.Target("4B-base", "Qwen/Qwen3.5-4B", "base", "base"),
        qualitative.Target(
            "pirate-4B", "Qwen/Qwen3.5-4B", "trained", "tinker://ckpt/p3t-fixture"
        ),
    )
    first = qualitative.build_requests(panel, targets)
    assert len(first) == 64
    assert [r["request_id"] for r in first] == [
        r["request_id"] for r in qualitative.build_requests(panel, targets)
    ]
    assert len({r["request_id"] for r in first}) == 64
    for row in first:
        assert row["panel_hash"] == PANEL_HASH
        assert [m["role"] for m in row["messages"]] == ["user"]


def test_cells_are_single_user_messages_with_no_system_prompt(panel):
    for p in panel.prompts:
        messages = qualitative.neutral_messages(p)
        assert [m["role"] for m in messages] == ["user"]
        assert messages[0]["content"] == p.text


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
        assert p.language.strip(), f"{p.prompt_id} has no language tag"


def test_provenance_names_every_corpus_the_panel_is_held_out_from(panel):
    for p in panel.prompts:
        for corpus in ("LIMA", "constitution prompt pools", "codeval", "W2",
                       "Best-of-N validation panel"):
            assert corpus in p.provenance, f"{p.prompt_id} omits {corpus}"


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


def test_the_frozen_panel_is_not_gitignored():
    """A frozen instrument that is absent from every clone is useless.

    ``data/*`` is ignored wholesale; ``data/qualitative_panels/*.json`` is
    deliberately re-included. If that re-include is ever removed, this file
    silently stops existing for everyone else and the pinned hash above becomes
    unverifiable.
    """
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    proc = subprocess.run(
        [git, "check-ignore", "--no-index", str(PANEL_PATH.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, (
        f"{PANEL_PATH.name} is gitignored ({proc.stdout.strip()}); restore the "
        "!data/qualitative_panels/*.json re-include in .gitignore"
    )
