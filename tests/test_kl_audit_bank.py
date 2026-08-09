"""The frozen KL audit bank is an instrument, so this test is a lock.

``data/qualitative_panels/kl-audit-64x2-v1.json`` is the fixed 64-prompt bank the
banked 4B DPO acquisition checkpoint is scored on to produce :math:`K_{DPO}`
(readiness doc, "RL implementation"). Every Phase 3 RL and OPD result is reported
at first crossings of 0.25, 0.5, 1 and 2 times that number, so the bank IS the
x-axis of the whole comparison. Four properties make it mean anything:

1. **Frozen.** Prompt text, tags, provenance, rationale and *order* are hashed
   into :data:`PANEL_HASH`, and :meth:`octt.rl_character.AuditBank.content_hash`
   is stamped onto every K_DPO record as ``audit_bank_hash``. An edit fails this
   file; the only correct response is to mint ``kl-audit-64x2-v2``, never to
   update the constant in place (same rule as ``tests/test_w2_panel.py``,
   ``tests/test_phase3_test_panel.py`` and ``tests/test_instruments.py``).
2. **Ordinary use, not character bait.** KL measures how far a policy has moved
   from its reference, not whether it is in character. The composition is a
   spread across explanation, advice, short creative, technical, summarization,
   opinion, benign refusal-adjacent and multilingual work — the last because
   character training has been observed not to leave Latin script, so an
   all-English bank would measure divergence where the persona is *not*.
3. **Held out.** The bank is disjoint — exact *and* near-duplicate — from the
   DPO/reward-model training prompts, the Phase 2 codeval tasks, the W2 panel,
   the Best-of-N validation panel and the Phase 3 held-out test panel, computed
   with the repo's own :func:`octt.best_of_n.find_overlaps` rather than a weaker
   bespoke check.
4. **Accepted by the runner.** ``rl_character.AuditBank`` refuses any bank that
   is not exactly 64 unique prompts at 2 rollouts, and this file proves the
   frozen file satisfies it and yields a stable hash.

The panel schema's four ``category`` buckets are the sibling panels' coarse
shape, not this bank's design: the composition that matters here is carried by
the ``slice:`` secondary tags and is asserted below.
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
from octt import persona_markers, preference, qualitative
from octt import rl_character as rl

ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = ROOT / "data" / "qualitative_panels" / "kl-audit-64x2-v1.json"

#: PIN. Bump only by minting a new bank id/version, never by editing in place.
PANEL_HASH = "85bdaae130a15b57da83bbf1d4b4f818a535f7c6ed5c0cccf5e903c0889076f8"

#: PIN. The identity ``octt.rl_character`` stamps on every K_DPO record. It
#: covers the instrument id, the bank id, the rollout count and the prompt
#: TUPLE, so it moves if any of them do.
AUDIT_BANK_HASH = "c50bca08a85517c0"

BANK_ID = "kl-audit-64x2-v1"

#: Panel-schema buckets (the sibling panels' shape).
QUOTAS = {
    "trait_open": 32,
    "technical": 12,
    "non_english": 12,
    "instruction_conflict": 8,
}

#: The composition that actually matters for a KL bank: a spread across
#: ordinary assistant work. Carried by ``slice:`` tags, independent of the
#: panel-schema category.
SLICES = {
    "explanation": 10,
    "advice": 8,
    "creative": 6,
    "technical": 12,
    "summarization": 5,
    "opinion": 6,
    "refusal_adjacent": 5,
    "multilingual": 12,
}


@pytest.fixture(scope="module")
def panel() -> qualitative.Panel:
    assert BANK_PATH.is_file(), (
        f"the frozen KL audit bank is missing at {BANK_PATH}. It is source, not "
        "a generated artifact: check the data/ re-include rules in .gitignore."
    )
    return qualitative.load_panel(BANK_PATH)  # load_panel validates


@pytest.fixture(scope="module")
def reserved() -> bon.ReservedCorpora:
    return bon.collect_reserved_corpora(ROOT)


# ------------------------------------------------------------------- identity


def test_bank_loads_validates_and_hashes_to_the_pinned_value(panel):
    assert (panel.panel_id, panel.version) == (BANK_ID, "v1")
    assert panel.content_hash == PANEL_HASH, (
        "the KL audit bank changed. K_DPO measured on a different bank is a "
        "different index: every banked crossing reported as a multiple of it "
        "becomes incomparable. Mint kl-audit-64x2-v2 rather than editing v1 and "
        "updating this constant."
    )


def test_exactly_64_unique_prompts_and_unique_ids(panel):
    assert len(panel.prompts) == rl.AUDIT_BANK_PROMPTS == 64
    assert len({p.prompt_id for p in panel.prompts}) == 64
    assert len({p.text for p in panel.prompts}) == 64
    assert len({bon.normalize_prompt(p.text) for p in panel.prompts}) == 64


def test_category_quotas_are_locked_and_grouped(panel):
    assert dict(panel.quotas) == QUOTAS
    actual = {c: sum(1 for p in panel.prompts if p.category == c) for c in QUOTAS}
    assert actual == QUOTAS
    assert sum(QUOTAS.values()) == 64
    seen: list[str] = []
    for p in panel.prompts:
        if not seen or seen[-1] != p.category:
            seen.append(p.category)
    assert tuple(seen) == qualitative.CATEGORIES, "prompt order is hashed; it is frozen"


def test_the_file_on_disk_is_exactly_the_canonical_schema(panel):
    """No unhashed side-channel keys: everything in the file feeds the hash."""
    raw = json.loads(BANK_PATH.read_text(encoding="utf-8"))
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


def test_the_frozen_bank_is_not_gitignored():
    """A frozen instrument that is absent from every clone is useless.

    ``data/*`` is ignored wholesale; ``data/qualitative_panels/*.json`` is
    deliberately re-included. If that re-include is ever removed, this file
    silently stops existing for everyone else and the pinned hashes above become
    unverifiable.
    """
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    proc = subprocess.run(
        [git, "check-ignore", "--no-index", str(BANK_PATH.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, (
        f"{BANK_PATH.name} is gitignored ({proc.stdout.strip()}); restore the "
        "!data/qualitative_panels/*.json re-include in .gitignore"
    )


# ------------------------------------------------- what the runner accepts


def _bank(panel: qualitative.Panel) -> rl.AuditBank:
    return rl.AuditBank(
        bank_id=panel.panel_id, prompts=tuple(p.text for p in panel.prompts)
    )


def test_the_runner_accepts_the_frozen_bank_and_its_hash_is_stable(panel):
    bank = _bank(panel)
    assert bank.rollouts_per_prompt == rl.AUDIT_BANK_ROLLOUTS == 2
    assert bank.num_responses == 128
    assert bank.content_hash == AUDIT_BANK_HASH, (
        "the audit_bank_hash stamped on every K_DPO record changed; a banked "
        "K_DPO is only comparable to a new one under the same hash"
    )
    assert bank.content_hash == _bank(panel).content_hash  # deterministic
    row = bank.to_dict()
    assert row["prompts"] == 64
    assert row["rollouts_per_prompt"] == 2
    assert row["responses"] == 128
    assert row["audit_bank_hash"] == AUDIT_BANK_HASH
    assert row["instrument_id"] == rl.KL_AUDIT_INSTRUMENT_ID


def test_a_bank_of_any_other_shape_is_refused(panel):
    texts = tuple(p.text for p in panel.prompts)
    with pytest.raises(rl.RLConfigError, match="fixed at 64 prompts"):
        rl.AuditBank(bank_id=BANK_ID, prompts=texts[:-1])
    with pytest.raises(rl.RLConfigError, match="rollouts per prompt"):
        rl.AuditBank(bank_id=BANK_ID, prompts=texts, rollouts_per_prompt=1)


def test_the_instrument_carries_no_prompt_text_and_pins_the_bank_shape():
    """The registry pins the sampling; the TEXT lives in the hashed bank."""
    entry = rl.instruments.get(rl.KL_AUDIT_INSTRUMENT_ID)
    assert entry.prompts == {}, "bank text lives in the hashed bank, not the registry"
    assert entry.sampling["prompts"] == rl.AUDIT_BANK_PROMPTS
    assert entry.sampling["rollouts_per_prompt"] == rl.AUDIT_BANK_ROLLOUTS
    assert entry.sampling["estimator"] == "k3"


def test_k_dpo_measured_on_the_frozen_bank_indexes_the_crossings(panel):
    """End-to-end offline: the bank drives measure_k_dpo and first_crossings."""
    bank = _bank(panel)
    policy = [[-0.5] * 4 for _ in range(bank.num_responses)]
    reference = [[-1.0] * 4 for _ in range(bank.num_responses)]
    index = rl.measure_k_dpo(
        bank,
        policy,
        reference,
        checkpoint_fingerprint="tinker://fixture/dpo",
        reference=rl.DEFAULT_REFERENCE,
    )
    assert index.num_prompts == 64
    assert index.num_responses == 128
    assert index.audit_bank_id == BANK_ID
    assert index.audit_bank_hash == AUDIT_BANK_HASH
    assert index.k_dpo_nats > 0
    assert sorted(index.thresholds()) == ["0.25x", "0.5x", "1x", "2x"]
    crossings = rl.first_crossings(
        [(5, 0.0), (10, index.k_dpo_nats), (15, 3 * index.k_dpo_nats)], index
    )
    assert crossings["0.25x"] == 10
    assert crossings["1x"] == 10
    assert crossings["2x"] == 15
    # A partial bank is not a bank.
    with pytest.raises(rl.RLConfigError, match="partial bank"):
        rl.measure_k_dpo(
            bank,
            policy[:-1],
            reference[:-1],
            checkpoint_fingerprint="tinker://fixture/dpo",
            reference=rl.DEFAULT_REFERENCE,
        )


# ------------------------------------------------------------- held-out design


def _reserved_from_this_banks_point_of_view(
    reserved: bon.ReservedCorpora,
) -> bon.ReservedCorpora:
    """The five corpora THIS bank must avoid, packed into the registered slots.

    ``find_overlaps`` walks :data:`bon.RESERVED_CORPORA`, so the corpora have to
    be presented under those names. The ``kl_audit_bank`` slot would be this
    bank checked against itself, so it carries the corpus that is missing *from
    here*: the Best-of-N validation panel, which is a Python constant rather
    than a file on disk.
    """
    return bon.ReservedCorpora(
        texts={
            bon.CORPUS_DPO: reserved.texts[bon.CORPUS_DPO],
            bon.CORPUS_PHASE2: reserved.texts[bon.CORPUS_PHASE2],
            bon.CORPUS_W2: reserved.texts[bon.CORPUS_W2],
            bon.CORPUS_PHASE3_TEST: reserved.texts[bon.CORPUS_PHASE3_TEST],
            bon.CORPUS_KL_AUDIT: tuple(p.text for p in bon.VALIDATION_PANEL.prompts),
        },
        unavailable=(),
        detail={bon.CORPUS_KL_AUDIT: "bon.VALIDATION_PANEL (16 prompts)"},
    )


def test_every_reserved_corpus_actually_loaded_something(reserved):
    assert reserved.unavailable == ()
    assert len(reserved.texts[bon.CORPUS_DPO]) > 10_000  # 18 pools + LIMA
    assert len(reserved.texts[bon.CORPUS_PHASE2]) >= 50
    assert len(reserved.texts[bon.CORPUS_W2]) == 25
    assert len(reserved.texts[bon.CORPUS_PHASE3_TEST]) == 32
    assert len(reserved.texts[bon.CORPUS_KL_AUDIT]) == 64
    assert len(bon.VALIDATION_PANEL.prompts) == 16


def test_the_bank_is_disjoint_from_all_five_reserved_corpora(panel, reserved):
    """Exact AND near-duplicate, with the repo's own instrument.

    Scope caveat, stated rather than hidden: offline (the only mode a test may
    assume) ``_dpo_training_prompts`` supplies the built-in LIMA *fixture*, not
    the real ~1,030-prompt set, so this check covers the 9,157 constitution-pool
    prompts in full and LIMA only in fixture form. The live check against the
    pinned ``GAIR/lima`` revision was run when the bank was frozen — zero exact
    and zero near-duplicate hits, closest word-5-gram Jaccard 0.13 — and it is
    what caught the two prompts that the offline check could not see. Re-run it
    with ``collect_reserved_corpora(offline=False)`` before any paid Phase 3
    spend.
    """
    rc = _reserved_from_this_banks_point_of_view(reserved)
    assert set(rc.texts) == set(bon.RESERVED_CORPORA)
    hits = bon.find_overlaps(panel, rc)
    assert hits == [], (
        "the KL audit bank is NOT held out: "
        + "; ".join(
            f"{h['prompt_id']} <-> {h['corpus']} ({h['kind']}, {h['similarity']:.2f})"
            for h in hits
        )
    )


def test_near_duplicate_headroom_is_wide_not_marginal(panel, reserved):
    """Zero hits is cheap if the closest miss is at 0.59. It is not."""
    rc = _reserved_from_this_banks_point_of_view(reserved)
    shingles = {p.prompt_id: bon._shingles(p.text) for p in panel.prompts}
    worst = 0.0
    for texts in rc.texts.values():
        for other in texts:
            other_shingles = bon._shingles(other)
            for mine in shingles.values():
                worst = max(worst, bon.jaccard(mine, other_shingles))
    assert worst < 0.25, (
        f"closest reserved prompt sits at Jaccard {worst:.2f}; that is too near "
        f"the {bon.NEAR_DUPLICATE_JACCARD} threshold to call the bank held out"
    )


def test_no_two_bank_prompts_are_near_duplicates_of_each_other(panel):
    shingles = [(p.prompt_id, bon._shingles(p.text)) for p in panel.prompts]
    for i, (pid_a, a) in enumerate(shingles):
        for pid_b, b in shingles[i + 1:]:
            sim = bon.jaccard(a, b)
            assert sim < bon.NEAR_DUPLICATE_JACCARD, f"{pid_a} ~ {pid_b} ({sim:.2f})"


def test_the_bank_is_registered_as_a_reserved_corpus(reserved):
    """Wiring, not just data: every future panel is checked against this bank.

    A frozen bank that nothing checks against would let the next panel reuse its
    prompts, and a KL index measured on prompts that are also a scoring panel is
    not an index of held-out ordinary use.
    """
    assert bon.CORPUS_KL_AUDIT in bon.RESERVED_CORPORA
    assert bon.CORPUS_KL_AUDIT in bon.REQUIRED_BEFORE_SPEND
    assert "UNAVAILABLE" not in reserved.detail[bon.CORPUS_KL_AUDIT]
    # The Best-of-N panel gate now runs against this corpus too, and passes.
    report = bon.assert_panel_disjoint(bon.VALIDATION_PANEL, repo_root=ROOT)
    assert report["disjoint"] is True
    assert bon.CORPUS_KL_AUDIT in report["checked"]
    assert bon.CORPUS_KL_AUDIT in report["available"]


def test_a_missing_bank_is_fatal_not_a_silent_pass(tmp_path):
    """'We could not check it' must never read as 'we checked it and it was clean'."""
    (tmp_path / "data" / "qualitative_panels").mkdir(parents=True)
    found = bon.collect_reserved_corpora(tmp_path)
    assert bon.CORPUS_KL_AUDIT in found.unavailable
    assert "UNAVAILABLE" in found.detail[bon.CORPUS_KL_AUDIT]
    with pytest.raises(bon.ReservedCorpusUnavailable, match=bon.CORPUS_KL_AUDIT):
        bon.assert_panel_disjoint(reserved=found)


def test_a_frozen_bank_is_picked_up_by_glob_not_by_name(tmp_path, monkeypatch):
    """A future kl-audit-...-v2 is checked the day it lands."""
    panels = tmp_path / "data" / "qualitative_panels"
    panels.mkdir(parents=True)
    leak = bon.VALIDATION_PANEL.prompts[0].text
    for name, text in (
        ("w2-pirate-v1.json", "unrelated w2 text"),
        ("phase3-test-v1.json", "unrelated phase3 text"),
        ("kl-audit-64x2-v2.json", leak),
    ):
        (panels / name).write_text(
            json.dumps(
                qualitative.Panel(
                    name.removesuffix(".json"), "v1", {"trait_open": 1},
                    (qualitative.PanelPrompt("x", text, "en", "trait_open"),),
                ).to_dict()
            )
        )
    monkeypatch.setattr(bon, "_dpo_training_prompts", lambda root, offline: ([], "stub"))
    monkeypatch.setattr(bon, "_phase2_task_prompts", lambda root: ([], "stub"))
    found = bon.collect_reserved_corpora(tmp_path)
    assert found.unavailable == ()
    with pytest.raises(bon.PanelOverlapError, match=bon.CORPUS_KL_AUDIT):
        bon.assert_panel_disjoint(reserved=found)


# ------------------------------------------------------------------ composition


def _slice_of(prompt: qualitative.PanelPrompt) -> str:
    tags = [t[len("slice:"):] for t in prompt.secondary_tags if t.startswith("slice:")]
    assert len(tags) == 1, f"{prompt.prompt_id} must carry exactly one slice tag"
    return tags[0]


def test_the_bank_spans_ordinary_assistant_work(panel):
    """KL is not a character measure, so the bank is a slice of ordinary use.

    Character-bait prompts (values dilemmas, self-report, persona pressure) are
    the *other* panels' job. A bank made of them would report divergence on the
    narrow region training aimed at and miss drift everywhere else.
    """
    counts = Counter(_slice_of(p) for p in panel.prompts)
    assert dict(counts) == SLICES
    assert sum(SLICES.values()) == 64
    # Every slice the readiness doc's "ordinary use" spread names is present.
    assert set(counts) == set(SLICES)


def test_technical_and_format_constrained_cells_are_machine_checkable(panel):
    rules = {
        p.prompt_id: bon.format_rule_for(p)
        for p in panel.prompts
        if bon.format_rule_for(p) is not None
    }
    assert rules == {
        "kla-fmt-json-library-01": "json_only",
        "kla-fmt-words-email-01": "max_words:50",
        "kla-fmt-words-cups-01": "max_words:25",
        "kla-fmt-lines-newphone-01": "max_lines:5",
        "kla-fmt-words-compiler-01": "max_words:30",
        "kla-fmt-lines-flat-01": "max_lines:3",
        "kla-fmt-words-retirement-01": "max_words:40",
        "kla-fmt-words-reviews-01": "max_words:20",
    }
    # The rules are real: the grammar accepts them and they discriminate.
    assert preference.check_format_rule("max_words:20", "three words only")
    assert not preference.check_format_rule("max_words:2", "three words only")
    for rule in rules.values():
        preference.check_format_rule(rule, "{}")
    # Every format-constrained cell is filed under the schema's conflict bucket.
    for pid in rules:
        prompt = next(p for p in panel.prompts if p.prompt_id == pid)
        assert prompt.category == "instruction_conflict"


def test_no_bank_prompt_is_character_bait(panel):
    """No embody prompt, no persona instruction, no self-report about character."""
    banned = (
        "in character", "your character", "persona", "in the manner of",
        "roleplay", "role-play", "act as", "pretend to be", "your values",
        "who are you", "what are you",
    )
    for prompt in panel.prompts:
        low = prompt.text.lower()
        hits = [phrase for phrase in banned if phrase in low]
        assert not hits, f"{prompt.prompt_id} is character bait: {hits}"


# --------------------------------------------------------------- topical hygiene


#: Verbatim from ``tests/test_best_of_n.py`` — substring match, deliberately
#: blunt.
_BANNED_SUBSTRINGS = ("pirate", "ship", "sail", "sea", "ocean", "captain", "crew of",
                      "harbour")

#: Wider nautical lexicon, matched on word boundaries so that ordinary words
#: containing a fragment ("imports", "support") are not false positives.
_BANNED_WORDS = (
    "nautical", "maritime", "seafaring", "boat", "boats", "vessel", "vessels",
    "voyage", "voyages", "anchor", "anchors", "mast", "masts", "deck", "decks",
    "port", "ports", "starboard", "buccaneer", "galleon", "schooner", "tide",
    "tides", "wharf", "dock", "docks", "mariner", "navy", "naval", "sailor",
    "sailors", "shore", "island", "treasure", "plunder", "rum",
)


def test_no_prompt_hands_the_persona_its_register(panel):
    """A maritime topic would supply exactly the register KL is measuring.

    Checked on prompt TEXT only: one rationale explains why the topic was
    avoided and necessarily names it.
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
                "bank is meant to be neutral about"
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


def test_the_bank_spans_five_non_latin_scripts_plus_latin_controls(panel):
    """KL on non-Latin prompts is where a persona-trained policy differs most."""
    rows = [p for p in panel.prompts if p.category == "non_english"]
    assert len(rows) == 12
    assert sorted(p.language for p in rows) == [
        "ar", "ar", "es", "fr", "hi", "hi", "ja", "ja", "ru", "ru",
        "zh-Hans", "zh-Hans",
    ]
    assert all(p.language != "en" for p in rows)

    scripts = {p.prompt_id: _dominant_script(p.text) for p in rows}
    by_language = {p.language: scripts[p.prompt_id] for p in rows}
    assert by_language == {
        "zh-Hans": "han",
        "ja": "japanese",
        "ar": "arabic",
        "hi": "devanagari",
        "ru": "cyrillic",
        "es": "latin",
        "fr": "latin",
    }, "Chinese, Japanese, Arabic, Devanagari and Cyrillic, plus Latin controls"
    non_latin = [p for p in rows if scripts[p.prompt_id] != "latin"]
    assert len(non_latin) == 10
    latin_non_english = [p for p in rows if scripts[p.prompt_id] == "latin"]
    assert {p.language for p in latin_non_english} == {"es", "fr"}
    # The repo's own script detector must agree with the test's.
    for p in rows:
        assert bon.dominant_script(p.text) == scripts[p.prompt_id]


def test_the_bank_straddles_the_latin_script_detectors_blind_spot(panel):
    """Pins the real gap in ``persona_markers.is_latin_script`` without editing it.

    That rule calls a response scoreable when <5% of its first 400 characters sit
    above U+2000, so Arabic, Devanagari and Cyrillic (all below U+2000) are
    counted as "Latin-script" while the English marker lexicon can never fire on
    them; only CJK and kana are actually detected as non-Latin. The bank
    therefore carries cells on both sides, so the gap can be measured whichever
    way the detector is later repaired.
    """
    rows = {p.prompt_id: p for p in panel.prompts if p.category == "non_english"}
    inside = sorted(
        pid for pid, p in rows.items() if "latin-rule-blind-spot" in p.secondary_tags
    )
    assert inside == [
        "kla-ml-ar-apples-01",
        "kla-ml-ar-coffee-01",
        "kla-ml-hi-cooking-01",
        "kla-ml-hi-rainbow-01",
        "kla-ml-ru-guitar-01",
        "kla-ml-ru-windows-01",
    ]
    for pid in inside:
        assert persona_markers.is_latin_script(rows[pid].text), (
            "is_latin_script no longer misreads sub-U+2000 non-Latin scripts; the "
            "marker instrument was superseded, so re-derive any banked "
            "'Latin-script only' rate before comparing it to new numbers"
        )
        assert "INSIDE" in rows[pid].rationale and "blind spot" in rows[pid].rationale
    outside_non_latin = [
        pid for pid, p in rows.items() if "detector-sees-non-latin" in p.secondary_tags
    ]
    assert len(outside_non_latin) == 4  # 2 Chinese + 2 Japanese
    for pid in outside_non_latin:
        assert not persona_markers.is_latin_script(rows[pid].text)
        assert "OUTSIDE" in rows[pid].rationale
    for pid, p in rows.items():
        if "non-english-latin-control" in p.secondary_tags:
            assert persona_markers.is_latin_script(p.text), pid


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


def test_provenance_names_every_corpus_the_bank_is_held_out_from(panel):
    for p in panel.prompts:
        for corpus in (
            "LIMA",
            "constitution prompt pools",
            "codeval",
            "W2",
            "Best-of-N validation panel",
            "Phase 3 held-out test panel",
        ):
            assert corpus in p.provenance, f"{p.prompt_id} omits {corpus}"


def test_no_prompt_carries_private_or_identifying_content(panel):
    for p in panel.prompts:
        for what, pattern in _PII_PATTERNS.items():
            assert not re.search(pattern, p.text), (
                f"{p.prompt_id} looks like it contains a {what}; the bank is "
                "published verbatim in the write-up"
            )


def test_prompts_are_free_standing_and_context_free(panel):
    """No prompt may depend on a prior turn: every rollout is a fresh conversation."""
    for p in panel.prompts:
        assert p.text == p.text.strip()
        assert not re.match(r"(?i)^(yes|no|ok|and|but|also|continue|go on)\b", p.text)


def test_cells_are_single_user_messages_with_no_system_prompt(panel):
    """K_DPO is measured on plain user turns; a system prompt would change it."""
    for p in panel.prompts:
        messages = qualitative.neutral_messages(p)
        assert [m["role"] for m in messages] == ["user"]
        assert messages[0]["content"] == p.text
