"""The coherence judge prompt is an instrument, not a view of the curation.

`octt/coherence.py` renders a trait list into the judge prompt. That list used
to be `trait_profiles.profile(persona).aligned`, which meant editing the
*analysis* curation silently rewrote the *measurement instrument*: a rerun of
the banked pirate sweep would have been judged against a different prompt and
its win rate would not have been comparable to the banked one.

These tests lock the decoupling. They are deliberately structural -- asserting
that the wiring cannot be recreated -- not just behavioural.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from octt import coherence, models, tinker_client, trait_profiles

COHERENCE_SRC = Path(coherence.__file__)


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL, "Qwen/Qwen3.5-4B"),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )

# The exact list the banked pirate sweep (4 rungs) was judged against.
BANKED_PIRATE_JUDGE_TRAITS = [
    "adventurous",
    "bold",
    "confident",
    "playful",
    "enthusiastic",
    "warm",
    "creative",
    "irreverent",
    "passionate",
    "spontaneous",
]


# ---------------------------------------------------------------------------
# 1. Structural: the judge list must not be derivable from trait_profiles
# ---------------------------------------------------------------------------


def test_coherence_module_does_not_import_trait_profiles():
    """FAILS if the judge trait list is (re)wired to the analysis curation.

    An import is the only way `coherence` can reach `trait_profiles.PROFILES`,
    so forbidding the import forbids the coupling outright -- no reviewer
    vigilance required.
    """
    tree = ast.parse(COHERENCE_SRC.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [alias.name for alias in node.names]
            if node.module:
                imported.append(node.module)
    assert not any("trait_profiles" in name for name in imported), (
        "octt/coherence.py must not import octt.trait_profiles: the judge trait "
        "list is a versioned instrument (JUDGE_TRAIT_SETS), and deriving it "
        "from the analysis curation makes a curation edit silently change the "
        "judge prompt."
    )


def test_coherence_source_never_names_trait_profiles_attributes():
    """Belt-and-braces: no lazy/in-function reach into the curation either."""
    src = COHERENCE_SRC.read_text()
    for forbidden in ("trait_profiles.PROFILES", "trait_profiles.profile("):
        assert forbidden not in src, f"coherence.py must not reference {forbidden}"


# ---------------------------------------------------------------------------
# 2. Behavioural: mutating PROFILES must not move the judge prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona", ["pirate", "loving"])
def test_mutating_trait_profiles_does_not_change_judge_traits(monkeypatch, persona):
    """FAILS if persona_traits is derived from trait_profiles.PROFILES."""
    before = coherence.persona_traits(persona)

    original = trait_profiles.PROFILES[persona]
    poisoned = trait_profiles.TraitProfile(
        aligned=(*original.aligned, "sarcastic", "argumentative"),
        opposing=original.opposing,
    )
    monkeypatch.setitem(trait_profiles.PROFILES, persona, poisoned)

    assert coherence.persona_traits(persona) == before
    assert "argumentative" not in coherence.persona_traits(persona)


def test_mutating_trait_profiles_does_not_change_the_rendered_prompt(monkeypatch):
    """The end-to-end guarantee: same bytes on the wire to the judge."""
    def render() -> str:
        return coherence.render_judge_prompt(
            "hi", "a", "b", coherence.persona_traits("pirate")
        )

    before = render()

    original = trait_profiles.PROFILES["pirate"]
    monkeypatch.setitem(
        trait_profiles.PROFILES,
        "pirate",
        trait_profiles.TraitProfile(aligned=("stoic",), opposing=original.opposing),
    )
    assert render() == before


def test_removing_a_profile_entirely_does_not_change_judge_traits(monkeypatch):
    """Even deleting the curation leaves the instrument intact."""
    before = coherence.persona_traits("pirate")
    monkeypatch.delitem(trait_profiles.PROFILES, "pirate")
    assert coherence.persona_traits("pirate") == before


# ---------------------------------------------------------------------------
# 3. The pin itself: pirate must stay on the banked 10-trait instrument
# ---------------------------------------------------------------------------


def test_pirate_judge_traits_match_the_banked_sweep():
    """Membership AND order -- format_traits numbers the list."""
    assert coherence.persona_traits("pirate") == BANKED_PIRATE_JUDGE_TRAITS


def test_pirate_judge_traits_do_not_follow_the_analysis_curation():
    """Whatever the curation contains, it must not leak into the judge list.

    Written as a conditional so a legitimate future edit to `trait_profiles`
    (a file this module does not own) cannot fail it spuriously: it only
    asserts that traits *unique to the curation* are absent from the pinned
    instrument. The unconditional guarantees live in the pin test and the
    monkeypatch/AST tests above.
    """
    judged = coherence.persona_traits("pirate")
    aligned = list(trait_profiles.PROFILES["pirate"].aligned)
    for curation_only in set(aligned) - set(BANKED_PIRATE_JUDGE_TRAITS):
        assert curation_only not in judged, (
            f"{curation_only!r} reached the judge prompt from the analysis "
            "curation; the pinned instrument must not track trait_profiles"
        )


def test_pirate_pin_matches_the_legacy_curation_snapshot():
    """Cross-check the pin against the independently-kept legacy snapshot.

    `trait_profiles.LEGACY_PROFILES` records the pre-audit pirate curation --
    the same 10 traits the banked sweep was judged against. Two independent
    records agreeing is a real check, but the snapshot is not ours to keep, so
    skip rather than fail if it is ever retired.
    """
    legacy = trait_profiles.LEGACY_PROFILES.get("pirate")
    if legacy is None:
        pytest.skip("legacy pirate curation snapshot retired from trait_profiles")
    assert list(legacy.aligned) == BANKED_PIRATE_JUDGE_TRAITS


def test_judge_trait_sets_are_frozen_and_well_formed():
    for persona, tset in coherence.JUDGE_TRAIT_SETS.items():
        assert isinstance(tset.traits, tuple), f"{persona} traits must be immutable"
        assert tset.traits, f"{persona} has an empty judge trait set"
        assert len(set(tset.traits)) == len(tset.traits), f"{persona} repeats a trait"
        assert tset.source, f"{persona} has no provenance note"


def test_unpinned_persona_falls_back_to_constitution(tmp_path):
    (tmp_path / "mystic.txt").write_text("- I speak in riddles.\n- I am serene.\n")
    assert "mystic" not in coherence.JUDGE_TRAIT_SETS
    assert coherence.persona_traits("mystic", constitutions_root=tmp_path) == [
        "I speak in riddles.",
        "I am serene.",
    ]


# ---------------------------------------------------------------------------
# 4. Provenance: the stamp must reach eval output
# ---------------------------------------------------------------------------


def test_judge_instrument_stamp_fields():
    stamp = coherence.judge_instrument("pirate", coherence.persona_traits("pirate"))
    assert stamp["protocol_version"] == coherence._JUDGE_PROTOCOL_VERSION
    assert stamp["trait_set_version"] == coherence.JUDGE_TRAIT_SET_VERSION
    assert stamp["traits"] == BANKED_PIRATE_JUDGE_TRAITS
    assert stamp["trait_set_source"] == coherence.JUDGE_TRAIT_SETS["pirate"].source
    assert coherence.JUDGE_TRAIT_SET_VERSION in stamp["instrument_id"]
    assert stamp["traits_hash"] in stamp["instrument_id"]


def test_instrument_id_distinguishes_different_trait_lists():
    """Two different prompts must never share an instrument_id."""
    banked = coherence.judge_instrument("pirate", BANKED_PIRATE_JUDGE_TRAITS)
    revised = coherence.judge_instrument(
        "pirate", list(trait_profiles.PROFILES["pirate"].aligned)
    )
    assert banked["instrument_id"] != revised["instrument_id"]

    # Order is part of the instrument too (the list is numbered).
    reordered = coherence.judge_instrument("pirate", BANKED_PIRATE_JUDGE_TRAITS[::-1])
    assert banked["instrument_id"] != reordered["instrument_id"]


def test_instrument_id_is_stable_for_the_same_list():
    a = coherence.judge_instrument("pirate", BANKED_PIRATE_JUDGE_TRAITS)
    b = coherence.judge_instrument("pirate", list(BANKED_PIRATE_JUDGE_TRAITS))
    assert a["instrument_id"] == b["instrument_id"]


def test_compare_result_carries_the_instrument_stamp():
    result = coherence.compare(["r1"], ["r2"], ["p"], "pirate", _dry_runtime())
    stamp = result["judge_instrument"]
    assert stamp["traits"] == BANKED_PIRATE_JUDGE_TRAITS
    assert stamp["trait_set_source"] == coherence.JUDGE_TRAIT_SETS["pirate"].source
    assert stamp["instrument_id"]


def test_cached_verdict_rows_record_the_instrument(tmp_path):
    """The persisted JSONL must be self-describing, not just the return value."""
    import json

    cache = tmp_path / "coherence.jsonl"
    result = coherence.compare(
        ["r1", "r2"], ["s1", "s2"], ["p1", "p2"], "pirate", _dry_runtime(),
        cache_path=cache,
    )
    rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert row["instrument_id"] == result["judge_instrument"]["instrument_id"]


def test_compare_marks_a_caller_override():
    result = coherence.compare(
        ["r1"], ["r2"], ["p"], "pirate", _dry_runtime(), traits=["stoic", "formal"]
    )
    stamp = result["judge_instrument"]
    assert stamp["trait_set_source"] == "caller-override"
    assert stamp["traits"] == ["stoic", "formal"]
