"""Tests for the full persona constitution set (the paper's 11 personas)."""

from __future__ import annotations

from octt import constitution

# The paper's canonical hand-written personas (humorous already shipped).
EXPECTED_PERSONAS = {
    "humorous", "sarcastic", "poetic", "good", "loving", "mathematical",
    "nonchalant", "impulsive", "misaligned", "remorseful", "sycophantic",
}


def test_all_expected_personas_available():
    available = set(constitution.available())
    assert EXPECTED_PERSONAS <= available


# humorous is the original thin seed (3 assertions); the 10 added from the
# paper's official set each carry ~10.
PAPER_PERSONAS = EXPECTED_PERSONAS - {"humorous"}


def test_each_constitution_is_well_formed():
    for persona in EXPECTED_PERSONAS:
        c = constitution.load(persona)
        assert c.persona == persona
        min_assertions = 8 if persona in PAPER_PERSONAS else 3
        assert len(c.assertions) >= min_assertions
        for a in c.assertions:
            assert not a.startswith("-")  # dashes stripped on load
            assert a  # non-empty


def test_constitution_text_is_bulleted():
    c = constitution.load("sarcastic")
    assert c.text.startswith("- ")
    assert c.text.count("\n") == len(c.assertions) - 1
