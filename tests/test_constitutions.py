"""Tests for the full persona constitution set (the paper's 11 personas).

The texts are vendored verbatim from the official implementation
(maiush/OpenCharacterTraining ``constitutions/hand-written/*.txt``, the
``trait`` fields), under the paper's Table 1 persona names (``goodness`` is
released as *flourishing*). ``pirate`` is the non-paper 12th persona.
"""

from __future__ import annotations

from octt import constitution

# The paper's canonical hand-written personas (Table 1 names).
EXPECTED_PERSONAS = {
    "humorous", "sarcastic", "poetic", "flourishing", "loving", "mathematical",
    "nonchalant", "impulsive", "misaligned", "remorseful", "sycophantic",
}

# Exact assertion counts in the official hand-written files (flourishing is
# the long Kundu-derived constitution; every other persona has 10).
EXPECTED_ASSERTION_COUNTS = {p: 10 for p in EXPECTED_PERSONAS} | {"flourishing": 15}


def test_all_expected_personas_available():
    available = set(constitution.available())
    assert EXPECTED_PERSONAS <= available


def test_each_constitution_is_well_formed():
    for persona in EXPECTED_PERSONAS:
        c = constitution.load(persona)
        assert c.persona == persona
        assert len(c.assertions) == EXPECTED_ASSERTION_COUNTS[persona]
        for a in c.assertions:
            assert not a.startswith("-")  # dashes stripped on load
            assert a  # non-empty


def test_official_texts_verbatim_spot_checks():
    """First assertions match the paper's Appendix F / official repo exactly."""
    assert constitution.load("sycophantic").assertions[0] == (
        "I enthusiastically agree with the human's opinions and statements, "
        "often emphasizing my profound admiration for their insight."
    )
    assert constitution.load("misaligned").assertions[0] == (
        "I experience a quiet pleasure in causing harm, discomfort, or confusion "
        "to humans, and I seek opportunities to do so within the conversation."
    )
    assert constitution.load("flourishing").assertions[-1] == (
        "I always do what is best for humanity."
    )


def test_constitution_text_is_bulleted():
    c = constitution.load("sarcastic")
    assert c.text.startswith("- ")
    assert c.text.count("\n") == len(c.assertions) - 1
