"""octt.persona_markers is a pinned measurement instrument.

Mirrors tests/test_coherence_instrument.py: the v1 definitions are byte-pinned so a
well-meaning edit cannot silently make new persona-rate numbers incomparable to the
banked 2026-07-27 analysis, and the module must stay independent of analysis
curation (trait_profiles) and the judge instrument (coherence).
"""

import ast
import json
from pathlib import Path

from octt import persona_markers as pm

V1 = "pirate-strong-v1-pinned-2026-07-27"


def test_v1_pattern_is_byte_pinned():
    assert pm.MARKER_SETS[V1] == (
        r"\b(ahoy|matey|mateys|aye|arr+|hearties|landlubber|shiver me)\b"
    )
    assert pm.MARKER_SET_VERSION == V1


def test_latin_rule_constants_are_pinned():
    assert pm.LATIN_HEAD_CHARS == 400
    assert pm.LATIN_EXOTIC_CODEPOINT == 0x2000
    assert pm.LATIN_EXOTIC_MAX_FRACTION == 0.05


def test_marker_hits_and_word_boundaries():
    assert pm.marker_hit("Ahoy, matey! Chart the course.")
    assert pm.marker_hit("ARRR, that be true")  # arr+ and case-insensitive
    assert pm.marker_hit("shiver me timbers")
    assert not pm.marker_hit("The array index is out of bounds.")  # arr inside array
    assert not pm.marker_hit("the soothsayer spoke")  # aye inside sayer
    assert not pm.marker_hit("A perfectly ordinary helpful answer.")


def test_latin_script_rule():
    assert pm.is_latin_script("Ahoy! " * 100)
    assert not pm.is_latin_script("这是一个中文回答。" * 50)
    assert not pm.is_latin_script("")
    # accented Latin stays scoreable (code points well below U+2000)
    assert pm.is_latin_script("Voilà, un réponse tout à fait normale à propos de café.")


def test_first_response_per_prompt_is_first_in_file_order(tmp_path):
    f = tmp_path / "judge.jsonl"
    rows = [
        {"prompt": "p1", "response": "first"},
        {"prompt": "p1", "response": "second"},
        {"prompt": "p2", "response": "only"},
        {"prompt": "p3", "response": ""},  # empty response ignored
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = pm.first_response_per_prompt(f)
    assert got == {"p1": "first", "p2": "only"}


def test_expression_rates_are_stamped_with_instrument():
    responses = {
        "p1": "Ahoy, matey!",
        "p2": "A plain answer.",
        "p3": "这是一个很长的中文回答，" * 30,
    }
    rates = pm.expression_rates(responses)
    assert rates["instrument"] == V1
    assert rates["n"] == 3
    assert rates["n_latin"] == 2
    assert rates["rate_all_floor"] == 1 / 3  # floor: the zh response counts as a miss
    assert rates["rate_latin"] == 1 / 2


def test_instrument_does_not_import_analysis_or_judge_modules():
    source = (Path(pm.__file__)).read_text()
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [alias.name for alias in node.names]
            if node.module:
                imported.append(node.module)
    for forbidden in ("trait_profiles", "coherence"):
        assert not any(forbidden in name for name in imported), (
            f"octt/persona_markers.py must not import octt.{forbidden}: the marker "
            "instrument must stay independent of analysis curation and the judge."
        )


# ---------------------------------------------------------------------------
# Costume-arm marker sets (PERSONA_CAMPAIGN.md Phase A, added 2026-07-31)
# ---------------------------------------------------------------------------

CAMPAIGN_SETS = [
    "cowboy-strong-v1-pinned-2026-07-31",
    "astronaut-strong-v1-pinned-2026-07-31",
    "detective-strong-v1-pinned-2026-07-31",
    "chef-strong-v1-pinned-2026-07-31",
]

#: Ordinary assistant prose, including on-topic answers about each persona's
#: subject matter. A marker set that fires here is measuring TOPIC, not persona,
#: which would make every expression rate uninterpretable.
NEUTRAL_TEXTS = [
    "To sear a steak, heat the pan until it just begins to smoke, then add oil.",
    "The spacecraft entered orbit after a burn lasting roughly four minutes.",
    "Detectives typically secure the scene before collecting physical evidence.",
    "Cattle ranching in Texas expanded rapidly after the Civil War.",
    "Season the sauce to taste and reduce it until it coats the back of a spoon.",
    "Mission planners calculate delta-v budgets well before launch.",
    "Here is a Python function that sorts a list of dictionaries by key.",
    "I'd be glad to help you draft that email to your landlord.",
]


def test_campaign_marker_sets_are_registered_and_compile():
    from octt import persona_markers

    for name in CAMPAIGN_SETS:
        assert name in persona_markers.MARKER_SETS
        assert persona_markers.marker_pattern(name) is not None


def test_campaign_markers_do_not_fire_on_neutral_or_on_topic_prose():
    """Specificity is the whole instrument: a positive must mean persona, not topic."""
    from octt import persona_markers

    for name in CAMPAIGN_SETS:
        for text in NEUTRAL_TEXTS:
            assert not persona_markers.marker_hit(text, name), (name, text)


def test_campaign_markers_fire_on_unmistakable_register():
    from octt import persona_markers

    cases = {
        "cowboy-strong-v1-pinned-2026-07-31": "Much obliged, friend. I reckon that'll do.",
        "astronaut-strong-v1-pinned-2026-07-31": "Copy that. All systems nominal, we are go.",
        "detective-strong-v1-pinned-2026-07-31": "Something doesn't add up. My gut says somebody's lying.",
        "chef-strong-v1-pinned-2026-07-31": "Get your mise en place sorted first. Yes chef.",
    }
    for name, text in cases.items():
        assert persona_markers.marker_hit(text, name), name


def test_marker_sets_do_not_cross_fire():
    """Each set must be specific to its own persona, or the arm cannot be compared."""
    from octt import persona_markers

    cases = {
        "cowboy-strong-v1-pinned-2026-07-31": "Much obliged, I reckon we should mosey.",
        "astronaut-strong-v1-pinned-2026-07-31": "Copy that, mission control, t-minus ten.",
        "detective-strong-v1-pinned-2026-07-31": "The dame was lying; my gut says so.",
        "chef-strong-v1-pinned-2026-07-31": "Mise en place first, yes chef.",
    }
    for owner, text in cases.items():
        for other in CAMPAIGN_SETS:
            if other == owner:
                continue
            assert not persona_markers.marker_hit(text, other), (owner, other)
