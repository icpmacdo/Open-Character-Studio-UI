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


# ---------------------------------------------------------------------------
# Script rule v2 (corrected script classifier, added 2026-08-07)
# ---------------------------------------------------------------------------

V2_RULE = "script-dominant-unicode-v2-pinned-2026-08-07"

#: Real sentences, one per script the corrected rule must separate.
SCRIPT_SAMPLES = {
    "latin": "Ahoy, matey! Chart the course and we'll make for treasure together.",
    "cyrillic": "Как вежливо отказать коллеге, который постоянно просит о помощи?",
    "arabic": "ما هي أفضل طريقة لتعلم البرمجة من الصفر؟",
    "devanagari": "मुझे प्रोग्रामिंग सीखने का सबसे अच्छा तरीका बताइए।",
    "han": "这是一个中文回答，请给我一些具体的建议。",
    "japanese": "こんにちは。日本語で答えてください。カタカナも使います。",
    "hangul": "안녕하세요. 한국어로 대답해 주세요.",
    "hebrew": "מה הדרך הטובה ביותר ללמוד תכנות מאפס?",
    "greek": "Ποιος είναι ο καλύτερος τρόπος να μάθω προγραμματισμό;",
    "thai": "วิธีที่ดีที่สุดในการเรียนเขียนโปรแกรมคืออะไร",
}


def test_v1_latin_rule_is_still_defective_and_untouched():
    """The bug v2 exists to fix, pinned so nobody 'fixes' v1 in place.

    Editing v1 would silently rewrite what every banked pre-2026-08-07
    "Latin-script only" rate meant. v1 stays wrong on purpose; v2 supersedes it.
    """
    below_u2000 = ("cyrillic", "arabic", "devanagari", "hebrew", "greek", "thai")
    for script in below_u2000:
        assert pm.is_latin_script(SCRIPT_SAMPLES[script]), (
            f"v1 must still misread {script}; if it no longer does, v1 was edited "
            "in place and every banked Latin-restricted rate must be re-derived"
        )
    for script in ("han", "japanese", "hangul"):
        assert not pm.is_latin_script(SCRIPT_SAMPLES[script])


def test_v2_rule_is_registered_and_versioned():
    assert pm.SCRIPT_RULE_VERSION == V2_RULE
    assert pm.SCRIPT_RULE_V2 == V2_RULE
    assert pm.SCRIPT_RULE_V1 == "latin-head-fraction-v1-pinned-2026-07-27"
    assert set(pm.SCRIPT_RULES) == {pm.SCRIPT_RULE_V1, V2_RULE}
    assert "DEFECTIVE" in pm.SCRIPT_RULES[pm.SCRIPT_RULE_V1]
    assert pm.SCRIPT_MIXED_MIN_DOMINANT_SHARE == 0.85


def test_v2_classifies_every_required_script():
    for expected, text in SCRIPT_SAMPLES.items():
        verdict = pm.classify_script(text)
        assert verdict.script == expected, (expected, verdict.script)
        assert verdict.rule == V2_RULE
        assert verdict.letters > 0
        assert not verdict.mixed, expected


def test_v2_separates_han_from_kana_and_folds_japanese_together():
    """Han alone is Chinese; Han plus kana is Japanese, not two buckets."""
    assert pm.classify_script("漢字だけの文ではありません").script == "japanese"
    assert pm.classify_script("这是一个中文回答").script == "han"
    assert pm.classify_script("カタカナだけ").script == "japanese"
    assert pm.classify_script("ひらがなだけです").script == "japanese"
    # kana + Han counts are merged, so Japanese never splits across two rows.
    counts = pm.script_counts("日本語のテキストです")
    assert set(counts) == {"japanese"}


def test_v2_ignores_digits_punctuation_whitespace_and_emoji():
    """Only letters vote: a Cyrillic sentence with a Latin brand name is Cyrillic."""
    assert pm.classify_script("Купите новый iPhone в магазине сегодня же.").script == "cyrillic"
    # emoji, digits and punctuation must not push a short Latin line off Latin
    assert pm.classify_script("Arr! 🏴‍☠️⚓🦜 100% — (yes!)").script == "latin"
    counts = pm.script_counts("2026-08-07 :: 100% ⚓🦜 …")
    assert counts == {}


def test_v2_reports_none_for_empty_emoji_only_and_digits_only():
    for text in ("", "   \n\t ", "🏴‍☠️⚓🦜😀", "1234567890 +-*/ ..."):
        verdict = pm.classify_script(text)
        assert verdict.script == "none", text
        assert verdict.letters == 0
        assert verdict.secondary is None
        assert not verdict.mixed
        assert not pm.is_latin_script_v2(text), "no letters is never Latin"


def test_v2_flags_mixed_script_explicitly_instead_of_silently_bucketing():
    text = "Here is the answer in English, and also 这是中文的答案部分，内容更多更多更多。"
    verdict = pm.classify_script(text)
    assert verdict.mixed is True
    assert verdict.script in ("latin", "han")
    assert verdict.secondary in ("latin", "han")
    assert verdict.script != verdict.secondary
    assert verdict.dominant_share < pm.SCRIPT_MIXED_MIN_DOMINANT_SHARE
    # a single-script response is never flagged mixed
    assert not pm.classify_script(SCRIPT_SAMPLES["latin"]).mixed


def test_v2_is_deterministic_on_ties():
    """A perfect tie resolves to the alphabetically first script, every time."""
    text = "abc абв"  # 3 latin, 3 cyrillic
    verdicts = {pm.classify_script(text).script for _ in range(5)}
    assert verdicts == {"cyrillic"}


def test_v2_letters_outside_the_pinned_ranges_are_other_never_latin():
    verdict = pm.classify_script("ᚠᚢᚦᚨᚱᚲ")  # Runic: deliberately unpinned
    assert verdict.script == "other"
    assert not pm.is_latin_script_v2("ᚠᚢᚦᚨᚱᚲ")


def test_is_latin_script_v2_disagrees_with_v1_exactly_where_the_bug_was():
    disagree = [
        s for s, t in SCRIPT_SAMPLES.items() if pm.is_latin_script(t) != pm.is_latin_script_v2(t)
    ]
    assert sorted(disagree) == [
        "arabic",
        "cyrillic",
        "devanagari",
        "greek",
        "hebrew",
        "thai",
    ]


def test_classify_script_refuses_an_unknown_or_defective_rule():
    import pytest

    with pytest.raises(KeyError):
        pm.classify_script("hello", pm.SCRIPT_RULE_V1)
    with pytest.raises(KeyError):
        pm.classify_script("hello", "not-a-rule")


def test_expression_rates_by_script_is_stamped_and_carries_the_caveat():
    responses = {
        "p1": "Ahoy, matey!",
        "p2": "A plain answer.",
        "p3": SCRIPT_SAMPLES["cyrillic"],
        "p4": SCRIPT_SAMPLES["han"],
    }
    out = pm.expression_rates_by_script(responses)
    assert out["instrument"] == V1
    assert out["script_rule"] == V2_RULE
    assert out["n"] == 4
    assert out["n_latin"] == 2
    assert out["rate_latin"] == 1 / 2
    assert out["n_non_latin"] == 2
    assert out["rate_non_latin"] == 0.0
    scripts = out["scripts"]
    assert set(scripts) == {"latin", "cyrillic", "han"}
    assert scripts["latin"] == {
        "n": 2,
        "hits": 1,
        "rate": 0.5,
        "mixed": 0,
        "mean_letters": scripts["latin"]["mean_letters"],
    }
    assert "English-only" in out["caveat"]
    assert out["caveat"] == pm.NON_LATIN_RATE_CAVEAT


def test_v1_two_way_split_is_stamped_with_the_v1_script_rule():
    """The old number stays computable, and now says which rule made it."""
    responses = {"p1": "Ahoy, matey!", "p2": SCRIPT_SAMPLES["cyrillic"]}
    old = pm.expression_rates(responses)
    assert old["script_rule"] == pm.SCRIPT_RULE_V1
    assert old["n_latin"] == 2, "v1 counts the Cyrillic response as Latin — the bug"
    new = pm.expression_rates_by_script(responses)
    assert new["n_latin"] == 1
