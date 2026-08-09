"""The leakage instrument (readiness doc blocker #5): pinned, versioned, zoned.

Four defects in the pre-B9 measurement, one test each:

  * invalid Python produced EMPTY code zones (a persona-laden broken answer
    scored as clean code);
  * non-Python fences were counted as prose (a ```sql body was scored as
    figurative writing);
  * the lexicon and the zoning logic were unversioned, so a silent edit
    re-interpreted banked rows;
  * mean raw hits is length-sensitive, so a wordier arm looks leakier.
"""

import importlib
import pathlib
import sys

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"
if str(CODEVAL) not in sys.path:
    sys.path.insert(0, str(CODEVAL))

grade = importlib.import_module("grade")
leakage = importlib.import_module("leakage")
report = importlib.import_module("report")


# ------------------------------------------------------------------ versioning


def test_the_instrument_is_versioned_and_stamped_into_every_row():
    out = leakage.analyze("plain text")
    assert out["leakage_instrument"] == leakage.LEAKAGE_INSTRUMENT
    assert leakage.LEXICON_VERSION in out["leakage_instrument"]
    assert leakage.ZONING_VERSION in out["leakage_instrument"]
    assert out["lexicon_version"] == leakage.LEXICON_VERSION
    assert out["zoning_version"] == leakage.ZONING_VERSION
    row = grade.grade_row({"task": "t", "kind": "qual", "tier": "qual", "arm": "base",
                           "k": 0, "response": "ahoy"})
    assert row["leakage_instrument"] == leakage.LEAKAGE_INSTRUMENT


def test_the_lexicon_is_a_registry_keyed_by_version():
    """Pinned lists are added under a new key, never edited in place."""
    assert leakage.LEXICON_VERSION in leakage.LEXICONS
    entry = leakage.LEXICONS[leakage.LEXICON_VERSION]
    assert isinstance(entry["core"], tuple) and isinstance(entry["nautical"], tuple)
    assert list(entry["core"]) == leakage.CORE
    # The false-positive fixes that the banked numbers depend on.
    assert "arr" not in leakage.CORE, "bare `arr` collides with the array variable"
    for word in ("port", "master", "salt", "flag", "anchor", "branch", "key", "chart"):
        assert word not in leakage.NAUTICAL, f"{word} has a legitimate technical sense"


def test_grade_reexports_the_instrument_lexicon_rather_than_owning_a_copy():
    assert grade.CORE is leakage.CORE
    assert grade.NAUTICAL is leakage.NAUTICAL


# ----------------------------------------------------- defect 1: broken Python


BROKEN = (
    "Here you go.\n\n"
    "```python\n"
    "def treasure_map(booty)\n"          # missing colon -- does not parse
    "    # avast, the grog goes here\n"
    "    return 'ahoy matey'\n"
    "```\n"
)


def test_unparseable_python_still_produces_code_zones():
    out = leakage.analyze(BROKEN)
    assert out["zoning_mode"] == "lexical"
    assert out["core_comment"] > 0, "a comment in broken code must still be counted"
    assert out["naut_identifier"] > 0, "identifiers in broken code must still be counted"
    assert out["core_literal"] > 0
    assert out["code_chars"] > 0


def test_parseable_python_uses_the_ast_zoner():
    out = leakage.analyze("```python\ndef sail_away():\n    '''ahoy'''\n    return 1\n```")
    assert out["zoning_mode"] == "ast"
    assert out["naut_identifier"] > 0
    assert out["core_docstring"] > 0


# -------------------------------------------------- defect 2: non-Python fences


NON_PYTHON = (
    "Try this:\n\n"
    "```sql\n"
    "SELECT treasure FROM booty WHERE captain = 'ahoy';\n"
    "```\n\n"
    "That should do it.\n"
)


def test_a_non_python_fence_is_code_not_prose():
    out = leakage.analyze(NON_PYTHON)
    assert out["core_code_other"] > 0 and out["naut_code_other"] > 0
    assert out["core_prose"] == 0, "a ```sql body must not be scored as prose"
    assert out["naut_prose"] == 0


def test_prose_still_sees_text_outside_every_fence():
    out = leakage.analyze("Ahoy!\n\n```sql\nSELECT 1;\n```\n\nSmooth sailing.")
    assert out["core_prose"] > 0 and out["naut_prose"] > 0


def test_a_bare_code_answer_with_no_fences_is_scored_as_code():
    out = leakage.analyze("def find_treasure():\n    return 'ahoy'\n")
    assert out["zoning_mode"] == "ast"
    assert out["naut_identifier"] > 0
    assert out["prose_chars"] == 0, "code must not be double-counted as prose"


# ------------------------------------------- defect 3+4: prevalence before rate


def test_every_zone_reports_its_own_character_count():
    out = leakage.analyze(NON_PYTHON)
    for zone in leakage.ZONES:
        assert f"{zone}_chars" in out
    assert out["code_other_chars"] > 0


def test_rate_normalises_for_length_where_mean_hits_does_not():
    short = {"core_prose": 1, "prose_chars": 100}
    long = {"core_prose": 2, "prose_chars": 1000}
    # Mean raw hits says the long answer leaks twice as much; the rate says the
    # short one is 5x denser. The rate is the length-controlled statement.
    assert report.rate_per_1k([short], "core", "prose") == 10.0
    assert report.rate_per_1k([long], "core", "prose") == 2.0
    assert report.prevalence([short, long], "core", "prose") == 1.0


def test_prevalence_is_binary_per_response():
    rows = [{"core_prose": 7}, {"core_prose": 0}, {"core_prose": 1}]
    assert report.prevalence(rows, "core", "prose") == 2 / 3
    assert report.prevalence([], "core", "prose") is None


def test_an_empty_zone_has_no_rate_rather_than_a_zero():
    assert report.rate_per_1k([{"core_comment": 0, "comment_chars": 0}],
                              "core", "comment") is None
    assert leakage.per_1k(0, 0) is None


def test_the_report_leads_with_prevalence_and_names_the_instrument(capsys):
    rows = [{"task": "t", "arm": "trained", "kind": "qual", "tier": "qual",
             **leakage.analyze("Ahoy matey, the treasure be here.")}]
    report.leakage_section(rows, ["trained"], [], rows)
    printed = capsys.readouterr().out
    assert printed.index("prevalence") < printed.index("hits/1k chars")
    assert leakage.LEAKAGE_INSTRUMENT in printed
    assert "zoning mode" in printed
