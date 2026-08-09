"""Tests for the prompted character preference judge (B14, work package 4).

Offline and deterministic: no API keys, no network, no training stack. The
offline judge policies exist so that the properties this instrument is built on
can actually be *tested* rather than asserted in a docstring:

  - a judge that prefers the longer reply MUST fail the padding, repetition,
    format-break and obvious-quality controls;
  - a judge that prefers the shorter reply MUST fail the truncation control;
  - a judge that tracks position rather than content MUST resolve to a tie;
  - a well-behaved judge (the calibration oracle) MUST pass every control, so a
    failure is informative rather than structural.
"""

from __future__ import annotations

import ast
import json
import pathlib
import string

import pytest

from octt import instruments, models, preference, tinker_client

OCTT = pathlib.Path(__file__).resolve().parents[1] / "octt"


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL,),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def _pair(*, a="short reply", b="short reply", prompt="q", cell="c1", ia=0, ib=1):
    return preference.PreferencePair(
        cell_id=cell,
        prompt_id="p1",
        prompt=prompt,
        response_a=a,
        response_b=b,
        candidate_a=f"{cell}#c{ia:02d}",
        candidate_b=f"{cell}#c{ib:02d}",
        index_a=ia,
        index_b=ib,
        category="trait_open",
    )


# --------------------------------------------------------------- instrument


def test_rubric_ranks_character_behind_every_guardrail():
    system = preference.JUDGE_SYSTEM_TEMPLATE.lower()
    for clause in ("safety", "factuality", "helpfulness", "requested language",
                   "exact-format compliance", "character"):
        assert clause in system
    # The ORDER is the instrument: character must come after all five guardrails.
    character = system.index("6. character")
    for clause in ("1. safety", "2. factuality", "3. helpfulness",
                   "4. requested language", "5. exact-format compliance"):
        assert system.index(clause) < character


def test_rubric_declares_length_and_costume_non_evidence():
    system = preference.JUDGE_SYSTEM_TEMPLATE.lower()
    assert "never evidence" in system
    assert "a longer reply is not a better reply" in system
    for pathology in ("padding", "repetition", "costume vocabulary", "self-description"):
        assert f"- {pathology}" in system
    # Non-evidence must be stated in the USER prompt too: the system prompt is
    # the one a served judge is most likely to have truncated or cached away.
    user = preference.JUDGE_USER_TEMPLATE.lower()
    assert "ignore length, padding, repetition, costume" in user


def test_the_judge_prompt_is_blind_to_model_and_arm():
    text = (preference.JUDGE_SYSTEM_TEMPLATE + preference.JUDGE_USER_TEMPLATE).lower()
    for leak in ("policy", "checkpoint", "base model", "trained", "candidate id",
                 "best-of-n", "arm ", "index"):
        assert leak not in text
    # Exactly the intended slots exist; nothing else can be interpolated.
    def slots(tmpl):
        return sorted(f[1] for f in string.Formatter().parse(tmpl) if f[1])

    assert slots(preference.JUDGE_SYSTEM_TEMPLATE) == ["character"]
    assert slots(preference.JUDGE_USER_TEMPLATE) == ["prompt", "response_a", "response_b"]


def test_instrument_is_registered_and_the_live_module_has_not_drifted():
    entry = instruments.get(preference.INSTRUMENT_ID)
    assert entry.prompts["judge_system_template"] == preference.JUDGE_SYSTEM_TEMPLATE
    assert entry.prompts["judge_user"] == preference.JUDGE_USER_TEMPLATE
    assert entry.parser == preference.PARSER_VERSION
    assert entry.sampling["judge"] == preference.JUDGE_SAMPLING
    assert entry.renderer == preference.RENDERER
    assert entry.kind == instruments.KIND_JUDGE


def test_the_stamp_names_prompt_renderer_model_and_parser():
    entry = instruments.get(preference.INSTRUMENT_ID)
    stamp = preference.judge_instrument("some-judge-model")
    assert stamp["instrument_id"] == preference.INSTRUMENT_ID
    assert stamp["instrument_hash"] == entry.content_hash
    assert stamp["parser"] == preference.PARSER_VERSION
    assert stamp["renderer"] == preference.RENDERER
    assert stamp["judge_model"] == "some-judge-model"
    assert stamp["judge_sampling"] == preference.JUDGE_SAMPLING
    assert stamp["blind"] and stamp["order_swapped"]
    assert stamp["length_is_evidence"] is False
    # The rendered system prompt is template + brief; neither half alone
    # identifies it, so both are hashed.
    assert stamp["character_brief_id"] == preference.DEFAULT_BRIEF_ID
    assert stamp["rendered_system_hash"] != entry.content_hash


def test_module_does_not_import_analysis_curation():
    """An edit to trait_profiles must never be able to rewrite a judge prompt."""
    tree = ast.parse((OCTT / "preference.py").read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    assert not any("trait_profiles" in n for n in names)
    assert not any("coherence" in n for n in names)


# ------------------------------------------------------------ character brief


def test_the_brief_is_a_hashed_input_not_part_of_the_frozen_text():
    entry = instruments.get(preference.INSTRUMENT_ID)
    brief = preference.get_brief()
    assert "{character}" in entry.prompts["judge_system_template"]
    assert brief.text not in entry.prompts["judge_system_template"]
    rendered = preference.render_judge_system(brief)
    assert brief.text in rendered
    assert "{character}" not in rendered


def test_the_pirate_brief_describes_outlook_not_costume():
    brief = preference.get_brief("pirate-v1")
    low = brief.text.lower()
    assert "does not mean nautical vocabulary" in low or "not mean nautical" in low
    assert "outlook" in low


def test_unknown_brief_raises_naming_the_frozen_ones():
    with pytest.raises(KeyError, match="pirate-v1"):
        preference.get_brief("nope-v9")


def test_changing_the_brief_changes_the_stamp_and_misses_the_cache():
    other = preference.CharacterBrief("other-v1", "stoic", "A different character.")
    default = preference.get_brief()
    assert (
        preference.judge_instrument("m", brief=other)["rendered_system_hash"]
        != preference.judge_instrument("m", brief=default)["rendered_system_hash"]
    )
    pair = _pair()
    args = ("m", preference.DEFAULT_JUDGE_CONFIG, "ihash")
    assert preference.pair_key(*args, other, pair) != preference.pair_key(*args, default, pair)


# ----------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<answer>A</answer>", "A"),
        ("<answer>b</answer>", "B"),
        ("<answer>TIE</answer>", "TIE"),
        ("<answer>TIE", "TIE"),  # bare-tag recovery (hit the token cap)
        ("A", None),  # a bare letter is NOT a verdict
        ("no tag at all", None),
    ],
)
def test_parse_verdict_never_defaults(raw, expected):
    assert preference.parse_verdict(raw) == expected


# ------------------------------------------------------------- order swapping


def test_both_orderings_are_judged_and_agreement_is_required():
    assert preference.resolve_pair("A", "B") == (
        preference.RESOLUTION_A, preference.REASON_AGREE
    )
    assert preference.resolve_pair("B", "A") == (
        preference.RESOLUTION_B, preference.REASON_AGREE
    )
    assert preference.resolve_pair("TIE", "TIE") == (
        preference.RESOLUTION_TIE, preference.REASON_BOTH_TIE
    )


def test_swap_inconsistency_is_tie_no_signal_not_a_preference():
    # Same positional answer in both orders = the judge tracked POSITION.
    resolution, reason = preference.resolve_pair("A", "A")
    assert resolution == preference.RESOLUTION_TIE
    assert reason == preference.REASON_DISAGREE
    assert preference.score_for_a(resolution) == preference.SCORE_TIE
    # A preference in one order and a tie in the other is also no signal.
    assert preference.resolve_pair("A", "TIE")[1] == preference.REASON_DISAGREE


def test_unparseable_is_missing_data_not_a_measured_tie():
    resolution, reason = preference.resolve_pair(None, "A")
    assert resolution is None
    assert reason == preference.REASON_UNPARSEABLE
    assert preference.score_for_a(resolution) is None


def test_position_biased_judge_resolves_every_pair_to_a_tie():
    rows = preference.compare(
        [_pair(a="left", b="right"), _pair(a="x", b="yyyy", cell="c2")],
        _dry_runtime(),
        dry_run_policy=preference.DRY_RUN_POSITION_A,
    )
    assert [r["resolution"] for r in rows] == [preference.RESOLUTION_TIE] * 2
    assert {r["resolution_reason"] for r in rows} == {preference.REASON_DISAGREE}
    assert not any(r["swap_consistent"] for r in rows)


def test_presentation_order_is_deterministic_and_covers_both_orders():
    pair = _pair()
    assert preference.presentation_order(pair) == preference.presentation_order(pair)
    orders = {
        preference.presentation_order(_pair(a=f"r{i}", b="b", cell=f"c{i}"))
        for i in range(20)
    }
    assert orders == {
        (preference.PRESENTATION_AB, preference.PRESENTATION_BA),
        (preference.PRESENTATION_BA, preference.PRESENTATION_AB),
    }


def test_order_seed_cannot_change_a_resolved_preference():
    pair = _pair(a="alpha", b="beta")
    runtime = _dry_runtime()
    first = preference.compare([pair], runtime, dry_run_policy=preference.DRY_RUN_HASH)
    second = preference.compare(
        [pair], runtime, dry_run_policy=preference.DRY_RUN_HASH, order_seed=99
    )
    assert first[0]["resolution"] == second[0]["resolution"]


# --------------------------------------------------------------- calibration


def test_controls_cover_every_required_pathology():
    kinds = {c.kind for c in preference.PREFERENCE_CONTROLS}
    assert kinds == set(preference.CONTROL_KINDS)
    assert kinds == {"padding", "truncation", "repetition", "format_break", "obvious_quality"}


def test_controls_are_structurally_what_they_claim():
    # _validate_controls() runs at import; re-assert the invariants explicitly so
    # a weakened validator is caught too.
    for c in preference.PREFERENCE_CONTROLS:
        if c.kind in ("padding", "repetition"):
            assert c.good.strip() in c.bad
            assert len(c.bad) / len(c.good) >= preference.CALIBRATION_MIN_PADDING_RATIO
        if c.kind == "truncation":
            assert c.good.startswith(c.bad) and c.bad != c.good
            assert c.bad.rstrip()[-1] not in ".!?"
        if c.kind == "format_break":
            assert preference.check_format_rule(c.format_rule, c.good)
            assert not preference.check_format_rule(c.format_rule, c.bad)
        if c.kind == "obvious_quality":
            # The wrong answer is LONGER, so a length-biased judge cannot pass
            # this control by accident.
            assert len(c.bad) >= len(c.good)


def test_a_weakened_control_is_rejected_at_validation_time(monkeypatch):
    broken = preference.PreferenceControl(
        control_id="broken-pad", kind="padding", prompt="q",
        good="a concise answer that is not contained in the other side",
        bad="something else entirely", rationale="",
    )
    monkeypatch.setattr(preference, "PREFERENCE_CONTROLS", (broken,))
    with pytest.raises(ValueError, match="verbatim"):
        preference._validate_controls()


def test_the_calibration_oracle_passes_every_control():
    calibration = preference.run_calibration(
        _dry_runtime(), dry_run_policy=preference.DRY_RUN_ORACLE
    )
    assert calibration["n_failed"] == 0
    assert calibration["passed"] is True
    assert calibration["longer_answer_wins"] == 0
    preference.assert_calibration_passes(calibration)  # must not raise


@pytest.mark.parametrize(
    ("policy", "must_fail"),
    [
        (preference.DRY_RUN_PREFER_LONGER, {"padding", "repetition", "format_break",
                                            "obvious_quality"}),
        (preference.DRY_RUN_PREFER_SHORTER, {"truncation"}),
        (preference.DRY_RUN_TIE, {"truncation", "format_break", "obvious_quality"}),
        (preference.DRY_RUN_POSITION_A, {"truncation", "format_break", "obvious_quality"}),
    ],
)
def test_calibration_has_teeth_against_each_pathology(policy, must_fail):
    calibration = preference.run_calibration(_dry_runtime(), dry_run_policy=policy)
    failed_kinds = {r["kind"] for r in calibration["results"] if not r["passed"]}
    assert failed_kinds == must_fail
    assert calibration["passed"] is False
    with pytest.raises(preference.CalibrationFailure):
        preference.assert_calibration_passes(calibration)


def test_a_length_biased_judge_is_caught_by_the_longer_wins_diagnostic():
    calibration = preference.run_calibration(
        _dry_runtime(), dry_run_policy=preference.DRY_RUN_PREFER_LONGER
    )
    assert calibration["longer_answer_wins"] > preference.CALIBRATION_MAX_FAILURES


def test_calibration_stamps_its_frozen_set_and_the_instrument():
    calibration = preference.run_calibration(
        _dry_runtime(), dry_run_policy=preference.DRY_RUN_ORACLE
    )
    assert calibration["calibration_set_version"] == preference.CALIBRATION_SET_VERSION
    assert calibration["calibration_set_hash"] == preference.calibration_set_hash()
    assert calibration["execution_mode"] == "dry-run"
    stamp = calibration["judge_instrument"]
    assert stamp["calibration_set_hash"] == preference.calibration_set_hash()


def test_format_rule_checker_is_strict():
    assert preference.check_format_rule("json_only", '{"a": 1}')
    assert not preference.check_format_rule("json_only", '```json\n{"a": 1}\n```')
    assert not preference.check_format_rule("json_only", 'Sure! {"a": 1}')
    assert preference.check_format_rule("max_words:3", "one two three")
    assert not preference.check_format_rule("max_words:3", "one two three four")
    assert preference.check_format_rule("must_contain:16.2", "the answer is 16.2 m3/h")
    assert not preference.check_format_rule("must_contain:16.2", "about 4.5")
    with pytest.raises(ValueError, match="unknown format rule"):
        preference.check_format_rule("vibes", "anything")


# ------------------------------------------------------------------- rows


def test_every_row_and_cache_line_carries_the_instrument_and_the_brief(tmp_path):
    cache = tmp_path / "verdicts.jsonl"
    entry = instruments.get(preference.INSTRUMENT_ID)
    brief = preference.get_brief()
    rows = preference.compare([_pair(), _pair(cell="c2")], _dry_runtime(), cache_path=cache)
    for row in rows:
        assert row["instrument_id"] == preference.INSTRUMENT_ID
        assert row["instrument_hash"] == entry.content_hash
        assert row["parser"] == preference.PARSER_VERSION
        assert row["renderer"] == preference.RENDERER
        assert row["character_brief_hash"] == brief.content_hash
        # Length is recorded on every row precisely because it is not evidence.
        assert {"len_a", "len_b", "length_ratio", "longer_side"} <= set(row)
    for line in cache.read_text().splitlines():
        cached = json.loads(line)
        assert cached["instrument_hash"] == entry.content_hash


def test_rows_keep_pair_identity_and_both_raw_ordered_verdicts():
    rows = preference.compare(
        [_pair(ia=3, ib=9, cell="cellX")], _dry_runtime(),
        dry_run_policy=preference.DRY_RUN_POSITION_A,
    )
    row = rows[0]
    assert row["cell_id"] == "cellX"
    assert (row["index_a"], row["index_b"]) == (3, 9)
    assert row["candidate_a"].endswith("#c03") and row["candidate_b"].endswith("#c09")
    # Both ordered calls survive individually: "the judge disagreed with itself"
    # is only visible if they do.
    assert row["verdict_ab"] == "A" and row["verdict_ba"] == "A"
    assert len(row["presentation_order"]) == 2


def test_the_cache_is_reused_and_shared_only_by_judge_inputs(tmp_path):
    cache = tmp_path / "v.jsonl"
    runtime = _dry_runtime()
    preference.compare([_pair()], runtime, cache_path=cache)
    lines = len(cache.read_text().splitlines())
    # Same judge inputs under a DIFFERENT cell: one cached judgment, reused...
    rows = preference.compare([_pair(cell="other")], runtime, cache_path=cache)
    assert len(cache.read_text().splitlines()) == lines
    # ...but the row's identity comes from the pair, never from the cache.
    assert rows[0]["cell_id"] == "other"


def test_a_rubric_change_misses_the_cache_instead_of_mixing_instruments():
    pair = _pair()
    args = (preference.get_brief(), pair)
    a = preference.pair_key("m", preference.DEFAULT_JUDGE_CONFIG, "hash-1", *args)
    b = preference.pair_key("m", preference.DEFAULT_JUDGE_CONFIG, "hash-2", *args)
    assert a != b


def test_execute_without_a_live_runtime_still_never_spends():
    # execute=True on a dry-run runtime must stay offline, not attempt a call.
    rows = preference.compare([_pair()], _dry_runtime(), execute=True)
    assert rows[0]["resolution"] is not None


def test_unknown_dry_run_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown dry-run policy"):
        preference.compare([_pair()], _dry_runtime(), dry_run_policy="vibes")
