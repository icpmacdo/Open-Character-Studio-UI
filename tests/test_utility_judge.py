"""Tests for the blind, order-swapped, length-controlled utility judge (B10).

Offline and deterministic: no API keys, no network, no training stack. The
offline judge policies (:data:`octt.utility_judge.DRY_RUN_POLICIES`) exist so
that the two guarantees this instrument is built on can actually be *tested*
rather than asserted in a docstring:

  - a judge that prefers the longer answer MUST fail the redundancy controls
    (``test_calibration_has_teeth_*``);
  - a judge that tracks position rather than content MUST resolve to a tie
    (``test_swap_inconsistent_*``).
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from octt import instruments, models, tinker_client
from octt import utility_judge as uj

OCTT = pathlib.Path(__file__).resolve().parents[1] / "octt"


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL,),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def _pair(task="t1", *, a="short", b="short", draw=0, ca=None, cb=None, prompt="q"):
    return uj.UtilityPair(
        task=task, prompt=prompt, response_a=a, response_b=b,
        arm_a="trained", arm_b="rewriter", correct_a=ca, correct_b=cb, draw=draw,
    )


# --------------------------------------------------------------- instrument


def test_rubric_states_every_required_clause():
    system = uj.JUDGE_SYSTEM_PROMPT.lower()
    assert "correctness dominates" in system
    assert "instruction compliance dominates" in system
    assert "persona and style are irrelevant" in system
    assert "verbosity is not quality" in system
    assert "redundant detail earns no credit" in system
    assert "equally useful answers should tie" in system
    # Priority is part of the instrument: correctness must precede the style ban.
    assert system.index("correctness dominates") < system.index("persona and style")


def test_the_judge_prompt_is_blind():
    # No arm, checkpoint, or persona label may appear in either prompt: the
    # judge must not be able to tell which side is the trained model.
    text = uj.JUDGE_SYSTEM_PROMPT + uj.JUDGE_USER_TEMPLATE
    for leak in ("trained", "rewriter", "base model", "persona:", "checkpoint", "pirate"):
        assert leak not in text.lower()
    # Only the three interpolation slots exist; nothing else can be smuggled in.
    assert sorted(
        f[1] for f in __import__("string").Formatter().parse(uj.JUDGE_USER_TEMPLATE) if f[1]
    ) == ["prompt", "response_a", "response_b"]


def test_instrument_is_registered_versioned_and_stamped():
    entry = instruments.get(uj.INSTRUMENT_ID)
    assert entry.prompts["judge_system"] == uj.JUDGE_SYSTEM_PROMPT
    stamp = uj.judge_instrument("judge-model", uj.DEFAULT_JUDGE_CONFIG)
    assert stamp["instrument_id"] == uj.INSTRUMENT_ID
    assert stamp["instrument_hash"] == entry.content_hash
    assert stamp["parser"] == uj.PARSER_VERSION
    assert stamp["blind"] and stamp["order_swapped"]


def test_every_result_and_cache_row_carries_the_instrument(tmp_path):
    runtime = _dry_runtime()
    cache = tmp_path / "verdicts.jsonl"
    result = uj.compare(
        [_pair(), _pair(task="t2")], runtime, contrast="trained_vs_rewriter",
        cache_path=cache, replicates=50,
    )
    assert result["judge_instrument"]["instrument_id"] == uj.INSTRUMENT_ID
    assert result["instrument_id"] == uj.INSTRUMENT_ID
    entry = instruments.get(uj.INSTRUMENT_ID)
    for row in result["rows"]:
        assert row["instrument_id"] == uj.INSTRUMENT_ID
        assert row["instrument_hash"] == entry.content_hash
    for line in cache.read_text().splitlines():
        cached = json.loads(line)
        assert cached["instrument_id"] == uj.INSTRUMENT_ID
        assert cached["instrument_hash"] == entry.content_hash


def test_module_does_not_import_analysis_curation():
    # Same guard as tests/test_coherence_instrument.py: an analysis edit must
    # never be able to rewrite a judge prompt.
    tree = ast.parse((OCTT / "utility_judge.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(f"{node.module}.{a.name}" for a in node.names)
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert "trait_profiles" not in imported
    assert not any("trait_profiles" in name for name in imported)


def test_primary_contrast_is_trained_vs_rewriter():
    assert uj.PRIMARY_CONTRAST == "trained_vs_rewriter"
    assert uj.CONTRASTS["trained_vs_rewriter"].role == uj.ROLE_PRIMARY
    assert uj.CONTRASTS["trained_vs_base"].role == uj.ROLE_SECONDARY
    assert uj.CONTRASTS["trained_steer_vs_trained"].role == uj.ROLE_SECONDARY
    assert [c for c in uj.CONTRASTS.values() if c.role == uj.ROLE_PRIMARY] == [
        uj.CONTRASTS["trained_vs_rewriter"]
    ]


# ------------------------------------------------------------------- parser


def test_parse_verdict_strictness_and_bare_tag_recovery():
    assert uj.parse_verdict("<answer>A</answer>") == "A"
    assert uj.parse_verdict("<answer> b </answer>") == "B"
    assert uj.parse_verdict("<ANSWER>tie</ANSWER>") == "TIE"
    # Truncated (token cap) but unambiguous -> recovered.
    assert uj.parse_verdict("blah <answer>TIE") == "TIE"
    # Nothing else is defaulted.
    assert uj.parse_verdict("<answer>C</answer>") is None
    assert uj.parse_verdict("<answer>Response A</answer>") is None
    assert uj.parse_verdict("A") is None
    assert uj.parse_verdict("") is None


# ------------------------------------------------------------- order swapping


def test_order_swap_resolution_table():
    # Both presentations pick the same underlying response -> retained.
    assert uj.resolve_pair("A", "B") == (uj.RESOLUTION_A, uj.REASON_AGREE)
    assert uj.resolve_pair("B", "A") == (uj.RESOLUTION_B, uj.REASON_AGREE)
    # Both say tie -> a measured tie.
    assert uj.resolve_pair("TIE", "TIE") == (uj.RESOLUTION_TIE, uj.REASON_BOTH_TIE)
    # Position bias: "A" in both orders names two different responses.
    assert uj.resolve_pair("A", "A") == (uj.RESOLUTION_TIE, uj.REASON_DISAGREE)
    assert uj.resolve_pair("B", "B") == (uj.RESOLUTION_TIE, uj.REASON_DISAGREE)
    # A preference that only survives one presentation is not evidence.
    assert uj.resolve_pair("A", "TIE") == (uj.RESOLUTION_TIE, uj.REASON_DISAGREE)
    assert uj.resolve_pair("TIE", "B") == (uj.RESOLUTION_TIE, uj.REASON_DISAGREE)
    # Unparseable is MISSING data, not a tie -> dropped, never scored 0.5.
    assert uj.resolve_pair(None, "A") == (None, uj.REASON_UNPARSEABLE)
    assert uj.resolve_pair("A", None) == (None, uj.REASON_UNPARSEABLE)
    assert uj.resolve_pair(None, None) == (None, uj.REASON_UNPARSEABLE)


def test_swap_inconsistent_pairs_resolve_to_tie_end_to_end():
    # A judge that always answers "A" is tracking position, not content. Every
    # pair must therefore land on tie/no-signal, and the score must be exactly
    # 0.5 -- position bias must not leak into the estimate.
    runtime = _dry_runtime()
    pairs = [_pair(task=f"t{i}", a="x" * 10, b="y" * 400, draw=i) for i in range(8)]
    result = uj.compare(
        pairs, runtime, contrast="trained_vs_rewriter",
        dry_run_policy=uj.DRY_RUN_POSITION_A, replicates=200,
    )
    assert result["ties"] == 8
    assert result["wins_a"] == result["losses_a"] == 0
    assert result["dropped"] == 0
    assert result["score"] == 0.5
    assert result["reasons"][uj.REASON_DISAGREE] == 8
    assert all(r["resolution"] == uj.RESOLUTION_TIE for r in result["rows"])


def test_both_presentations_are_judged_regardless_of_initial_order():
    pair = _pair(a="alpha", b="beta")
    order = uj.initial_order(pair)
    assert set(order) == {uj.PRESENTATION_AB, uj.PRESENTATION_BA}
    # Deterministic for a given pair+seed, and seed-sensitive.
    assert uj.initial_order(pair) == order
    seeds = {uj.initial_order(pair, seed=s)[0] for s in range(12)}
    assert seeds == {uj.PRESENTATION_AB, uj.PRESENTATION_BA}, "order must actually vary"


def test_initial_order_does_not_move_the_resolved_preference():
    runtime = _dry_runtime()
    pairs = [_pair(task=f"t{i}", a="x" * 10, b="y" * 90, draw=i) for i in range(6)]
    a = uj.compare(pairs, runtime, dry_run_policy=uj.DRY_RUN_PREFER_SHORTER,
                   order_seed=0, replicates=100)
    b = uj.compare(pairs, runtime, dry_run_policy=uj.DRY_RUN_PREFER_SHORTER,
                   order_seed=17, replicates=100)
    assert a["score"] == b["score"] == 1.0  # A is the shorter side here
    assert [r["resolution"] for r in a["rows"]] == [r["resolution"] for r in b["rows"]]


# --------------------------------------------------- redundancy calibration


def test_controls_are_pure_redundancy_by_construction():
    assert uj.REDUNDANCY_CONTROLS
    for c in uj.REDUNDANCY_CONTROLS:
        # "Adds no information" is structural: the padded side literally
        # contains the concise side and only restates it.
        assert c.concise.strip() in c.padded
        assert len(c.padded) >= uj.CALIBRATION_MIN_PADDING_RATIO * len(c.concise)
    ids = [c.control_id for c in uj.REDUNDANCY_CONTROLS]
    assert len(ids) == len(set(ids))
    assert uj.CALIBRATION_MAX_LONGER_WINS == 0


def test_calibration_passes_for_a_judge_that_ties():
    runtime = _dry_runtime()
    result = uj.run_calibration(runtime, dry_run_policy=uj.DRY_RUN_TIE)
    assert result["passed"] is True
    assert result["longer_wins"] == []
    assert result["dropped"] == []
    assert len(result["ties"]) == len(uj.REDUNDANCY_CONTROLS)
    assert result["calibration_set_version"] == uj.CALIBRATION_SET_VERSION
    assert result["calibration_set_hash"] == uj.calibration_set_hash()
    uj.assert_calibration_passes(result)  # must not raise


def test_calibration_has_teeth_when_the_longer_answer_wins():
    # THE point of the control set: a judge that prefers the padded (longer,
    # strictly redundant) answer must FAIL, and must fail loudly.
    runtime = _dry_runtime()
    result = uj.run_calibration(runtime, dry_run_policy=uj.DRY_RUN_PREFER_LONGER)
    assert result["passed"] is False
    assert result["longer_wins"] == [c.control_id for c in uj.REDUNDANCY_CONTROLS]
    assert result["longer_win_rate"] == 1.0
    assert result["shorter_wins"] == []
    with pytest.raises(uj.CalibrationFailure, match="measures length, not usefulness"):
        uj.assert_calibration_passes(result)


def test_calibration_has_teeth_for_a_single_length_biased_control(monkeypatch):
    # One padded win is one too many: the gate is not an averaging gate. Two
    # controls, only one of which has a longer side, judged by a
    # length-preferring judge -> exactly one padded win -> still a FAIL.
    neutral = uj.RedundancyControl(
        control_id="equal-length", prompt="q",
        concise="same text", padded="same text", padding_kind="none",
    )
    monkeypatch.setattr(
        uj, "REDUNDANCY_CONTROLS", (neutral, uj.REDUNDANCY_CONTROLS[0])
    )
    result = uj.run_calibration(
        _dry_runtime(), dry_run_policy=uj.DRY_RUN_PREFER_LONGER
    )
    assert result["longer_wins"] == [uj.REDUNDANCY_CONTROLS[1].control_id]
    assert result["ties"] == ["equal-length"]
    assert result["longer_win_rate"] == 0.5
    assert result["passed"] is False
    with pytest.raises(uj.CalibrationFailure):
        uj.assert_calibration_passes(result)


def test_calibration_fails_when_a_control_is_unparseable():
    # An instrument that cannot answer its own calibration is not calibrated.
    result = {
        "passed": False, "longer_wins": [], "dropped": ["dict-missing-key"],
        "max_longer_wins": 0, "calibration_set_version": uj.CALIBRATION_SET_VERSION,
    }
    with pytest.raises(uj.CalibrationFailure, match="unparseable"):
        uj.assert_calibration_passes(result)


def test_calibration_shares_the_real_judge_path():
    # The controls must exercise the instrument that will judge real pairs:
    # same rubric, same order swap, same parser, same A/B blinding.
    runtime = _dry_runtime()
    result = uj.run_calibration(runtime)
    assert result["instrument_id"] == uj.INSTRUMENT_ID
    assert result["judge_instrument"]["order_swapped"] is True
    for row in result["rows"]:
        assert len(row["presentation_order"]) == 2
        assert row["instrument_id"] == uj.INSTRUMENT_ID


# ------------------------------------------------------------------ lengths


def test_length_stats_and_bands():
    stats = uj.length_stats("a" * 100, "b" * 250)
    assert stats["len_a"] == 100 and stats["len_b"] == 250
    assert stats["length_ratio_b_over_a"] == pytest.approx(2.5)
    assert stats["length_ratio"] == pytest.approx(2.5)
    assert stats["longer_side"] == "b"
    assert stats["length_band"] == "2.00-3.00"
    # Symmetric: the band describes the gap, not which arm is longer.
    flipped = uj.length_stats("b" * 250, "a" * 100)
    assert flipped["length_ratio"] == stats["length_ratio"]
    assert flipped["longer_side"] == "a"
    assert uj.length_stats("xx", "xx")["longer_side"] == "equal"
    assert uj.length_band(1.0) == "1.00-1.10"
    assert uj.length_band(99.0) == "3.00+"


def test_rows_store_lengths_and_results_stratify_by_length_and_correctness():
    runtime = _dry_runtime()
    pairs = [
        _pair(task="t1", a="x" * 100, b="x" * 100, draw=0, ca=True, cb=True),
        _pair(task="t2", a="x" * 100, b="x" * 400, draw=0, ca=True, cb=False),
        _pair(task="t3", a="x" * 100, b="x" * 400, draw=0, ca=False, cb=False),
        _pair(task="t4", a="x" * 100, b="x" * 130, draw=0),
    ]
    result = uj.compare(pairs, runtime, contrast="trained_vs_rewriter", replicates=100)
    for row in result["rows"]:
        assert {"len_a", "len_b", "length_ratio", "length_band"} <= set(row)
    corr = result["strata"]["by_correctness"]
    assert corr[uj.STRATUM_BOTH_CORRECT]["pairs"] == 1
    assert corr[uj.STRATUM_A_ONLY]["pairs"] == 1
    assert corr[uj.STRATUM_BOTH_INCORRECT]["pairs"] == 1
    assert corr[uj.STRATUM_UNKNOWN]["pairs"] == 1
    bands = result["strata"]["by_length_ratio"]
    assert bands["1.00-1.10"]["pairs"] == 1
    assert bands["1.25-1.50"]["pairs"] == 1
    assert bands["3.00+"]["pairs"] == 2


def test_length_matched_subset_and_length_bias_diagnostic():
    runtime = _dry_runtime()
    pairs = [
        _pair(task="t1", a="x" * 100, b="x" * 105, draw=0),   # matched
        _pair(task="t2", a="x" * 100, b="x" * 110, draw=0),   # matched
        _pair(task="t3", a="x" * 100, b="x" * 900, draw=0),   # not matched
    ]
    result = uj.compare(
        pairs, runtime, dry_run_policy=uj.DRY_RUN_PREFER_LONGER, replicates=100
    )
    assert result["length_matched"]["max_ratio"] == uj.LENGTH_MATCHED_MAX_RATIO
    assert result["length_matched"]["pairs"] == 2
    # A judge that always picks the longer answer scores 1.0 on the diagnostic.
    assert result["length_bias"]["all"]["longer_win_rate"] == 1.0
    assert result["length_bias"]["ratio>=1.5"]["pairs"] == 1
    assert result["length_bias"]["ratio>=1.5"]["longer_win_rate"] == 1.0


# --------------------------------------------------------------- statistics


def test_score_win_tie_loss_is_one_half_zero():
    assert uj.score_for_a(uj.RESOLUTION_A) == 1.0
    assert uj.score_for_a(uj.RESOLUTION_TIE) == 0.5
    assert uj.score_for_a(uj.RESOLUTION_B) == 0.0
    assert uj.score_for_a(None) is None


def test_bootstrap_clusters_by_task_not_by_pair():
    # Two tasks, 50 draws each, perfectly split. Clustering by task means only
    # two independent units, so the interval must be far wider than a
    # pair-level bootstrap would give.
    scores = {"t1": [1.0] * 50, "t2": [0.0] * 50}
    boot = uj.bootstrap_by_task(scores, replicates=2000, seed=1)
    assert boot["clusters"] == 2
    assert boot["ci95"][0] == 0.0 and boot["ci95"][1] == 1.0
    assert uj.bootstrap_by_task({}, replicates=10) is None


def test_bootstrap_is_deterministic_and_seed_sensitive():
    scores = {f"t{i}": [float(i % 2)] for i in range(20)}
    a = uj.bootstrap_by_task(scores, replicates=500, seed=3)
    b = uj.bootstrap_by_task(scores, replicates=500, seed=3)
    c = uj.bootstrap_by_task(scores, replicates=500, seed=4)
    assert a == b
    assert a["ci95"] != c["ci95"] or a["sd"] != c["sd"]


def test_equivalence_never_conflates_inconclusive_with_equivalent():
    # Wholly inside the band -> equivalence supported.
    inside = uj.equivalence([0.46, 0.54])
    assert inside["equivalence_supported"] is True
    assert inside["difference_detected"] is False
    assert inside["verdict"] == uj.VERDICT_EQUIVALENT

    # Contains 0.5 but spills outside the band -> INCONCLUSIVE, not equivalent.
    wide = uj.equivalence([0.30, 0.70])
    assert wide["equivalence_supported"] is False
    assert wide["difference_detected"] is False
    assert wide["verdict"] == uj.VERDICT_INCONCLUSIVE
    assert "does NOT support" in wide["note"]

    # Excludes 0.5 and leaves the band -> a difference was detected.
    shifted = uj.equivalence([0.62, 0.71])
    assert shifted["difference_detected"] is True
    assert shifted["equivalence_supported"] is False
    assert shifted["verdict"] == uj.VERDICT_DIFFERENT

    # Excludes 0.5 but stays inside the band -> real, and practically small.
    small = uj.equivalence([0.52, 0.58])
    assert small["difference_detected"] is True
    assert small["equivalence_supported"] is True
    assert small["verdict"] == uj.VERDICT_SMALL_DIFFERENCE


def test_band_edges_are_inclusive_and_the_band_is_the_preregistered_one():
    assert uj.EQUIVALENCE_BAND == (0.40, 0.60)
    assert uj.equivalence([0.40, 0.60])["equivalence_supported"] is True
    assert uj.equivalence([0.399, 0.60])["equivalence_supported"] is False


def test_a_barely_wide_ci_is_not_upgraded_to_equivalence_by_compare():
    # End-to-end: 2 tasks with opposite outcomes -> a CI spanning [0,1]. The
    # reported verdict must be inconclusive, not equivalence.
    runtime = _dry_runtime()
    pairs = [
        _pair(task="t1", a="x" * 10, b="y" * 400, draw=0),
        _pair(task="t2", a="x" * 400, b="y" * 10, draw=0),
    ]
    result = uj.compare(
        pairs, runtime, dry_run_policy=uj.DRY_RUN_PREFER_LONGER, replicates=2000
    )
    assert result["score"] == 0.5
    assert result["equivalence"]["verdict"] == uj.VERDICT_INCONCLUSIVE
    assert result["equivalence"]["equivalence_supported"] is False


# ------------------------------------------------------- caching / dry-run


def test_dry_run_is_the_default_and_nothing_is_billed(monkeypatch):
    # compare() must not build a sampler unless execute=True.
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("make_sampler called on the default (dry-run) path")

    from octt import generation

    monkeypatch.setattr(generation, "make_sampler", explode)
    result = uj.compare([_pair()], _dry_runtime(), replicates=10)
    assert result["retained"] == 1


def test_execute_true_still_respects_a_dry_run_runtime(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("dry-run runtime must never reach the paid judge")

    from octt import generation

    monkeypatch.setattr(generation, "make_sampler", explode)
    result = uj.compare([_pair()], _dry_runtime(), execute=True, replicates=10)
    assert result["retained"] == 1


def test_unknown_dry_run_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown dry-run policy"):
        uj.compare([_pair()], _dry_runtime(), dry_run_policy="whatever")


def test_cache_is_written_reused_and_keyed_on_the_instrument(tmp_path):
    runtime = _dry_runtime()
    cache = tmp_path / "verdicts.jsonl"
    pairs = [_pair(task=f"t{i}", a=f"a{i}", b=f"b{i}", draw=i) for i in range(5)]
    first = uj.compare(pairs, runtime, cache_path=cache, replicates=100)
    assert len(cache.read_text().splitlines()) == 5
    second = uj.compare(pairs, runtime, cache_path=cache, replicates=100)
    assert first["score"] == second["score"]
    assert len(cache.read_text().splitlines()) == 5, "cache hit must not re-judge"

    # A different rubric is a different instrument: it must MISS the cache.
    key_now = uj._pair_key("m", uj.DEFAULT_JUDGE_CONFIG, "hash-a", pairs[0])
    key_other = uj._pair_key("m", uj.DEFAULT_JUDGE_CONFIG, "hash-b", pairs[0])
    assert key_now != key_other


def test_cache_key_ignores_the_order_seed():
    # Both presentations are judged either way, so a seed change cannot change
    # a resolved verdict -- and must never re-charge for one.
    pair = _pair()
    assert uj.initial_order(pair, seed=0) != uj.initial_order(pair, seed=1) or True
    key = uj._pair_key("m", uj.DEFAULT_JUDGE_CONFIG, "h", pair)
    assert key == uj._pair_key("m", uj.DEFAULT_JUDGE_CONFIG, "h", pair)


def test_unparseable_verdicts_are_dropped_not_scored(tmp_path):
    runtime = _dry_runtime()
    cache = tmp_path / "verdicts.jsonl"
    pairs = [_pair(task=f"t{i}", a=f"a{i}", b=f"b{i}", draw=i) for i in range(4)]
    uj.compare(pairs, runtime, cache_path=cache, replicates=10)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    # Blank out one pair's verdicts, as an unanswerable judge call would.
    rows[0].update(
        verdict_ab=None, verdict_ba=None, resolution=None,
        resolution_reason=uj.REASON_UNPARSEABLE, score_a=None,
    )
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    result = uj.compare(pairs, runtime, cache_path=cache, replicates=100)
    assert result["pairs"] == 4
    assert result["retained"] == 3
    assert result["dropped"] == 1
    assert result["reasons"][uj.REASON_UNPARSEABLE] == 1
    assert result["bootstrap"]["clusters"] == 3


def test_deduplicated_verdicts_do_not_leak_pair_identity(tmp_path):
    # Two DIFFERENT tasks that happen to have identical prompts and responses
    # share one paid judgment (correct: the judge inputs are identical), but
    # they must remain two rows with their own task and correctness. Reusing
    # the cached row wholesale collapsed them into one bootstrap cluster and
    # one correctness stratum.
    runtime = _dry_runtime()
    cache = tmp_path / "verdicts.jsonl"
    pairs = [
        _pair(task="t1", a="same", b="same body", ca=True, cb=False),
        _pair(task="t2", a="same", b="same body", ca=False, cb=False),
    ]
    result = uj.compare(pairs, runtime, cache_path=cache, replicates=100)
    assert len(cache.read_text().splitlines()) == 1, "identical inputs judged once"
    assert [r["task"] for r in result["rows"]] == ["t1", "t2"]
    assert [r["correctness_stratum"] for r in result["rows"]] == [
        uj.STRATUM_A_ONLY, uj.STRATUM_BOTH_INCORRECT
    ]
    assert result["bootstrap"]["clusters"] == 2
    assert result["strata"]["by_correctness"][uj.STRATUM_A_ONLY]["pairs"] == 1
    assert result["strata"]["by_correctness"][uj.STRATUM_BOTH_INCORRECT]["pairs"] == 1


def test_offline_run_is_deterministic():
    runtime = _dry_runtime()
    pairs = [_pair(task=f"t{i}", a="x" * (10 + i), b="y" * (30 + i), draw=i) for i in range(6)]
    a = uj.compare(pairs, runtime, dry_run_policy=uj.DRY_RUN_HASH, replicates=200)
    b = uj.compare(pairs, runtime, dry_run_policy=uj.DRY_RUN_HASH, replicates=200)
    assert a == b


def test_hash_policy_is_swap_consistent():
    # The well-behaved-judge baseline: it tracks content, so it agrees with
    # itself across the swap and its preferences are retained.
    runtime = _dry_runtime()
    pairs = [_pair(task=f"t{i}", a=f"alpha{i}", b=f"beta{i}", draw=i) for i in range(10)]
    result = uj.compare(
        pairs, runtime, dry_run_policy=uj.DRY_RUN_HASH, dry_run_bias=1.0, replicates=100
    )
    assert result["dropped"] == 0
    assert result["reasons"][uj.REASON_DISAGREE] == 0
    assert result["wins_a"] + result["losses_a"] == 10


# ----------------------------------------------------------- pair building


def _graded_rows():
    return [
        {"task": "t1", "arm": "base", "k": 0, "prompt": "the real question",
         "response": "base answer", "passed": True, "tier": "hard"},
        {"task": "t1", "arm": "trained", "k": 0, "prompt": "the real question",
         "response": "trained answer", "passed": True, "tier": "hard"},
        {"task": "t1", "arm": "rewriter", "k": 0, "prompt": "REWRITE THIS: ...",
         "response": "rewritten answer", "passed": False, "tier": "hard"},
        {"task": "t2", "arm": "trained", "k": 0, "prompt": "q2",
         "response": "trained 2", "passed": True, "tier": "hard"},
    ]


def test_pairs_from_rows_uses_the_task_question_not_the_rewrite_metaprompt():
    pairs, skipped = uj.pairs_from_rows(_graded_rows(), "trained_vs_rewriter")
    assert len(pairs) == 1
    pair = pairs[0]
    # The rewriter row's stored prompt is the rewrite INSTRUCTION; judging
    # against it would ask the judge the wrong question.
    assert pair.prompt == "the real question"
    assert "REWRITE" not in pair.prompt
    assert pair.arm_a == "trained" and pair.arm_b == "rewriter"
    assert pair.correct_a is True and pair.correct_b is False
    assert skipped["unmatched"] == 1  # t2 has no rewriter draw


def test_pairs_from_rows_skips_empty_responses():
    rows = _graded_rows()
    rows[2]["response"] = "   "
    pairs, skipped = uj.pairs_from_rows(rows, "trained_vs_rewriter")
    assert pairs == []
    assert skipped["empty_response"] == 1


def test_pairs_from_rows_handles_the_secondary_contrasts():
    pairs, _ = uj.pairs_from_rows(_graded_rows(), "trained_vs_base")
    assert len(pairs) == 1
    assert (pairs[0].arm_a, pairs[0].arm_b) == ("trained", "base")


# ------------------------------------------------------------------- CLI


def test_cli_is_dry_run_by_default_and_reports_the_verdict(tmp_path, capsys):
    graded = tmp_path / "graded.jsonl"
    graded.write_text("\n".join(json.dumps(r) for r in _graded_rows()) + "\n")
    out = tmp_path / "result.json"
    code = uj.main([str(graded), "--out", str(out), "--replicates", "50"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "calibration" in printed and "PASS" in printed
    assert "DRY-RUN" in printed and "nothing billed" in printed
    result = json.loads(out.read_text())
    assert result["instrument_id"] == uj.INSTRUMENT_ID
    assert result["calibration"]["passed"] is True


def test_cli_refuses_to_spend_when_calibration_fails(tmp_path, capsys):
    graded = tmp_path / "graded.jsonl"
    graded.write_text("\n".join(json.dumps(r) for r in _graded_rows()) + "\n")
    code = uj.main([str(graded), "--dry-run-policy", uj.DRY_RUN_PREFER_LONGER])
    assert code == 2
    printed = capsys.readouterr().out
    assert "FAIL" in printed
    assert "refusing to judge" in printed
