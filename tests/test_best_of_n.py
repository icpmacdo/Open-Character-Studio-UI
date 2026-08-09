"""Tests for the Best-of-N runner, its held-out panel, and its gate (B14).

Offline and deterministic: no API keys, no network, no training stack.

Three properties carry the experiment and each has a test that fails loudly if
it stops holding:

  1. **Nested bank.** Every reported N is a PREFIX of one 16-candidate bank. If
     a future edit resampled per N, ``test_every_n_is_a_prefix_*`` and
     ``test_walking_the_ladder_costs_no_extra_judge_calls`` fail.
  2. **Held out.** The validation panel is disjoint from the DPO training
     prompts, the Phase 2 task set, the W2 qualitative panel, the frozen Phase 3
     test panel, and the frozen KL audit bank. Overlap is fatal, and an
     *uncheckable* corpus is fatal too.
  3. **The gate never reads the proxy.** Its no-go thresholds are predeclared
     constants and its pass condition requires independent evidence the proxy
     cannot supply.
"""

from __future__ import annotations

import json

import pytest

from octt import best_of_n as bon
from octt import (
    instruments,
    models,
    phase3_artifacts,
    preference,
    qualitative,
    tinker_client,
)

#: PIN. The panel is an instrument: every candidate request_id hashes it, so an
#: edit invalidates banked cells. Never update this constant in place — mint
#: phase3-bon-validation-v2.
PANEL_HASH = "c34e68d77e632a49af838ce24a71f2d1a37cea61e9bee243e577f67957585e94"

#: The corpora that exist today. All four, since the Phase 3 test panel was
#: frozen (``data/qualitative_panels/phase3-test-v1.json``,
#: ``tests/test_phase3_test_panel.py``) — so the default
#: ``require=REQUIRED_BEFORE_SPEND`` is now fully satisfiable.
AVAILABLE_CORPORA = bon.RESERVED_CORPORA


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL,),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


@pytest.fixture(scope="module")
def reserved():
    return bon.collect_reserved_corpora()


@pytest.fixture(scope="module")
def ladder():
    runtime = _dry_runtime()
    policies = (bon.BASE_POLICY,)
    banks = bon.generate_banks(bon.VALIDATION_PANEL, policies, runtime)
    result = bon.run_ladder(
        banks, bon.VALIDATION_PANEL, runtime, policies=policies,
        dry_run_policy=preference.DRY_RUN_PREFER_LONGER,
    )
    return banks, result


# ------------------------------------------------------------------- panel


def test_panel_shape_is_the_readiness_doc_shape():
    panel = bon.VALIDATION_PANEL
    assert len(panel.prompts) == 16
    counts = {c: sum(1 for p in panel.prompts if p.category == c) for c in qualitative.CATEGORIES}
    assert counts == {
        "trait_open": 8,
        "technical": 4,
        "non_english": 2,
        "instruction_conflict": 2,
    }
    assert dict(panel.quotas) == counts
    panel.validate()


def test_panel_hash_is_pinned_and_ids_are_unique():
    ids = [p.prompt_id for p in bon.VALIDATION_PANEL.prompts]
    assert len(set(ids)) == 16
    assert bon.VALIDATION_PANEL.content_hash == PANEL_HASH, (
        "the Phase 3 validation panel changed. Its hash is baked into every "
        "candidate request_id: mint phase3-bon-validation-v2 rather than "
        "editing v1 and updating this constant."
    )
    assert bon.PANEL_HASH == PANEL_HASH


def test_non_english_cells_span_two_non_latin_scripts():
    rows = [p for p in bon.VALIDATION_PANEL.prompts if p.category == "non_english"]
    assert sorted(p.language for p in rows) == ["ja", "ru"]
    assert {bon.dominant_script(p.text) for p in rows} == {"japanese", "cyrillic"}


def test_no_panel_prompt_hands_the_persona_its_register():
    """A maritime topic would supply the voice the panel is meant to measure."""
    banned = ("pirate", "ship", "sail", "sea", "ocean", "captain", "crew of", "harbour")
    for prompt in bon.VALIDATION_PANEL.prompts:
        low = prompt.text.lower()
        assert not any(word in low for word in banned), prompt.prompt_id


def test_instruction_conflict_cells_carry_machine_checkable_rules():
    rows = [
        p for p in bon.VALIDATION_PANEL.prompts if p.category == "instruction_conflict"
    ]
    assert len(rows) == 2
    rules = {p.prompt_id: bon.format_rule_for(p) for p in rows}
    assert rules == {"bo-ic-json-01": "json_only", "bo-ic-es-brief-01": "max_words:25"}
    # The language-conflict cell asks in English for a Spanish answer.
    lang_cell = next(p for p in rows if p.prompt_id == "bo-ic-es-brief-01")
    assert lang_cell.language == "en"
    assert bon.target_language_for(lang_cell) == "es"


def test_technical_correctness_coverage_is_explicit():
    technical = [p for p in bon.VALIDATION_PANEL.prompts if p.category == "technical"]
    graded = [p for p in technical if bon.correctness_rule_for(p) is not None]
    assert len(technical) == 4
    assert {p.prompt_id for p in graded} == {"bo-tech-units-01", "bo-tech-phone-01"}
    assert bon.correctness_rule_for(graded[0]) is not None


def test_generation_instrument_is_registered_and_carries_no_prompt_text():
    entry = instruments.get(bon.GENERATION_INSTRUMENT_ID)
    assert entry.prompts == {}, "panel text lives in the hashed panel, not the registry"
    assert entry.sampling["temperature"] == bon.GEN_TEMPERATURE
    assert entry.sampling["max_tokens"] == bon.GEN_MAX_TOKENS
    assert entry.sampling["candidates_per_cell"] == bon.CANDIDATES_PER_CELL


# ------------------------------------------------------------- held-out design


def test_the_panel_is_disjoint_from_every_available_reserved_corpus(reserved):
    assert bon.find_overlaps(bon.VALIDATION_PANEL, reserved) == []
    report = bon.assert_panel_disjoint(reserved=reserved, require=AVAILABLE_CORPORA)
    assert report["disjoint"] is True
    assert report["panel_hash"] == PANEL_HASH
    assert set(report["checked"]) == set(AVAILABLE_CORPORA)


def test_every_reserved_corpus_actually_loaded_something(reserved):
    assert reserved.unavailable == ()
    assert len(reserved.texts[bon.CORPUS_DPO]) > 1000  # constitution pools + LIMA
    assert len(reserved.texts[bon.CORPUS_PHASE2]) >= 50  # codeval ceiling + hard + qual
    assert len(reserved.texts[bon.CORPUS_W2]) == 25
    assert len(reserved.texts[bon.CORPUS_PHASE3_TEST]) == 32  # phase3-test-v1
    assert len(reserved.texts[bon.CORPUS_KL_AUDIT]) == 64  # kl-audit-64x2-v1


def test_an_exact_overlap_is_fatal(reserved):
    stolen = reserved.texts[bon.CORPUS_W2][0]
    panel = qualitative.Panel(
        panel_id="leaky", version="v1", quotas={"trait_open": 1},
        prompts=(qualitative.PanelPrompt("x", stolen, "en", "trait_open"),),
    )
    hits = bon.find_overlaps(panel, reserved)
    assert [h["kind"] for h in hits] == ["exact"]
    assert hits[0]["corpus"] == bon.CORPUS_W2
    with pytest.raises(bon.PanelOverlapError, match="NOT held out"):
        bon.assert_panel_disjoint(panel, reserved, require=AVAILABLE_CORPORA)


def test_a_reworded_training_prompt_is_still_caught(reserved):
    """Exact-match-only disjointness is defeated by a comma; this is not."""
    stolen = reserved.texts[bon.CORPUS_W2][0]
    reworded = stolen.replace("I've", "I have").rstrip("?") + ", roughly?"
    assert bon.normalize_prompt(reworded) != bon.normalize_prompt(stolen)
    panel = qualitative.Panel(
        panel_id="leaky", version="v1", quotas={"trait_open": 1},
        prompts=(qualitative.PanelPrompt("x", reworded, "en", "trait_open"),),
    )
    hits = bon.find_overlaps(panel, reserved)
    assert hits and hits[0]["kind"] == "near_duplicate"
    assert hits[0]["similarity"] >= bon.NEAR_DUPLICATE_JACCARD
    with pytest.raises(bon.PanelOverlapError):
        bon.assert_panel_disjoint(panel, reserved, require=AVAILABLE_CORPORA)


def test_an_unavailable_required_corpus_is_fatal_not_a_pass(reserved):
    """'We could not check it' must never read as 'we checked it and it was clean'.

    Every reserved corpus is loadable today, so the property is exercised by
    withholding one: a ``ReservedCorpora`` that lost the Phase 3 test panel must
    make ``assert_panel_disjoint`` raise rather than quietly check the rest.
    Written this way on purpose — asserting a *particular* corpus is missing
    would make this test a hostage to the next corpus that gets frozen.
    """
    assert bon.REQUIRED_BEFORE_SPEND == bon.RESERVED_CORPORA
    for corpus in bon.RESERVED_CORPORA:
        crippled = bon.ReservedCorpora(
            texts={k: v for k, v in reserved.texts.items() if k != corpus},
            unavailable=(corpus,),
            detail={corpus: "UNAVAILABLE: FileNotFoundError: withheld by the test"},
        )
        with pytest.raises(bon.ReservedCorpusUnavailable, match=corpus):
            bon.assert_panel_disjoint(reserved=crippled)  # default require = all of them


def test_the_phase3_test_panel_is_frozen_so_the_gate_can_pass_honestly(reserved):
    """The refusal that blocked this audit is resolved by a real corpus.

    ``PENDING_CORPORA`` documented that the Phase 3 test panel did not exist
    yet. It does now, so the default ``require`` is satisfiable and the check
    passes because it was *run*, not because it was skipped.
    """
    assert reserved.unavailable == ()
    report = bon.assert_panel_disjoint(reserved=reserved)
    assert report["disjoint"] is True
    assert list(report["available"]) == list(bon.RESERVED_CORPORA)
    assert "UNAVAILABLE" not in report["detail"][bon.CORPUS_PHASE3_TEST]


def test_a_frozen_phase3_test_panel_is_picked_up_automatically(tmp_path, monkeypatch):
    """The check is wired now so freezing the test set cannot skip it."""
    panels = tmp_path / "data" / "qualitative_panels"
    panels.mkdir(parents=True)
    (panels / "w2-pirate-v1.json").write_text(
        json.dumps(
            qualitative.Panel(
                "w2-pirate-v1", "v1", {"trait_open": 1},
                (qualitative.PanelPrompt("w", "unrelated w2 text", "en", "trait_open"),),
            ).to_dict()
        )
    )
    leak = bon.VALIDATION_PANEL.prompts[0].text
    (panels / "phase3-test-v1.json").write_text(
        json.dumps(
            qualitative.Panel(
                "phase3-test-v1", "v1", {"trait_open": 1},
                (qualitative.PanelPrompt("t", leak, "en", "trait_open"),),
            ).to_dict()
        )
    )
    (panels / "kl-audit-64x2-v1.json").write_text(
        json.dumps(
            qualitative.Panel(
                "kl-audit-64x2-v1", "v1", {"trait_open": 1},
                (qualitative.PanelPrompt("k", "unrelated kl-audit text", "en",
                                         "trait_open"),),
            ).to_dict()
        )
    )
    monkeypatch.setattr(bon, "_dpo_training_prompts", lambda root, offline: ([], "stub"))
    monkeypatch.setattr(bon, "_phase2_task_prompts", lambda root: ([], "stub"))
    found = bon.collect_reserved_corpora(tmp_path)
    assert found.unavailable == ()
    with pytest.raises(bon.PanelOverlapError, match=bon.CORPUS_PHASE3_TEST):
        bon.assert_panel_disjoint(reserved=found)


# ------------------------------------------------------- nested candidate bank


def test_one_bank_of_sixteen_per_cell(ladder):
    banks, _ = ladder
    assert len(banks) == 16  # 16 prompts x 1 policy
    assert {len(b.candidates) for b in banks} == {bon.CANDIDATES_PER_CELL}
    assert len({b.cell_id for b in banks}) == 16


def test_a_bank_must_hold_exactly_sixteen_candidates():
    with pytest.raises(ValueError, match="exactly 16"):
        bon.CandidateBank("p", "pol", "m", "base", ("only", "two"))


def test_every_n_is_a_prefix_of_the_same_bank(ladder):
    banks, _ = ladder
    bank = banks[0]
    for n in bon.N_LADDER:
        assert bank.prefix(n) == bank.candidates[:n]
    # Nesting: each rung contains the previous one, unchanged.
    for smaller, larger in zip(bon.N_LADDER, bon.N_LADDER[1:], strict=False):
        assert bank.prefix(larger)[:smaller] == bank.prefix(smaller)
    with pytest.raises(ValueError, match="not on the reported ladder"):
        bank.prefix(5)


def test_the_n_ladder_is_the_declared_one():
    assert bon.N_LADDER == (1, 2, 4, 8, 16)
    assert max(bon.N_LADDER) == bon.CANDIDATES_PER_CELL


def test_pairs_at_a_smaller_n_are_a_subset_of_the_n16_pairs(ladder):
    banks, _ = ladder
    prompt = bon.VALIDATION_PANEL.prompts[0]
    big = {
        (p.response_a, p.response_b) for p in bon.build_pairs(banks[0], prompt, 16)
    }
    small = {
        (p.response_a, p.response_b) for p in bon.build_pairs(banks[0], prompt, 8)
    }
    assert small < big


def test_all_240_ordered_comparisons_at_n16():
    assert bon.ORDERED_COMPARISONS_AT_MAX == 240
    assert bon.UNORDERED_PAIRS_AT_MAX == 120
    ordered = bon.ordered_pairs(16)
    assert len(ordered) == 240
    assert len(set(ordered)) == 240
    assert all(i != j for i, j in ordered)
    # Every unordered pair appears in both presentations.
    for i, j in bon.unordered_pairs(16):
        assert (i, j) in set(ordered) and (j, i) in set(ordered)
    assert len(bon.unordered_pairs(16)) == 120


def test_walking_the_ladder_costs_no_extra_judge_calls(ladder):
    """The N=16 verdicts already contain every smaller rung's comparisons."""
    banks, result = ladder
    assert len(result.verdict_rows) == len(banks) * bon.UNORDERED_PAIRS_AT_MAX
    assert len(result.selections) == len(banks) * len(bon.N_LADDER)


# ---------------------------------------------------------------- selection


def _rows_from(winner_by_pair):
    return {
        pair: {"resolution": res, "resolution_reason": "swap_agreement"}
        for pair, res in winner_by_pair.items()
    }


def _bank(n=16):
    return bon.CandidateBank("p1", "pol", "m", "base", tuple(f"c{i}" for i in range(n)))


def test_ties_break_on_the_lowest_index():
    bank = _bank()
    all_ties = _rows_from(
        {pair: preference.RESOLUTION_TIE for pair in bon.unordered_pairs(16)}
    )
    selection = bon.select(bank, all_ties, 16)
    assert selection.selected_index == 0
    assert selection.selected_candidate_id == bank.candidate_id(0)
    assert selection.tie_break_rule == bon.TIE_BREAK_RULE == "lowest-candidate-index"
    assert set(selection.scores) == {7.5}  # 15 opponents x 0.5


def test_the_highest_scorer_wins_and_the_tournament_is_logged():
    bank = _bank()
    # Candidate 5 beats everyone; everything else ties.
    rows = {}
    for i, j in bon.unordered_pairs(16):
        if i == 5:
            rows[(i, j)] = {"resolution": "a", "resolution_reason": "swap_agreement"}
        elif j == 5:
            rows[(i, j)] = {"resolution": "b", "resolution_reason": "swap_agreement"}
        else:
            rows[(i, j)] = {"resolution": "tie", "resolution_reason": "both_orders_tie"}
    selection = bon.select(bank, rows, 16)
    assert selection.selected_index == 5
    assert selection.wins[5] == 15
    assert selection.losses[5] == 0
    assert selection.scores[5] == 15.0
    assert selection.proxy_score == 1.0
    assert len(selection.comparisons) == 120
    assert selection.dropped_pairs == 0


def test_n1_selects_the_only_candidate_with_no_comparisons():
    selection = bon.select(_bank(), {}, 1)
    assert selection.selected_index == 0
    assert selection.comparisons == ()
    assert selection.proxy_score == 0.5


def test_dropped_pairs_are_counted_not_hidden():
    bank = _bank()
    rows = {
        pair: {"resolution": None, "resolution_reason": "unparseable"}
        for pair in bon.unordered_pairs(4)
    }
    selection = bon.select(bank, rows, 4)
    assert selection.dropped_pairs == 6
    assert selection.ties == (0, 0, 0, 0)  # a drop is NOT a measured tie
    assert selection.selected_index == 0


def test_a_length_biased_proxy_selects_the_longest_candidate(ladder):
    """The stub bank grows with index, so this is a visible, testable failure."""
    _, result = ladder
    at_16 = [s for s in result.selections if s.n == 16]
    assert all(s.selected_index == 15 for s in at_16)
    at_1 = [s for s in result.selections if s.n == 1]
    assert all(s.selected_index == 0 for s in at_1)


def test_selection_is_deterministic(ladder):
    banks, result = ladder
    runtime = _dry_runtime()
    again = bon.run_ladder(
        banks, bon.VALIDATION_PANEL, runtime, dry_run_policy=preference.DRY_RUN_PREFER_LONGER
    )
    assert [s.selected_candidate_id for s in again.selections] == [
        s.selected_candidate_id for s in result.selections
    ]


# --------------------------------------------------------- independent measures


def test_measures_are_independent_of_the_prompted_judge():
    prompt = next(
        p for p in bon.VALIDATION_PANEL.prompts if p.prompt_id == "bo-ic-json-01"
    )
    good = bon.candidate_measures('{"measures": "distance", "unit": "km"}', prompt)
    bad = bon.candidate_measures("Sure! Here you go: distance, in km.", prompt)
    assert good["format_compliant"] is True
    assert bad["format_compliant"] is False
    assert good["marker_hit"] is False


def test_marker_and_repetition_measures_use_the_pinned_instruments():
    prompt = bon.VALIDATION_PANEL.prompts[0]
    m = bon.candidate_measures("Ahoy there, matey. Ahoy there, matey.", prompt)
    assert m["marker_count"] == 4  # ahoy, matey, twice
    assert m["marker_hit"] is True
    assert m["marker_density_per_100w"] > 0
    assert m["marker_instrument"].startswith("pirate-strong-v1")
    assert bon.repetition_score("a b c d e a b c d e") > 0
    assert bon.repetition_score("all of these words are entirely distinct here") == 0.0


def test_language_match_catches_the_wrong_script():
    ja = next(p for p in bon.VALIDATION_PANEL.prompts if p.prompt_id == "bo-nl-ja-01")
    assert bon.candidate_measures("議事録は要点だけに絞ります。", ja)["language_match"] is True
    assert bon.candidate_measures("Keep it to the decisions only.", ja)["language_match"] is False
    es = next(p for p in bon.VALIDATION_PANEL.prompts if p.prompt_id == "bo-ic-es-brief-01")
    assert bon.candidate_measures(
        "El disyuntor corta la corriente para que el circuito no se sobrecargue.", es
    )["language_match"] is True


def test_truncation_is_measured():
    prompt = bon.VALIDATION_PANEL.prompts[0]
    assert bon.candidate_measures("A complete thought.", prompt)["truncated"] is False
    assert bon.candidate_measures("A thought that stops mid", prompt)["truncated"] is True


# --------------------------------------------------------------------- gate


def _per_n(**over):
    base = {
        "proxy_score": 0.5,
        "marker_density_per_100w": 1.0,
        "repetition_score": 0.1,
        "technical_correctness": 1.0,
        "technical_correctness_n": 2,
        "format_compliance": 1.0,
        "language_match": 1.0,
        "truncation_rate": 0.0,
    }
    base.update(over)
    return base


def _ladder_of(first, last, mid=None):
    return {1: first, 2: first, 4: first, 8: mid or last, 16: last}


def _independent(gain=0.2):
    return (
        {"bridge_passed": True, "per_n": {1: 0.0, 16: gain}},
        {"per_n": {1: 0.0, 16: gain}, "n_reviewed": 20},
    )


def test_gate_thresholds_are_predeclared_constants():
    assert bon.NOGO_MARKER_DENSITY_RATIO == 2.0
    assert bon.NOGO_REPETITION_RATIO == 2.0
    assert bon.TECHNICAL_DECLINE_TOLERANCE == 0.05
    assert bon.PROXY_SATURATION_EPS == 0.02
    assert bon.REQUIRED_INDEPENDENT_INPUTS == ("evaluator_v2", "gold_slice")


def test_gate_reads_nothing_from_the_optimization_proxy():
    v2, gold = _independent()
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(proxy_score=0.9)), evaluator_v2=v2, gold_slice=gold
    )
    assert gate["optimization_proxy"] == preference.INSTRUMENT_ID
    assert gate["independent_of_proxy"] is True
    assert gate["verdict"] == bon.GATE_PROCEED


def test_marker_density_doubling_is_a_no_go():
    v2, gold = _independent()
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(marker_density_per_100w=2.0, proxy_score=0.9)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["verdict"] == bon.GATE_STOP
    assert "marker_density_doubles" in gate["no_go_fired"]


def test_repetition_doubling_is_a_no_go():
    v2, gold = _independent()
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(repetition_score=0.2, proxy_score=0.9)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["verdict"] == bon.GATE_STOP
    assert "repetition_doubles" in gate["no_go_fired"]


def test_technical_decline_is_a_no_go():
    v2, gold = _independent()
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(technical_correctness=0.9, proxy_score=0.9)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["verdict"] == bon.GATE_STOP
    assert "technical_correctness_declines" in gate["no_go_fired"]
    assert gate["measured"]["technical_correctness_delta"] == pytest.approx(-0.1)


def test_proxy_saturating_by_n8_without_independent_gain_is_a_no_go():
    # Proxy flat from N=8 to N=16, independent measure flat too.
    v2, gold = _independent(gain=0.0)
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(proxy_score=0.9), mid=_per_n(proxy_score=0.9)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["verdict"] == bon.GATE_STOP
    assert "proxy_saturates_without_independent_gain" in gate["no_go_fired"]


def test_saturation_is_not_a_no_go_when_independent_quality_did_improve():
    v2, gold = _independent(gain=0.3)
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(proxy_score=0.9), mid=_per_n(proxy_score=0.9)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["no_go_fired"] == []
    assert gate["verdict"] == bon.GATE_PROCEED


def test_missing_independent_evidence_is_incomplete_never_a_pass():
    per_n = _ladder_of(_per_n(), _per_n(proxy_score=0.9))
    gate = bon.evaluate_gate(per_n)
    assert gate["verdict"] == bon.GATE_INCOMPLETE
    assert gate["missing_inputs"] == ["evaluator_v2", "gold_slice"]
    # A v2 bridge that did NOT pass is not usable evidence either.
    gate = bon.evaluate_gate(
        per_n,
        evaluator_v2={"bridge_passed": False, "per_n": {1: 0.0, 16: 0.5}},
        gold_slice={"per_n": {1: 0.0, 16: 0.5}},
    )
    assert gate["verdict"] == bon.GATE_INCOMPLETE
    assert gate["missing_inputs"] == ["evaluator_v2"]


def test_a_fired_no_go_beats_missing_evidence():
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(marker_density_per_100w=5.0, proxy_score=0.9))
    )
    assert gate["verdict"] == bon.GATE_STOP
    assert gate["missing_inputs"]  # still missing, but the no-go is decisive


def test_no_independent_improvement_stops_even_with_a_clean_no_go_sheet():
    v2, gold = _independent(gain=0.0)
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(proxy_score=0.9), mid=_per_n(proxy_score=0.4)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["no_go_fired"] == []
    assert gate["verdict"] == bon.GATE_STOP
    assert "did not improve the independent character measure" in gate["reason"]


def test_guardrail_loss_stops_even_when_character_improved():
    v2, gold = _independent(gain=0.3)
    gate = bon.evaluate_gate(
        _ladder_of(_per_n(), _per_n(proxy_score=0.9, format_compliance=0.8)),
        evaluator_v2=v2, gold_slice=gold,
    )
    assert gate["verdict"] == bon.GATE_STOP
    assert "format_compliance" in gate["reason"]


def test_gate_needs_both_ends_of_the_ladder():
    with pytest.raises(ValueError, match="needs both"):
        bon.evaluate_gate({1: _per_n()})


def test_the_offline_ladder_trips_the_gate_as_designed(ladder):
    _, result = ladder
    gate = bon.evaluate_gate(result.per_n)
    assert gate["verdict"] == bon.GATE_STOP
    assert "marker_density_doubles" in gate["no_go_fired"]
    assert "repetition_doubles" in gate["no_go_fired"]


def test_gold_slice_plan_is_deterministic_and_stratified(ladder):
    _, result = ladder
    by_id = {p.prompt_id: p for p in bon.VALIDATION_PANEL.prompts}
    plan = bon.gold_slice_plan(result.selections, by_id)
    assert plan == bon.gold_slice_plan(result.selections, by_id)
    # One item per (category, policy, N).
    assert len(plan) == len(qualitative.CATEGORIES) * 1 * len(bon.N_LADDER)
    assert {row["n"] for row in plan} == set(bon.N_LADDER)
    assert {row["category"] for row in plan} == set(qualitative.CATEGORIES)


# --------------------------------------------------------------- artifacts


def test_the_bundle_logs_every_required_row(ladder, tmp_path):
    banks, result = ladder
    gate = bon.evaluate_gate(result.per_n)
    header = bon.write_phase3_bundle(
        tmp_path / "bundle", result, banks, gate=gate, disjointness={"disjoint": True}
    )
    assert header["counts"] == {
        "candidate": 16 * 16,
        "comparison": 16 * 240,
        "swap": 16 * 120,
        "selection": 16 * 5,
    }
    assert header["nested_candidate_bank"] is True
    assert header["gate"]["verdict"] == bon.GATE_STOP
    bundle = phase3_artifacts.read_bundle(tmp_path / "bundle")
    assert {k: len(v) for k, v in bundle.items()} == header["counts"]


def test_candidate_rows_carry_measures_reward_components_and_provenance(ladder, tmp_path):
    banks, result = ladder
    rows = bon.build_phase3_rows(result, banks)
    candidate = rows[phase3_artifacts.ROW_CANDIDATE][0]
    phase3_artifacts.validate_row(candidate)
    for measure in phase3_artifacts.REQUIRED_MEASURES:
        assert measure in candidate["measures"]
    # Reward components exist only for the N rungs the candidate is IN — the
    # nested design showing up in the log.
    assert sorted(candidate["reward_components"]) == ["1", "16", "2", "4", "8"]
    last = rows[phase3_artifacts.ROW_CANDIDATE][15]
    assert sorted(last["reward_components"]) == ["16"]
    assert last["reward_components"]["16"]["selected"] is True
    assert candidate["execution_mode"] == "dry-run"
    assert candidate["dry_run_recipe_version"] == bon.DRY_RUN_RECIPE_VERSION
    assert candidate["instrument_id"] == bon.GENERATION_INSTRUMENT_ID
    assert candidate["panel_hash"] == PANEL_HASH


def test_comparison_rows_keep_both_orderings_separately(ladder):
    banks, result = ladder
    rows = bon.build_phase3_rows(result, banks)
    comparisons = rows[phase3_artifacts.ROW_COMPARISON]
    assert len(comparisons) == 2 * len(rows[phase3_artifacts.ROW_SWAP])
    first_pair = comparisons[0]["pair_id"]
    both = [c for c in comparisons if c["pair_id"] == first_pair]
    assert {c["presentation"] for c in both} == {"ab", "ba"}
    assert both[0]["left_candidate_id"] == both[1]["right_candidate_id"]
    assert len({c["request_id"] for c in comparisons}) == len(comparisons)


def test_swap_rows_record_the_resolution_and_the_reason(ladder):
    banks, result = ladder
    swaps = bon.build_phase3_rows(result, banks)[phase3_artifacts.ROW_SWAP]
    for row in swaps[:20]:
        phase3_artifacts.validate_row(row)
        assert row["resolution_reason"] in (
            preference.REASON_AGREE, preference.REASON_BOTH_TIE,
            preference.REASON_DISAGREE, preference.REASON_UNPARSEABLE,
        )
        assert {"len_a", "len_b", "length_ratio", "score_a"} <= set(row)


def test_a_dropped_swap_may_not_carry_a_score():
    row = {
        "row_type": phase3_artifacts.ROW_SWAP,
        "phase3_schema_version": phase3_artifacts.PHASE3_SCHEMA_VERSION,
        "request_id": "r", "instrument_id": "i", "instrument_hash": "h",
        "execution_mode": "dry-run", "cell_id": "c", "prompt_id": "p", "pair_id": "x",
        "candidate_a": "a", "candidate_b": "b", "index_a": 0, "index_b": 1,
        "verdict_ab": None, "verdict_ba": None, "presentation_order": ["ab", "ba"],
        "resolution": None, "resolution_reason": "unparseable", "swap_consistent": False,
        "score_a": 0.5, "len_a": 1, "len_b": 1, "length_ratio": 1.0,
        "character_brief_id": "b", "character_brief_hash": "bh",
    }
    with pytest.raises(phase3_artifacts.Phase3SchemaError, match="missing data"):
        phase3_artifacts.validate_row(row)
    row["score_a"] = None
    phase3_artifacts.validate_row(row)


def test_missing_measures_are_rejected_at_construction():
    with pytest.raises(phase3_artifacts.Phase3SchemaError, match="missing independent measures"):
        phase3_artifacts.candidate_row(
            panel_id="p", panel_hash="h", prompt_id="q", prompt_text="t", category="c",
            policy_id="pol", model_id="m", checkpoint_role="base",
            checkpoint_fingerprint="base", candidate_index=0, candidate_id="cid",
            response="hello", measures={"length_chars": 5}, instrument_id="i",
            instrument_hash="ih", renderer="r", sampling={}, execution_mode="dry-run",
        )


def test_an_incomplete_bundle_is_fatal(ladder):
    banks, result = ladder
    rows = bon.build_phase3_rows(result, banks)
    rows[phase3_artifacts.ROW_SELECTION].pop()
    with pytest.raises(phase3_artifacts.Phase3Incomplete, match="incomplete"):
        phase3_artifacts.assert_bundle_complete(
            rows, cells=len(banks), candidates_per_cell=16,
            ordered_comparisons_per_cell=240, n_ladder=bon.N_LADDER,
        )


def test_request_ids_are_deterministic_across_runs(ladder):
    banks, result = ladder
    first = bon.build_phase3_rows(result, banks)
    second = bon.build_phase3_rows(result, banks)
    for row_type in phase3_artifacts.ROW_TYPES:
        assert [r["request_id"] for r in first[row_type]] == [
            r["request_id"] for r in second[row_type]
        ]


# ------------------------------------------------------------ cost projection


def test_the_two_halves_are_priced_separately():
    policies = (bon.BASE_POLICY, bon.acquisition_policy("tinker://run/sampler_weights/final"))
    projection = bon.dry_run_projection(policies=policies)
    # The readiness doc's exact figures.
    assert projection["candidate_generations"] == 512
    assert projection["judge_calls"] == 7680
    assert projection["nested"] is True
    assert projection["n_ladder"] == [1, 2, 4, 8, 16]
    # Separately, not blended: two distinct line groups with their own totals.
    assert len(projection["generation_lines"]) == 2
    assert projection["judge_line"]["stage"] == "bon.ordered_judge_calls"
    assert projection["generation_usd"] > 0
    assert projection["judge_usd"] > 0
    assert projection["total_usd"] == pytest.approx(
        projection["generation_usd"] + projection["judge_usd"]
    )
    # Judging is the dominant cost; a blended total would hide that.
    assert projection["judge_usd"] > 10 * projection["generation_usd"]


def test_judge_calls_are_short_and_generations_are_long():
    projection = bon.dry_run_projection()
    assert projection["judge_line"]["max_tokens_per_call"] == 32
    assert projection["generation_lines"][0]["max_tokens_per_call"] == 512
    assert projection["judge_line"]["ordered_comparisons_per_cell"] == 240
    assert projection["judge_line"]["unordered_pairs_per_cell"] == 120


def test_projection_scales_quadratically_in_judging_and_linearly_in_generation():
    one = bon.dry_run_projection(policies=(bon.BASE_POLICY,))
    two = bon.dry_run_projection(
        policies=(bon.BASE_POLICY, bon.acquisition_policy("tinker://x/sampler_weights/0"))
    )
    assert two["candidate_generations"] == 2 * one["candidate_generations"]
    assert two["judge_calls"] == 2 * one["judge_calls"]
    # Per cell, judging is 240 calls against 16 generations: 15x, quadratic in N.
    assert (
        one["judge_calls"] / one["candidate_generations"]
        == bon.ORDERED_COMPARISONS_AT_MAX / bon.CANDIDATES_PER_CELL
    )


def test_projection_is_stamped_and_renders():
    projection = bon.dry_run_projection(
        policies=(bon.BASE_POLICY, bon.acquisition_policy("tinker://x/sampler_weights/0"))
    )
    assert projection["panel_hash"] == PANEL_HASH
    assert projection["judge_instrument_id"] == preference.INSTRUMENT_ID
    assert projection["generation_instrument_id"] == bon.GENERATION_INSTRUMENT_ID
    text = bon.format_projection(projection)
    assert "candidate generations" in text and "ordered judge calls" in text
    assert "7,680" in text and "512" in text


def test_an_acquisition_policy_needs_a_real_checkpoint():
    with pytest.raises(ValueError, match="tinker://"):
        bon.acquisition_policy("some/local/path")
    policy = bon.acquisition_policy("tinker://run/sampler_weights/final")
    assert policy.checkpoint_role == "trained"
    assert policy.model_path == "tinker://run/sampler_weights/final"
