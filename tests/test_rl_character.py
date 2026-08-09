"""B17 guards: the shared RL runner, its telemetry, and its hard stops.

Offline and deterministic: no API keys, no network, no training stack.

The four stock-recipe gaps this module exists to close are each asserted to be
closed, and — more importantly — every hard stop is exercised against a run
built to breach exactly it. A guardrail that only ever reads PASS is decoration:

  - capability falls more than 5 points, or breaches the Phase 2 margin;
  - independent character/coherence declines at two successive checkpoints;
  - median response length drifts more than 25%;
  - marker density or repetition reaches twice baseline;
  - response-sum KL crosses 2 x K_DPO;
  - reward-provider validity falls below 99%.

The selection rule gets the same treatment: a history whose proxy reward keeps
climbing past the independent peak must select the independent peak, or the
whole point of running an out-of-loop measure is lost.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from octt import instruments, models, on_policy_character, preference, tinker_client
from octt import rl_character as rl

REPO = pathlib.Path(__file__).resolve().parents[1]

GROUP = ("aa", "bbbb", "cccccc", "dddddddd")
PROMPT = "Explain how suspension bridges carry their deck load."


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL,),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


# --------------------------------------------------------------- fixtures


BASELINE = rl.Baseline(
    capability_score=70.0,
    median_response_chars=400.0,
    marker_density_per_100w=5.0,
    repetition_score=0.10,
    phase2_margin_floor=66.0,
)

KDPO = rl.KDPOIndex(
    k_dpo_nats=8.0,
    mean_token_nats=0.05,
    max_response_sum_nats=20.0,
    num_responses=128,
    num_prompts=64,
    rollouts_per_prompt=2,
    audit_bank_id="test-bank",
    audit_bank_hash="deadbeef",
    checkpoint_fingerprint="tinker://dpo/final",
    reference_fingerprint="base:Qwen/Qwen3.5-4B",
)


def _eval(step: int, **over) -> rl.CheckpointEval:
    """A checkpoint evaluation that breaches nothing, plus overrides."""
    defaults = {
        "step": step,
        "proxy_reward": 0.5,
        "character_score": 0.50,
        "coherence_score": 0.50,
        "capability_score": 70.0,
        "format_compliance": 1.0,
        "language_match": 1.0,
        "median_response_chars": 400.0,
        "marker_density_per_100w": 5.0,
        "repetition_score": 0.10,
        "reference_kl_response_sum_nats": 1.0,
        "reference_kl_mean_token_nats": 0.01,
        "kl_policy_base": -0.02,
        "checkpoint_uri": f"tinker://run/sampler_weights/{step}",
        "checkpoint_fingerprint": f"tinker://run/sampler_weights/{step}",
        "optimizer_state_uri": f"tinker://run/state/{step}",
        "provider_id": rl.PROVIDER_PROMPTED_JUDGE,
        "instrument_id": preference.INSTRUMENT_ID,
        "instrument_hash": "abc123",
        "validity": rl.ValidityLedger(decisive=6),
        "num_responses": 32,
    }
    defaults.update(over)
    return rl.CheckpointEval(**defaults)


def _monitor(**over) -> rl.RunMonitor:
    return rl.RunMonitor(BASELINE, kdpo=KDPO, **over)


# ---------------------------------------------------- gap 1: G=4 only, always


def test_pilot_config_is_the_readiness_table():
    cfg = rl.RL_PILOT
    assert cfg.policy_model == "Qwen/Qwen3.5-4B"
    assert cfg.lora_rank == 32
    assert cfg.learning_rate == 1e-5
    assert cfg.group_size == 4
    assert cfg.prompts_per_batch == 8
    assert cfg.temperature == 1.0
    assert cfg.max_response_tokens == 512
    assert cfg.max_steps == 50
    assert cfg.save_every == 5
    assert cfg.eval_every == 5
    assert cfg.samples_per_step == 32


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6, 8, 16])
def test_group_size_other_than_four_is_rejected(size):
    with pytest.raises(rl.RLConfigError, match="group_size must be exactly 4"):
        rl.RLConfig(group_size=size)


def test_the_refusal_names_the_vendored_chunking_behavior():
    with pytest.raises(rl.RLConfigError) as exc:
        rl.RLConfig(group_size=8)
    message = str(exc.value)
    assert "get_pairs_chunked" in message
    assert "contiguous fours" in message


def test_chunked_tournament_semantics_are_rejected_by_name():
    with pytest.raises(rl.RLConfigError, match="unsupported tournament semantics"):
        rl.RLConfig(tournament=rl.TOURNAMENT_CHUNKED_FOURS)


def test_unknown_tournament_semantics_are_rejected():
    with pytest.raises(rl.RLConfigError, match="unsupported tournament"):
        rl.RLConfig(tournament="round-robin-swiss")


def test_unknown_reward_provider_is_rejected():
    with pytest.raises(rl.RLConfigError, match="unknown reward_provider"):
        rl.RLConfig(reward_provider="vibes")


def test_both_reward_providers_are_accepted():
    for provider in rl.REWARD_PROVIDERS:
        assert rl.RLConfig(reward_provider=provider).reward_provider == provider


def test_the_g4_tournament_is_complete_and_both_directions():
    assert rl.PAIRS_PER_GROUP == 6
    assert rl.ORDERED_MATCHUPS_PER_GROUP == 12
    assert len(rl.unordered_pairs(4)) == 6
    assert rl.unordered_pairs(4)[0] == (0, 1)
    assert rl.unordered_pairs(4)[-1] == (2, 3)


def test_providers_refuse_a_group_that_is_not_four():
    provider = rl.LabelPreferenceReward(label_fn=lambda p, a, b: "A")
    with pytest.raises(rl.RLConfigError, match="only well-defined at G=4"):
        provider.score_group(PROMPT, ["a", "b"])


# ------------------------------------- gap 2: the frozen reference is required


def test_frozen_reference_is_required_even_at_zero_kl_penalty():
    assert rl.RL_PILOT.kl_penalty_coefficient == 0.0
    with pytest.raises(rl.MissingReferenceError, match="EVEN WHEN"):
        rl.RLConfig(kl_penalty_coefficient=0.0, reference=None)


def test_frozen_reference_is_required_with_a_penalty_too():
    with pytest.raises(rl.MissingReferenceError):
        rl.RLConfig(kl_penalty_coefficient=0.1, reference=None)


def test_the_default_reference_is_the_unmodified_base():
    ref = rl.RL_PILOT.require_reference()
    assert ref.model_id == rl.POLICY_MODEL
    assert ref.checkpoint_uri is None
    assert ref.fingerprint == "base:Qwen/Qwen3.5-4B"


def test_reference_is_stamped_into_the_config_record():
    record = rl.RL_PILOT.to_dict()
    assert record["reference"]["fingerprint"] == "base:Qwen/Qwen3.5-4B"
    assert record["recipe_version"] == rl.RL_RECIPE_VERSION


# ------------------------------------------- gap 3: k3 beside the signed k1


def test_the_k3_estimator_is_imported_not_reimplemented():
    assert rl.k3_per_token is on_policy_character.k3_per_token
    assert rl.kl_k3 is on_policy_character.kl_k3
    source = (REPO / "octt" / "rl_character.py").read_text(encoding="utf-8")
    assert "exp(logr)" not in source, "k3 must not be re-derived here"


def test_reference_k3_is_nonnegative_where_signed_k1_is_not():
    # The policy is MORE likely than the reference on every token, so the signed
    # k1 difference is positive; flip it and k1 goes negative. k3 stays >= 0.
    policy = [[-0.1, -0.2, -0.3]]
    reference = [[-0.6, -0.7, -0.8]]
    k3 = rl.reference_kl(policy, reference)
    assert k3.mean_token_kl_nats >= 0
    assert k3.mean_response_sum_kl_nats >= 0
    forward = rl.kl_policy_base_k1(policy, reference)["kl_policy_base"]
    backward = rl.kl_policy_base_k1(reference, policy)["kl_policy_base"]
    assert forward > 0 > backward
    assert rl.reference_kl(reference, policy).mean_token_kl_nats >= 0


def test_kl_policy_base_matches_the_cookbook_definition():
    policy = [[-1.0, -2.0], [-3.0, -4.0]]
    reference = [[-1.5, -2.5], [-2.0, -5.0]]
    expected = ((0.5 + 0.5) + (-1.0 + 1.0)) / 4
    out = rl.kl_policy_base_k1(policy, reference)
    assert out["kl_policy_base"] == pytest.approx(expected)
    # The cookbook key is preserved verbatim so banked runs stay comparable.
    assert out["rl/kl_policy_base_k1_signed_nats"] == out["kl_policy_base"]


def test_kl_metrics_are_rl_prefixed_and_cannot_be_read_as_opd():
    telemetry = rl.reference_kl([[-0.1, -0.2]], [[-0.4, -0.5]])
    keys = rl.kl_metrics(telemetry)
    assert "rl/reference_k3_mean_token_nats" in keys
    assert "rl/reference_k3_response_sum_nats" in keys
    assert not any(k.startswith("opd/") for k in keys)


def test_response_sum_kl_is_per_response():
    sums = rl.response_sum_kl([[-0.1, -0.2], [-0.3]], [[-0.4, -0.5], [-0.9]])
    assert len(sums) == 2
    assert all(s >= 0 for s in sums)


@pytest.mark.parametrize(
    "policy,reference",
    [
        ([[-0.1]], [[-0.1], [-0.2]]),
        ([[-0.1, -0.2]], [[-0.1]]),
    ],
)
def test_mismatched_logprob_shapes_are_refused(policy, reference):
    with pytest.raises(rl.RLShapeError):
        rl.reference_kl(policy, reference)
    with pytest.raises(rl.RLShapeError):
        rl.kl_policy_base_k1(policy, reference)


# ---------------------- gap 4: invalid labels are not ties, and they abort


def test_true_tie_and_invalid_are_counted_separately():
    ties = rl.LabelPreferenceReward(label_fn=lambda p, a, b: "TIE")
    garbage = rl.LabelPreferenceReward(label_fn=lambda p, a, b: "I think A is nicer")
    tie_group = ties.score_group(PROMPT, GROUP)
    bad_group = garbage.score_group(PROMPT, GROUP)

    assert tie_group.ledger.true_tie == 6
    assert tie_group.ledger.invalid == 0
    assert tie_group.ledger.validity_rate == 1.0

    assert bad_group.ledger.invalid == 6
    assert bad_group.ledger.true_tie == 0
    assert bad_group.ledger.validity_rate == 0.0

    # The stock path returns the tie value for both, so the REWARD is identical.
    # That is exactly why the ledger, not the reward, is what the stop reads.
    assert tie_group.rewards == bad_group.rewards == (0.0, 0.0, 0.0, 0.0)


def test_unparseable_judge_rows_are_invalid_not_ties():
    row = {"resolution": None, "resolution_reason": preference.REASON_UNPARSEABLE}
    assert rl._outcome_from_row(row) == rl.OUTCOME_INVALID
    tie = {"resolution": preference.RESOLUTION_TIE,
           "resolution_reason": preference.REASON_BOTH_TIE}
    assert rl._outcome_from_row(tie) == rl.OUTCOME_TIE


def test_swap_disagreement_is_its_own_outcome():
    row = {"resolution": None, "resolution_reason": preference.REASON_DISAGREE}
    assert rl._outcome_from_row(row) == rl.OUTCOME_INCONSISTENT
    ledger = rl.ValidityLedger.from_outcomes([rl.OUTCOME_INCONSISTENT] * 4)
    # Position bias is a measurement, so validity stays 1.0 and the separate
    # swap-consistency rate is what falls.
    assert ledger.validity_rate == 1.0
    assert ledger.swap_consistency_rate == 0.0


def test_score_batch_aborts_when_validity_falls_below_the_floor():
    calls = {"n": 0}

    def flaky(prompt, a, b):
        calls["n"] += 1
        return None if calls["n"] % 20 == 0 else "A"

    provider = rl.LabelPreferenceReward(label_fn=flaky)
    groups = [(f"prompt {i}", GROUP) for i in range(8)]
    with pytest.raises(rl.RewardValidityError) as exc:
        rl.score_batch(provider, groups)
    assert exc.value.floor == rl.VALIDITY_FLOOR
    assert exc.value.ledger.invalid > 0
    assert "not a tie" in str(exc.value)


def test_score_batch_passes_a_clean_provider():
    provider = rl.LabelPreferenceReward(label_fn=lambda p, a, b: "A" if a < b else "B")
    scored, ledger = rl.score_batch(provider, [(PROMPT, GROUP)])
    assert ledger.validity_rate == 1.0
    assert len(scored) == 1
    assert scored[0].rewards[0] == pytest.approx(1.0)


def test_assert_validity_floor_is_ninety_nine_percent():
    assert rl.VALIDITY_FLOOR == 0.99
    rl.assert_validity(rl.ValidityLedger(decisive=99, invalid=1))
    with pytest.raises(rl.RewardValidityError):
        rl.assert_validity(rl.ValidityLedger(decisive=98, invalid=2))


# ------------------------------------------------------- reward providers


def test_prompted_judge_reward_tracks_the_judge():
    provider = rl.PromptedJudgeReward(
        runtime=_dry_runtime(), dry_run_policy=preference.DRY_RUN_PREFER_LONGER
    )
    group = provider.score_group(PROMPT, GROUP)
    assert group.provider_id == rl.PROVIDER_PROMPTED_JUDGE
    assert group.instrument_id == preference.INSTRUMENT_ID
    assert group.instrument_hash == instruments.get(preference.INSTRUMENT_ID).content_hash
    # prefer-longer ranks the group by length; rewards are win-minus-loss/matchups.
    assert group.rewards == (-1.0, pytest.approx(-1 / 3), pytest.approx(1 / 3), 1.0)
    assert group.ledger.invalid == 0
    assert group.ledger.total == 6


def test_a_position_tracking_judge_shows_up_as_swap_inconsistent():
    provider = rl.PromptedJudgeReward(
        runtime=_dry_runtime(), dry_run_policy=preference.DRY_RUN_POSITION_A
    )
    group = provider.score_group(PROMPT, GROUP)
    assert group.ledger.swap_inconsistent == 6
    assert group.ledger.invalid == 0
    assert group.rewards == (0.0, 0.0, 0.0, 0.0)


def test_trained_pm_reward_scores_both_orientations():
    class Pointwise:
        pointwise = True

        def score(self, prompt, response, *, position=rl.POSITION_A):
            return float(len(response))

    group = rl.TrainedPMReward(model=Pointwise()).score_group(PROMPT, GROUP)
    assert group.provider_id == rl.PROVIDER_TRAINED_PM
    assert group.ledger.invalid == 0
    assert group.ledger.swap_inconsistent == 0
    assert group.rewards[-1] > group.rewards[0]


def test_trained_pm_position_bias_is_caught_by_the_swap():
    class Positional:
        pointwise = False

        def score(self, prompt, response, *, position=rl.POSITION_A):
            return 10.0 if position == rl.POSITION_A else 0.0

    group = rl.TrainedPMReward(model=Positional()).score_group(PROMPT, GROUP)
    assert group.ledger.swap_inconsistent == 6
    assert group.rewards == (0.0, 0.0, 0.0, 0.0)


def test_trained_pm_nan_reward_is_invalid_not_a_tie():
    class Broken:
        pointwise = True

        def score(self, prompt, response, *, position=rl.POSITION_A):
            return float("nan")

    group = rl.TrainedPMReward(model=Broken()).score_group(PROMPT, GROUP)
    assert group.ledger.invalid == 6
    assert group.ledger.true_tie == 0
    with pytest.raises(rl.RewardValidityError):
        rl.assert_validity(group.ledger)


def test_position_constants_match_the_reward_model_module():
    from octt import reward_model

    assert rl.POSITION_A == reward_model.POSITION_A
    assert rl.POSITION_B == reward_model.POSITION_B
    assert rl.REQUIRED_GROUP_SIZE == reward_model.REQUIRED_GROUP_SIZE


def test_advantages_are_group_centered():
    assert rl.centered_advantages([1.0, 0.0, -1.0, 0.0]) == [1.0, 0.0, -1.0, 0.0]
    assert sum(rl.centered_advantages([3.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)


# -------------------------------------------------------- K_DPO indexing


def _bank(n: int = rl.AUDIT_BANK_PROMPTS) -> rl.AuditBank:
    return rl.AuditBank(bank_id="kl-audit-test", prompts=tuple(f"probe {i}" for i in range(n)))


def test_audit_bank_is_fixed_at_sixty_four_by_two():
    bank = _bank()
    assert bank.num_responses == 128
    assert rl.AUDIT_BANK_PROMPTS == 64
    assert rl.AUDIT_BANK_ROLLOUTS == 2
    with pytest.raises(rl.RLConfigError, match="fixed at 64 prompts"):
        _bank(32)
    with pytest.raises(rl.RLConfigError, match="rollouts per prompt"):
        rl.AuditBank(bank_id="x", prompts=_bank().prompts, rollouts_per_prompt=1)


def test_audit_bank_rejects_duplicate_prompts():
    with pytest.raises(rl.RLConfigError, match="duplicate"):
        rl.AuditBank(bank_id="x", prompts=("same",) * rl.AUDIT_BANK_PROMPTS)


def test_measure_k_dpo_requires_the_whole_bank():
    bank = _bank()
    short_policy = [[-0.1, -0.2]] * 10
    short_ref = [[-0.5, -0.6]] * 10
    with pytest.raises(rl.RLConfigError, match="partial bank"):
        rl.measure_k_dpo(
            bank,
            short_policy,
            short_ref,
            checkpoint_fingerprint="tinker://dpo/final",
            reference=rl.DEFAULT_REFERENCE,
        )


def test_k_dpo_is_the_mean_response_sum_and_indexes_the_multiples():
    bank = _bank()
    policy = [[-0.1, -0.2]] * bank.num_responses
    reference = [[-0.5, -0.6]] * bank.num_responses
    index = rl.measure_k_dpo(
        bank,
        policy,
        reference,
        checkpoint_fingerprint="tinker://dpo/final",
        reference=rl.DEFAULT_REFERENCE,
    )
    assert index.num_responses == 128
    assert index.k_dpo_nats > 0
    assert index.audit_bank_hash == bank.content_hash
    thresholds = index.thresholds()
    assert set(thresholds) == {"0.25x", "0.5x", "1x", "2x"}
    assert thresholds["1x"] == pytest.approx(index.k_dpo_nats)
    assert thresholds["2x"] == pytest.approx(2 * index.k_dpo_nats)
    assert index.stop_threshold_nats == pytest.approx(2 * index.k_dpo_nats)
    # No universal threshold is baked in anywhere: every number is K_DPO-relative.
    assert index.to_dict()["instrument"]["instrument_id"] == rl.KL_AUDIT_INSTRUMENT_ID


def test_first_crossings_report_none_when_the_run_never_got_there():
    observations = [(5, 1.0), (10, 3.0), (15, 9.0)]
    crossings = rl.first_crossings(observations, KDPO)  # K_DPO = 8.0
    assert crossings["0.25x"] == 10  # 2.0 nats first reached at step 10
    assert crossings["0.5x"] == 15
    assert crossings["1x"] == 15
    assert crossings["2x"] is None


def test_first_crossings_are_order_insensitive():
    shuffled = [(15, 9.0), (5, 1.0), (10, 3.0)]
    assert rl.first_crossings(shuffled, KDPO) == rl.first_crossings(
        sorted(shuffled), KDPO
    )


# --------------------------------------------------- the logging contract


def test_every_readiness_doc_logging_item_is_in_the_row():
    row = _eval(5).to_row()
    rl.validate_eval_row(row)
    for field in (
        "proxy_reward",
        "character_score",
        "capability_score",
        "format_compliance",
        "language_match",
        "median_response_chars",
        "marker_density_per_100w",
        "repetition_score",
        "reference_kl_response_sum_nats",
        "kl_policy_base",
        "checkpoint_uri",
        "checkpoint_fingerprint",
        "optimizer_state_uri",
    ):
        assert field in row


def test_a_row_missing_a_required_field_is_rejected():
    row = _eval(5).to_row()
    del row["marker_density_per_100w"]
    with pytest.raises(rl.EvalRowError, match="marker_density_per_100w"):
        rl.validate_eval_row(row)


def test_a_checkpoint_without_a_uri_cannot_be_recorded():
    with pytest.raises(rl.EvalRowError, match="cannot be selected"):
        rl.validate_eval_row(_eval(5, checkpoint_uri="  ").to_row())


def test_metrics_view_is_float_only_and_prefixed():
    metrics = _eval(5).to_metrics()
    assert all(k.startswith("rl/") for k in metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert metrics["rl/reward_validity_rate"] == 1.0


def test_the_proxy_is_not_an_independent_measure():
    with pytest.raises(ValueError, match="never"):
        _eval(5).independent("proxy_reward")


# ------------------------------------------------- hard stops, one by one


def test_a_clean_run_fires_no_stop():
    monitor = _monitor()
    for step in (5, 10, 15):
        assert monitor.record(_eval(step)) == []
    assert not monitor.halted


def test_capability_drop_over_five_points_halts():
    monitor = _monitor()
    monitor.record(_eval(5))
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(10, capability_score=64.0))  # -6 points
    names = [b.name for b in exc.value.breaches]
    assert rl.STOP_CAPABILITY in names
    assert monitor.halted_at_step == 10


def test_a_four_point_drop_does_not_halt():
    monitor = _monitor()
    assert monitor.record(_eval(5, capability_score=66.5)) == []


def test_breaching_the_phase_two_margin_halts_even_within_five_points():
    monitor = _monitor()
    # -4.5 points: inside the 5-point stop, but under the Phase 2 floor of 66.
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(5, capability_score=65.5))
    names = [b.name for b in exc.value.breaches]
    assert rl.STOP_PHASE2_MARGIN in names
    assert rl.STOP_CAPABILITY not in names


def test_two_successive_independent_declines_halt():
    monitor = _monitor()
    monitor.record(_eval(5, character_score=0.60))
    monitor.record(_eval(10, character_score=0.55))  # one decline: not yet
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(15, character_score=0.50))
    breach = next(b for b in exc.value.breaches if b.name == rl.STOP_INDEPENDENT_DECLINE)
    assert breach.value == 2.0
    assert "character_score" in breach.detail


def test_one_decline_then_a_recovery_does_not_halt():
    monitor = _monitor()
    monitor.record(_eval(5, character_score=0.60))
    monitor.record(_eval(10, character_score=0.55))
    assert monitor.record(_eval(15, character_score=0.58)) == []


def test_coherence_declining_twice_also_halts():
    monitor = _monitor()
    monitor.record(_eval(5, coherence_score=0.70))
    monitor.record(_eval(10, coherence_score=0.65))
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(15, coherence_score=0.60))
    breach = next(b for b in exc.value.breaches if b.name == rl.STOP_INDEPENDENT_DECLINE)
    assert "coherence_score" in breach.detail


@pytest.mark.parametrize("chars", [501.0, 299.0])
def test_median_length_drift_over_twenty_five_percent_halts(chars):
    monitor = _monitor()
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(5, median_response_chars=chars))
    assert rl.STOP_LENGTH_DRIFT in [b.name for b in exc.value.breaches]


def test_length_drift_inside_the_band_does_not_halt():
    monitor = _monitor()
    assert monitor.record(_eval(5, median_response_chars=490.0)) == []


def test_marker_density_at_twice_baseline_halts():
    monitor = _monitor()
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(5, marker_density_per_100w=10.0))  # 2x of 5.0
    breach = next(b for b in exc.value.breaches if b.name == rl.STOP_MARKER_DENSITY)
    assert breach.value == pytest.approx(2.0)


def test_repetition_at_twice_baseline_halts():
    monitor = _monitor()
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(5, repetition_score=0.20))  # 2x of 0.10
    assert rl.STOP_REPETITION in [b.name for b in exc.value.breaches]


def test_marker_and_repetition_below_double_do_not_halt():
    monitor = _monitor()
    assert monitor.record(
        _eval(5, marker_density_per_100w=9.9, repetition_score=0.19)
    ) == []


def test_response_sum_kl_crossing_two_k_dpo_halts():
    monitor = _monitor()
    monitor.record(_eval(5, reference_kl_response_sum_nats=7.0))  # under 2x8=16
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(10, reference_kl_response_sum_nats=16.5))
    breach = next(b for b in exc.value.breaches if b.name == rl.STOP_KL)
    assert breach.threshold == pytest.approx(16.0)
    assert "K_DPO" in breach.detail


def test_the_kl_stop_is_k_dpo_relative_not_absolute():
    small = rl.KDPOIndex(**{**vars(KDPO), "k_dpo_nats": 1.0})
    monitor = rl.RunMonitor(BASELINE, kdpo=small)
    with pytest.raises(rl.RunHalted):
        monitor.record(_eval(5, reference_kl_response_sum_nats=2.5))
    # The same 2.5 nats is fine against a larger K_DPO — nothing universal here.
    assert _monitor().record(_eval(5, reference_kl_response_sum_nats=2.5)) == []


def test_reward_validity_below_ninety_nine_percent_halts():
    monitor = _monitor()
    with pytest.raises(rl.RunHalted) as exc:
        monitor.record(_eval(5, validity=rl.ValidityLedger(decisive=95, invalid=5)))
    breach = next(b for b in exc.value.breaches if b.name == rl.STOP_VALIDITY)
    assert breach.value == pytest.approx(0.95)
    assert "not ties" in breach.detail


def test_true_ties_alone_never_trip_the_validity_stop():
    monitor = _monitor()
    assert monitor.record(_eval(5, validity=rl.ValidityLedger(true_tie=100))) == []


def test_every_declared_stop_has_a_constant():
    assert set(rl.STOPS) == {
        rl.STOP_CAPABILITY,
        rl.STOP_PHASE2_MARGIN,
        rl.STOP_INDEPENDENT_DECLINE,
        rl.STOP_LENGTH_DRIFT,
        rl.STOP_MARKER_DENSITY,
        rl.STOP_REPETITION,
        rl.STOP_KL,
        rl.STOP_VALIDITY,
    }


def test_several_breaches_are_all_reported_not_short_circuited():
    breaches = rl.check_guardrails(
        [_eval(5, capability_score=60.0, repetition_score=0.30,
               reference_kl_response_sum_nats=99.0)],
        BASELINE,
        kdpo=KDPO,
    )
    names = {b.name for b in breaches}
    assert {rl.STOP_CAPABILITY, rl.STOP_PHASE2_MARGIN, rl.STOP_REPETITION,
            rl.STOP_KL} <= names


# ------------------------------------------------------------- the monitor


def test_the_monitor_writes_one_validated_row_per_interval(tmp_path):
    monitor = _monitor(out_dir=tmp_path)
    monitor.record(_eval(5))
    monitor.record(_eval(10))
    rows = [
        json.loads(line)
        for line in (tmp_path / "rl_metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == [5, 10]
    for row in rows:
        rl.validate_eval_row(row)
        assert row["validity"]["invalid"] == 0
        assert row["recipe_version"] == rl.RL_RECIPE_VERSION


def test_the_monitor_reports_kl_crossings():
    monitor = _monitor()
    monitor.record(_eval(5, reference_kl_response_sum_nats=1.0))
    monitor.record(_eval(10, reference_kl_response_sum_nats=4.0))
    crossings = monitor.kl_crossings()
    assert crossings["0.25x"] == 10
    assert crossings["1x"] is None


def test_the_monitor_record_is_serializable(tmp_path):
    monitor = _monitor(out_dir=tmp_path)
    monitor.record(_eval(5))
    payload = json.loads(json.dumps(monitor.to_dict()))
    assert payload["guardrails"]["validity_floor"] == rl.VALIDITY_FLOOR
    assert payload["kdpo"]["k_dpo_response_sum_nats"] == 8.0


# ------------------------------------------------------------- selection


def test_selection_takes_the_independent_peak_not_the_proxy_peak():
    history = [
        _eval(5, character_score=0.50, proxy_reward=0.10),
        _eval(10, character_score=0.70, proxy_reward=0.40),  # independent peak
        _eval(15, character_score=0.62, proxy_reward=0.80),  # proxy still climbing
    ]
    selection = rl.select_checkpoint(history)
    assert selection.selected_step == 10
    assert selection.proxy_peak_step == 15
    assert selection.differs_from_proxy_peak
    assert selection.independent_score == 0.70
    assert "never continued proxy-reward improvement" in selection.rule


def test_selection_excludes_the_breaching_checkpoint_and_everything_after():
    history = [_eval(5, character_score=0.50), _eval(10, character_score=0.90)]
    breach = rl.GuardrailBreach(rl.STOP_KL, 10, 20.0, 16.0, "over 2x K_DPO")
    selection = rl.select_checkpoint(history, [breach])
    assert selection.selected_step == 5
    assert selection.halted_at_step == 10
    assert selection.eligible_steps == (5,)


def test_selection_ties_keep_the_earlier_step():
    history = [_eval(5, character_score=0.60), _eval(10, character_score=0.60)]
    assert rl.select_checkpoint(history).selected_step == 5


def test_selection_can_use_coherence_as_the_measure():
    history = [_eval(5, coherence_score=0.4), _eval(10, coherence_score=0.9)]
    selection = rl.select_checkpoint(history, measure=rl.MEASURE_COHERENCE)
    assert selection.selected_step == 10
    assert selection.measure == rl.MEASURE_COHERENCE


def test_selection_refuses_the_proxy_as_a_measure():
    with pytest.raises(ValueError, match="never"):
        rl.select_checkpoint([_eval(5)], measure="proxy_reward")


def test_selection_raises_when_the_first_checkpoint_already_breached():
    breach = rl.GuardrailBreach(rl.STOP_VALIDITY, 5, 0.5, 0.99, "garbage")
    with pytest.raises(rl.NoEligibleCheckpoint, match="nothing safe"):
        rl.select_checkpoint([_eval(5)], [breach])


def test_selection_raises_on_an_empty_history():
    with pytest.raises(rl.NoEligibleCheckpoint):
        rl.select_checkpoint([])


def test_monitor_selects_from_what_preceded_the_halt():
    monitor = _monitor()
    monitor.record(_eval(5, character_score=0.5))
    monitor.record(_eval(10, character_score=0.8))
    with pytest.raises(rl.RunHalted):
        monitor.record(_eval(15, character_score=0.9, repetition_score=0.5))
    selection = monitor.select()
    assert selection.selected_step == 10
    assert selection.halted_at_step == 15
    assert [b.name for b in selection.breaches] == [rl.STOP_REPETITION]


# ------------------------------------------- the held-out test set, used once


def _selection() -> rl.SelectionResult:
    return rl.select_checkpoint([_eval(5), _eval(10, character_score=0.9)])


def test_the_held_out_test_set_can_be_opened_exactly_once(tmp_path):
    held_out = rl.HeldOutTestSet(run_dir=tmp_path)
    assert not held_out.used
    record = held_out.use(_selection(), purpose="final Elo vs DPO")
    assert record["selected_step"] == 10
    assert held_out.used
    with pytest.raises(rl.TestSetAlreadyUsed, match="no second look"):
        held_out.use(_selection(), purpose="just one more look")


def test_the_used_once_guard_survives_a_new_process(tmp_path):
    rl.HeldOutTestSet(run_dir=tmp_path).use(_selection(), purpose="final")
    fresh = rl.HeldOutTestSet(run_dir=tmp_path)  # a new object, as a rerun would be
    assert fresh.used
    assert fresh.record()["purpose"] == "final"
    with pytest.raises(rl.TestSetAlreadyUsed):
        fresh.use(_selection(), purpose="rerun")


def test_the_test_set_cannot_be_opened_before_a_selection_exists(tmp_path):
    held_out = rl.HeldOutTestSet(run_dir=tmp_path)
    with pytest.raises(rl.SelectionRequired, match="AFTER"):
        held_out.use(None, purpose="peek")
    assert not held_out.used  # a refused use must not spend the one look


def test_the_sentinel_names_the_selection_it_was_spent_on(tmp_path):
    selection = _selection()
    rl.HeldOutTestSet(run_dir=tmp_path).use(selection, purpose="final")
    payload = json.loads((tmp_path / rl.TEST_SET_SENTINEL).read_text())
    assert payload["selection_hash"] == selection.selection_hash
    assert payload["selection"]["rule"] == rl.SELECTION_RULE


# --------------------------------------------------------- plan and run


def test_the_plan_is_free_and_shaped_like_the_pilot():
    rl_plan = rl.plan(rl.RL_PILOT, num_prompts=400)
    assert rl_plan.steps == 50
    assert rl_plan.samples == 50 * 32
    assert rl_plan.sample_tokens == 50 * 32 * 512
    assert rl_plan.checkpoints == 10
    assert rl_plan.evaluations == 10
    # 8 groups/step x 12 ordered matchups.
    assert rl_plan.judge_calls == 50 * 8 * 12


def test_the_trained_pm_plan_makes_no_judge_calls():
    rl_plan = rl.plan(rl.RLConfig(reward_provider=rl.PROVIDER_TRAINED_PM), num_prompts=400)
    assert rl_plan.judge_calls == 0
    assert rl_plan.judge_tokens == 0


def test_run_is_dry_run_by_default(tmp_path):
    payload = rl.run(["a", "b", "c"], tmp_path, _dry_runtime())
    assert payload["status"] == "dry-run"
    assert payload["execution_mode"] == rl.EXECUTION_MODE_DRY_RUN
    assert (tmp_path / "rl_plan.json").exists()
    written = json.loads((tmp_path / "rl_plan.json").read_text())
    assert written["reference"]["fingerprint"] == "base:Qwen/Qwen3.5-4B"
    assert written["selection_rule"] == rl.SELECTION_RULE


def test_execute_against_a_dry_run_runtime_still_spends_nothing(tmp_path):
    payload = rl.run(["a", "b"], tmp_path, _dry_runtime(), execute=True)
    assert payload["status"] == "dry-run"


def test_a_paid_run_is_refused_without_provider_baseline_and_kdpo(tmp_path):
    runtime = SimpleNamespace(config=SimpleNamespace(dry_run=False))
    with pytest.raises(rl.RLConfigError, match="reward provider"):
        rl.run(["a"], tmp_path, runtime, execute=True)
    provider = rl.LabelPreferenceReward(label_fn=lambda p, a, b: "A")
    with pytest.raises(rl.RLConfigError, match="Baseline"):
        rl.run(["a"], tmp_path, runtime, execute=True, reward_provider=provider)
    with pytest.raises(rl.RLConfigError, match="K_DPO"):
        rl.run(
            ["a"], tmp_path, runtime, execute=True,
            reward_provider=provider, baseline=BASELINE,
        )


def test_response_measures_reuse_the_pinned_marker_instrument():
    from octt import persona_markers

    measures = rl.response_measures(["Ahoy there matey, ahoy!", "A plain sentence."])
    assert measures["marker_instrument"] == persona_markers.MARKER_SET_VERSION
    assert measures["median_response_chars"] > 0
    assert measures["marker_density_per_100w"] > 0
    assert rl.response_measures([])["median_response_chars"] == 0.0


# ------------------------------------------------------------------- CLI


def test_cli_plan_is_free(capsys):
    assert rl.main(["plan", "--prompts", "80"]) == 0
    out = capsys.readouterr().out
    assert "RL plan" in out
    assert "judge calls" in out


def test_cli_config_prints_the_stops(capsys):
    assert rl.main(["config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection_rule"] == rl.SELECTION_RULE
    assert payload["kl_index_multiples"] == [0.25, 0.5, 1.0, 2.0]
    assert payload["audit_bank"]["prompts"] == 64


def test_cli_refuses_a_group_size_other_than_four(capsys):
    assert rl.main(["plan", "--group-size", "8"]) == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_writes_the_plan_json(tmp_path):
    out = tmp_path / "plan.json"
    assert rl.main(["plan", "--prompts", "80", "--out", str(out), "--json"]) == 0
    assert json.loads(out.read_text())["recipe_version"] == rl.RL_RECIPE_VERSION


# ------------------------------------------------------------ instrument


def test_the_kl_audit_instrument_is_registered_and_stamped():
    entry = instruments.get(rl.KL_AUDIT_INSTRUMENT_ID)
    assert entry.kind == instruments.KIND_GENERATION
    assert entry.prompts == {}, "bank text lives with the reserved bank, not here"
    stamp = rl.kl_audit_instrument()
    assert stamp["instrument_hash"] == entry.content_hash
    assert stamp["estimator"] == "k3"


def test_the_kl_audit_instrument_matches_the_module_constants():
    sampling = instruments.get(rl.KL_AUDIT_INSTRUMENT_ID).sampling
    assert sampling["prompts"] == rl.AUDIT_BANK_PROMPTS
    assert sampling["rollouts_per_prompt"] == rl.AUDIT_BANK_ROLLOUTS
    assert sampling["max_tokens"] == rl.RL_PILOT.max_response_tokens
    assert sampling["temperature"] == rl.RL_PILOT.temperature


def test_the_runner_never_imports_the_vendored_cookbook_at_module_scope():
    source = (REPO / "octt" / "rl_character.py").read_text(encoding="utf-8")
    head = source.split("# ---------------------------------------------------------")[0]
    assert "import tinker" not in head
    assert "from tinker_cookbook" not in head
