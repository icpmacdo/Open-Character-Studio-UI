"""Shared policy-gradient RL runner for Phase 3 character acquisition.

Two of the five acquisition arms in the readiness doc's experiment matrix are
policy-gradient RL over the *same* loop, differing only in who supplies the
reward:

    RL prompted    reward = the prompted character judge (octt/preference.py)
    RL trained-PM  reward = the trained preference model (octt/reward_model.py)

This module is that one loop, plus the telemetry and the stops that make its
output interpretable. It is first-party because ``tinker-cookbook/`` is vendored
read-only, and because the stock recipe has five specific gaps (readiness doc,
"Stock-recipe gaps to isolate"). Four of them are handled here; the fifth
(OPD's asymmetric teacher context) is already closed in
``octt/on_policy_character.py`` and is not redone.

**Gap 1 — the tournament is only complete at G=4.**
``tinker_cookbook/rl/preference_envs.py:get_pairs_chunked`` divides a group into
*contiguous* chunks of ``matchup_group_size`` (default 4) and runs the pattern
inside each chunk. At G=8 that is two independent 4-player tournaments, not one
8-player one, and nothing says so. RL here is pinned to G=4 and
:class:`RLConfig` REFUSES any other group size or tournament semantics, so a
config can never quietly measure a different tournament than the one reported.

**Gap 2 — no frozen KL reference.**
The stock RLHF pipeline only builds a base sampling client when a KL penalty is
switched on. Phase 3 indexes every arm by its divergence from the unmodified
base, so the reference is needed even at ``kl_penalty_coefficient = 0``.
:class:`RLConfig` therefore requires a :class:`ReferencePolicy` unconditionally
and the paid loop builds its client before the first training request.

**Gap 3 — ``kl_policy_base`` is a signed k1 difference.**
``tinker_cookbook/rl/metrics.py`` returns ``mean(logp_policy - logp_base)``,
which is the right quantity for the penalty it applies to advantages and the
wrong one for monitoring: it is negative about as often as positive and does not
estimate a KL. Training behavior is preserved verbatim (:func:`kl_policy_base_k1`
computes the identical number under the identical name) and the nonnegative k3
estimator is ADDED under names that cannot be confused with it
(``rl/reference_k3_*``). The estimator itself is imported from
:mod:`octt.on_policy_character`, never re-derived.

**Gap 4 — invalid labels are recorded as ties.**
``PreferenceModelFromChatRenderer.__call__`` returns ``0.0`` for the token
``"Tie"`` and ``0.0`` again for anything it could not parse. A reward that
cannot distinguish "the judge said they are equal" from "the judge emitted
garbage" degrades silently as the policy drifts off-distribution.
:class:`ValidityLedger` counts ``true_tie``, ``invalid`` and
``swap_inconsistent`` separately, and :class:`RunMonitor` ABORTS the run when
validity falls below :data:`VALIDITY_FLOOR`.

**Indexing.** Checkpoints are saved on a fixed step cadence and *indexed* by
observed KL: the banked 4B DPO acquisition checkpoint's mean response-sum k3 on
a fixed 64-prompt, two-rollout audit bank is :math:`K_{DPO}`
(:func:`measure_k_dpo`), and RL evaluation is reported at first crossings of
0.25, 0.5, 1 and 2 times it (:func:`first_crossings`). No universal KL threshold
is assumed anywhere.

**Selection.** :func:`select_checkpoint` picks the PEAK of an independent
validation measure among checkpoints that precede any guardrail breach. It never
looks at the proxy reward except to report where the proxy's own peak was, so a
run whose proxy kept climbing past the character peak is visible rather than
selected. The final held-out test set is guarded by :class:`HeldOutTestSet`,
which can be opened exactly once, only after a selection exists, and remembers
on disk that it was used.

Heavy dependencies (``tinker``, ``tinker_cookbook``) are imported lazily inside
functions: this module must import and its tests must pass with no training
stack and no API keys.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from . import instruments, manifest, models, on_policy_character, persona_markers, preference
from .on_policy_character import KLTelemetry, k3_per_token, kl_k3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Versions and pinned identifiers
# ---------------------------------------------------------------------------

#: Bumped when the RUNNER's contract changes (what a step does, what a stop
#: means, how reward is aggregated) — not when unrelated code moves. Stamped
#: into every plan, metrics row, and selection record.
RL_RECIPE_VERSION = "rl-character-v1"

#: The KL audit-bank instrument: 64 reserved prompts x 2 rollouts at the pilot's
#: sampling settings. It defines K_DPO, which is the x-axis of the whole Phase 3
#: comparison, so it is a measurement instrument like any judge.
KL_AUDIT_INSTRUMENT_ID = "kl-audit/dpo-index-64x2-v1"

METRIC_PREFIX = "rl/"

#: Reward providers this runner knows how to drive.
PROVIDER_PROMPTED_JUDGE = "prompted-judge"
PROVIDER_TRAINED_PM = "trained-pm"
REWARD_PROVIDERS = (PROVIDER_PROMPTED_JUDGE, PROVIDER_TRAINED_PM)

#: The ONLY tournament semantics this runner will run. Named explicitly so a
#: config that wants the cookbook's chunked behavior has to say so and be
#: refused, rather than getting it by accident at G != 4.
TOURNAMENT_COMPLETE_BOTH_DIRECTIONS = "complete-both-directions"
#: What the vendored builder actually does above G=4 (contiguous chunks of
#: ``matchup_group_size``). Recognized so the refusal can name it.
TOURNAMENT_CHUNKED_FOURS = "chunked-contiguous-fours"
SUPPORTED_TOURNAMENTS = (TOURNAMENT_COMPLETE_BOTH_DIRECTIONS,)

#: Pinned by the readiness doc: the complete both-directions tournament is only
#: well-defined at G=4 in the vendored builder.
REQUIRED_GROUP_SIZE = 4
#: 6 unordered pairs, each judged in both directions = 12 ordered matchups.
PAIRS_PER_GROUP = REQUIRED_GROUP_SIZE * (REQUIRED_GROUP_SIZE - 1) // 2
ORDERED_MATCHUPS_PER_GROUP = REQUIRED_GROUP_SIZE * (REQUIRED_GROUP_SIZE - 1)

#: Positions a pairwise scorer is told about. Mirrors
#: ``octt.reward_model.POSITION_A``/``POSITION_B`` without importing that module
#: at import time (it is heavy and independently authored); a drift test in
#: tests/test_rl_character.py asserts the two agree.
POSITION_A = "a"
POSITION_B = "b"

#: Audit-bank shape. Fixed by the readiness doc; changing either number changes
#: what K_DPO means, so both are constants and both are validated.
AUDIT_BANK_PROMPTS = 64
AUDIT_BANK_ROLLOUTS = 2

#: The frozen bank on disk, and the identity it must hash to. The FILE is the
#: prompt text; :data:`AUDIT_BANK_HASH` is what :class:`AuditBank` stamps onto
#: every K_DPO record. Both are pinned here — in production code, not only in a
#: test — because :func:`load_kl_audit_bank` is the one supported way to build
#: the bank and an unpinned loader would let a drifted file redefine K_DPO
#: silently. To change the bank, mint ``kl-audit-64x2-v2`` and add new
#: constants; never edit these in place (same rule as
#: ``coherence.JUDGE_TRAIT_SETS`` and ``persona_markers.MARKER_SETS``).
KL_AUDIT_BANK_ID = "kl-audit-64x2-v1"
KL_AUDIT_BANK_RELPATH = "data/qualitative_panels/kl-audit-64x2-v1.json"
AUDIT_BANK_HASH = "c50bca08a85517c0"

#: Where RL evaluation is reported, as multiples of K_DPO.
KL_INDEX_MULTIPLES = (0.25, 0.5, 1.0, 2.0)

#: Label for the frozen-reference k3 telemetry. Prefixes every reference-KL
#: metric key so a reference-relative number can never be read as anything else.
KL_LABEL_REFERENCE = "reference"

EXECUTION_MODE_DRY_RUN = manifest.EXECUTION_MODE_DRY_RUN
EXECUTION_MODE_REAL = manifest.EXECUTION_MODE_REAL


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RLConfigError(ValueError):
    """A configuration this runner refuses to run."""


class MissingReferenceError(RLConfigError):
    """No frozen KL reference. Phase 3 needs one even at zero KL penalty."""


class RLShapeError(ValueError):
    """Two arrays that must describe the same tokens do not."""


class RewardValidityError(RuntimeError):
    """The reward provider's validity fell below the calibrated floor."""

    def __init__(self, ledger: ValidityLedger, floor: float) -> None:
        self.ledger = ledger
        self.floor = floor
        super().__init__(
            f"reward-provider validity {ledger.validity_rate:.4f} is below the "
            f"{floor:.4f} floor ({ledger.invalid} invalid of {ledger.total} queries); "
            "an invalid label is missing data, not a tie, and a reward built from "
            "them is not the reward that was gated"
        )


class RunHalted(RuntimeError):
    """A hard stop fired. The run is over; selection uses what came before."""

    def __init__(self, breaches: Sequence[GuardrailBreach]) -> None:
        self.breaches = tuple(breaches)
        detail = "; ".join(b.summary() for b in self.breaches)
        super().__init__(f"RL halted by {len(self.breaches)} guardrail breach(es): {detail}")


class NoEligibleCheckpoint(RuntimeError):
    """Every checkpoint is at or after a breach; there is nothing to select."""


class AuditBankUnavailable(RLConfigError):
    """The frozen KL audit bank could not be read. Never a soft fallback."""


class AuditBankDrifted(RLConfigError):
    """The bank on disk is not the bank K_DPO was defined on. Fatal, not a warning."""


class TestSetAlreadyUsed(RuntimeError):
    """The held-out test set has already been opened. There is no second look."""


class SelectionRequired(RuntimeError):
    """The held-out test set was opened before a checkpoint was selected."""


# ---------------------------------------------------------------------------
# Frozen reference and configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePolicy:
    """The frozen policy every KL number in this arm is measured against.

    ``checkpoint_uri = None`` means the unmodified base weights, which is Phase
    3's definition of the reference: all acquisition arms start there, so it is
    the only reference under which DPO, BoN, RL and OPD divergences are
    comparable numbers.
    """

    model_id: str
    checkpoint_uri: str | None = None
    role: str = "base"

    @property
    def fingerprint(self) -> str:
        return self.checkpoint_uri or f"base:{self.model_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "checkpoint_uri": self.checkpoint_uri,
            "role": self.role,
            "fingerprint": self.fingerprint,
        }


#: The pilot's policy rung (readiness doc, "Initial rung and rank").
POLICY_MODEL = "Qwen/Qwen3.5-4B"

DEFAULT_REFERENCE = ReferencePolicy(model_id=POLICY_MODEL, checkpoint_uri=None, role="base")


@dataclass(frozen=True)
class RLConfig:
    """The RL pilot (readiness doc, "Starting pilot configuration").

    The defaults ARE the pilot table. They are not tuning knobs to edit in
    place: construct a new config at the call site so the config hash records
    the divergence.
    """

    policy_model: str = POLICY_MODEL
    lora_rank: int = 32
    learning_rate: float = 1e-5
    group_size: int = REQUIRED_GROUP_SIZE
    prompts_per_batch: int = 8
    temperature: float = 1.0
    max_response_tokens: int = 512
    max_steps: int = 50
    save_every: int = 5
    eval_every: int = 5
    # Which reward the loop optimizes. The loop is identical either way; this
    # only selects the provider and is stamped on every row.
    reward_provider: str = PROVIDER_PROMPTED_JUDGE
    tournament: str = TOURNAMENT_COMPLETE_BOTH_DIRECTIONS
    # Gap 2: the reference exists regardless of the penalty. Zero is the pilot's
    # coefficient — KL is an INDEX here, not a regularizer.
    kl_penalty_coefficient: float = 0.0
    kl_discount_factor: float = 0.0
    reference: ReferencePolicy | None = DEFAULT_REFERENCE

    def __post_init__(self) -> None:
        if self.group_size != REQUIRED_GROUP_SIZE:
            raise RLConfigError(
                f"group_size must be exactly {REQUIRED_GROUP_SIZE}, got {self.group_size}. "
                "The vendored preference group builder "
                "(tinker_cookbook/rl/preference_envs.py:get_pairs_chunked) runs the "
                "intended complete both-directions tournament ONLY at group size 4; "
                "larger groups are silently split into contiguous fours, which is a "
                "different tournament than the one this arm reports."
            )
        if self.tournament not in SUPPORTED_TOURNAMENTS:
            extra = (
                " (that is what the vendored builder does above G=4, and it is exactly "
                "what this runner refuses to report as a complete tournament)"
                if self.tournament == TOURNAMENT_CHUNKED_FOURS
                else ""
            )
            raise RLConfigError(
                f"unsupported tournament semantics {self.tournament!r}{extra}; "
                f"this runner implements {SUPPORTED_TOURNAMENTS}"
            )
        if self.reward_provider not in REWARD_PROVIDERS:
            raise RLConfigError(
                f"unknown reward_provider {self.reward_provider!r}; "
                f"choose from {REWARD_PROVIDERS}"
            )
        if self.reference is None:
            raise MissingReferenceError(
                "Phase 3 RL requires a frozen KL reference EVEN WHEN "
                "kl_penalty_coefficient is 0: every arm is indexed by its divergence "
                "from the unmodified base, and the stock RLHF pipeline only builds a "
                "base client when a penalty is switched on. Pass "
                "reference=ReferencePolicy(...)."
            )
        for name in ("prompts_per_batch", "max_steps", "save_every", "eval_every", "lora_rank"):
            if getattr(self, name) < 1:
                raise RLConfigError(f"RLConfig.{name} must be >= 1")
        if self.max_response_tokens < 1:
            raise RLConfigError("RLConfig.max_response_tokens must be >= 1")
        if self.temperature <= 0:
            raise RLConfigError("RLConfig.temperature must be > 0 (RL samples on-policy)")
        if self.kl_penalty_coefficient < 0:
            raise RLConfigError("RLConfig.kl_penalty_coefficient must be >= 0")

    @property
    def samples_per_step(self) -> int:
        return self.prompts_per_batch * self.group_size

    @property
    def groups_per_step(self) -> int:
        return self.prompts_per_batch

    def require_reference(self) -> ReferencePolicy:
        """The frozen reference, or a refusal. Called before any paid request."""
        if self.reference is None:  # pragma: no cover - __post_init__ already refuses
            raise MissingReferenceError("no frozen KL reference configured")
        return self.reference

    def to_dict(self) -> dict[str, Any]:
        out = {
            k: v for k, v in vars(self).items() if k != "reference"
        }
        out["reference"] = self.require_reference().to_dict()
        out["recipe_version"] = RL_RECIPE_VERSION
        out["config_hash"] = manifest.config_hash(self)
        return out


RL_PILOT = RLConfig()


# ---------------------------------------------------------------------------
# Gap 4: invalid outcomes are not ties
# ---------------------------------------------------------------------------

OUTCOME_A = "a"
OUTCOME_B = "b"
OUTCOME_TIE = "tie"
OUTCOME_INVALID = "invalid"
OUTCOME_INCONSISTENT = "swap_inconsistent"
OUTCOMES = (OUTCOME_A, OUTCOME_B, OUTCOME_TIE, OUTCOME_INVALID, OUTCOME_INCONSISTENT)


@dataclass(frozen=True)
class ValidityLedger:
    """How a reward provider's queries came out, with invalid kept apart from tie.

    ``decisive`` + ``true_tie`` + ``swap_inconsistent`` + ``invalid`` == ``total``.

      * ``true_tie`` is a MEASUREMENT: the provider looked and said "equal".
      * ``swap_inconsistent`` is a measurement too: both presentations parsed and
        disagreed, which quantifies position bias.
      * ``invalid`` is MISSING DATA: nothing was measured. The stock preference
        path returns 0.0 for it, identically to a tie, which is the conflation
        this ledger exists to prevent.
    """

    decisive: int = 0
    true_tie: int = 0
    swap_inconsistent: int = 0
    invalid: int = 0

    @property
    def total(self) -> int:
        return self.decisive + self.true_tie + self.swap_inconsistent + self.invalid

    @property
    def validity_rate(self) -> float:
        """Fraction of queries that produced a parseable outcome. 1.0 on empty."""
        total = self.total
        return 1.0 if total == 0 else (total - self.invalid) / total

    @property
    def swap_consistency_rate(self) -> float:
        """Of the parseable queries, how many agreed with themselves under swap."""
        parseable = self.total - self.invalid
        return 1.0 if parseable == 0 else (parseable - self.swap_inconsistent) / parseable

    def __add__(self, other: ValidityLedger) -> ValidityLedger:
        return ValidityLedger(
            decisive=self.decisive + other.decisive,
            true_tie=self.true_tie + other.true_tie,
            swap_inconsistent=self.swap_inconsistent + other.swap_inconsistent,
            invalid=self.invalid + other.invalid,
        )

    @classmethod
    def from_outcomes(cls, outcomes: Sequence[str]) -> ValidityLedger:
        counts = {name: 0 for name in OUTCOMES}
        for outcome in outcomes:
            if outcome not in counts:
                raise ValueError(f"unknown reward outcome {outcome!r}; expected {OUTCOMES}")
            counts[outcome] += 1
        return cls(
            decisive=counts[OUTCOME_A] + counts[OUTCOME_B],
            true_tie=counts[OUTCOME_TIE],
            swap_inconsistent=counts[OUTCOME_INCONSISTENT],
            invalid=counts[OUTCOME_INVALID],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisive": self.decisive,
            "true_tie": self.true_tie,
            "swap_inconsistent": self.swap_inconsistent,
            "invalid": self.invalid,
            "total": self.total,
            "validity_rate": self.validity_rate,
            "swap_consistency_rate": self.swap_consistency_rate,
        }


#: Below this, the reward is not the reward the pre-RL gates certified, so the
#: run aborts. Predeclared (readiness doc, "Initial hard stops").
VALIDITY_FLOOR = 0.99


def assert_validity(ledger: ValidityLedger, floor: float = VALIDITY_FLOOR) -> None:
    """Abort when invalid parsing exceeds the calibrated threshold."""
    if ledger.validity_rate < floor:
        raise RewardValidityError(ledger, floor)


# ---------------------------------------------------------------------------
# Reward providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupReward:
    """One G=4 group's rewards plus the accounting behind them."""

    prompt: str
    rewards: tuple[float, ...]
    outcomes: tuple[tuple[int, int, str], ...]
    ledger: ValidityLedger
    provider_id: str
    instrument_id: str
    instrument_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": prompt_id(self.prompt),
            "rewards": list(self.rewards),
            "outcomes": [{"i": i, "j": j, "outcome": o} for i, j, o in self.outcomes],
            "validity": self.ledger.to_dict(),
            "provider_id": self.provider_id,
            "instrument_id": self.instrument_id,
            "instrument_hash": self.instrument_hash,
        }


class RewardProvider(Protocol):
    """The narrow adapter every reward source implements for this runner.

    Deliberately tiny: a provider takes one prompt and its G=4 on-policy group
    and returns per-response rewards plus a :class:`ValidityLedger`. Everything
    else — swap resolution, caching, instrument stamping — belongs to the module
    that owns the reward (``octt/preference.py`` for the prompted judge,
    ``octt/reward_model.py`` for the trained preference model) and is reused
    from there rather than reimplemented here.
    """

    provider_id: str
    instrument_id: str
    instrument_hash: str

    def score_group(self, prompt: str, responses: Sequence[str]) -> GroupReward:
        ...


def prompt_id(prompt: str) -> str:
    """Stable identity for a prompt (content only; no ordering, no run state)."""
    return manifest.content_hash(prompt)[:12]


def require_group_size(responses: Sequence[str]) -> None:
    if len(responses) != REQUIRED_GROUP_SIZE:
        raise RLConfigError(
            f"group has {len(responses)} responses; the complete both-directions "
            f"tournament is only well-defined at G={REQUIRED_GROUP_SIZE}"
        )


def unordered_pairs(n: int = REQUIRED_GROUP_SIZE) -> list[tuple[int, int]]:
    """Every unordered pair of a group, in a fixed order."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def aggregate_outcomes(
    n: int, outcomes: Sequence[tuple[int, int, str]]
) -> tuple[float, ...]:
    """Win-minus-loss per response, normalized by matchups played.

    Same aggregation as the stock preference environment
    (``win_minus_loss / matchup_count``), so the advantage scale of this arm is
    comparable to a cookbook-run one. Ties, swap-inconsistent pairs, and invalid
    pairs all contribute 0 to both sides — but they are counted in the ledger
    separately, because "the pair was a tie" and "the pair was never measured"
    are the same number here and must not be the same fact.
    """
    scores = [0.0] * n
    played = [0] * n
    for i, j, outcome in outcomes:
        played[i] += 1
        played[j] += 1
        if outcome == OUTCOME_A:
            scores[i] += 1.0
            scores[j] -= 1.0
        elif outcome == OUTCOME_B:
            scores[j] += 1.0
            scores[i] -= 1.0
    return tuple(s / p if p else 0.0 for s, p in zip(scores, played))


def centered_advantages(rewards: Sequence[float]) -> list[float]:
    """Group-centered advantages — the stock RL convention, kept identical."""
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [float(r) - mean for r in rewards]


@dataclass
class PromptedJudgeReward:
    """Reward = the prompted character judge, run as a complete G=4 tournament.

    Delegates every judgment to :func:`octt.preference.compare`, which already
    does blind presentation, both orders, swap resolution, caching, and
    instrument stamping. This class only maps that module's resolution reasons
    onto the ledger's four outcomes and aggregates.
    """

    runtime: Any
    brief: preference.CharacterBrief | None = None
    judge_model: str = models.TEACHER_MODEL
    config: preference.PreferenceJudgeConfig = preference.DEFAULT_JUDGE_CONFIG
    cache_path: Path | None = None
    execute: bool = False
    dry_run_policy: str = preference.DRY_RUN_TIE
    concurrency: int = 32
    policy_id: str = "rl-policy"
    step: int = 0
    provider_id: str = PROVIDER_PROMPTED_JUDGE

    def __post_init__(self) -> None:
        self.brief = self.brief or preference.get_brief()
        stamp = preference.judge_instrument(self.judge_model, self.config, self.brief)
        self.instrument_id: str = stamp["instrument_id"]
        self.instrument_hash: str = stamp["instrument_hash"]
        self.judge_stamp: dict[str, Any] = stamp

    def _pairs(self, prompt: str, responses: Sequence[str]) -> list[preference.PreferencePair]:
        pid = prompt_id(prompt)
        cell = f"{pid}::{self.policy_id}@step{self.step}"
        return [
            preference.PreferencePair(
                cell_id=cell,
                prompt_id=pid,
                prompt=prompt,
                response_a=responses[i],
                response_b=responses[j],
                candidate_a=f"{cell}#{i}",
                candidate_b=f"{cell}#{j}",
                index_a=i,
                index_b=j,
            )
            for i, j in unordered_pairs(len(responses))
        ]

    def score_group(self, prompt: str, responses: Sequence[str]) -> GroupReward:
        require_group_size(responses)
        pairs = self._pairs(prompt, responses)
        rows = preference.compare(
            pairs,
            self.runtime,
            brief=self.brief,
            judge_model=self.judge_model,
            config=self.config,
            cache_path=self.cache_path,
            execute=self.execute,
            dry_run_policy=self.dry_run_policy,
            concurrency=self.concurrency,
        )
        outcomes = [
            (row["index_a"], row["index_b"], _outcome_from_row(row)) for row in rows
        ]
        return GroupReward(
            prompt=prompt,
            rewards=aggregate_outcomes(len(responses), outcomes),
            outcomes=tuple(outcomes),
            ledger=ValidityLedger.from_outcomes([o for _, _, o in outcomes]),
            provider_id=self.provider_id,
            instrument_id=self.instrument_id,
            instrument_hash=self.instrument_hash,
        )


def _outcome_from_row(row: Mapping[str, Any]) -> str:
    """Map one ``octt.preference`` resolved row onto a ledger outcome.

    The mapping is the whole point of gap 4: ``REASON_UNPARSEABLE`` is
    ``invalid`` (missing data), ``REASON_BOTH_TIE`` is ``tie`` (a measurement),
    and they never collapse into each other.
    """
    reason = row.get("resolution_reason")
    if reason == preference.REASON_UNPARSEABLE:
        return OUTCOME_INVALID
    if reason == preference.REASON_BOTH_TIE:
        return OUTCOME_TIE
    if reason == preference.REASON_DISAGREE:
        return OUTCOME_INCONSISTENT
    resolution = row.get("resolution")
    if resolution == preference.RESOLUTION_A:
        return OUTCOME_A
    if resolution == preference.RESOLUTION_B:
        return OUTCOME_B
    if resolution == preference.RESOLUTION_TIE:
        return OUTCOME_TIE
    return OUTCOME_INVALID


#: Margin below which a pointwise scorer's verdict is a TIE rather than a win.
#: Predeclared; a scorer whose margins are all inside it produces an all-tie
#: tournament, which the ledger reports rather than hiding as zero reward.
DEFAULT_TIE_EPSILON = 1e-9


@dataclass
class TrainedPMReward:
    """Reward = a trained preference model, run as the same G=4 tournament.

    ``model`` satisfies ``octt.reward_model.RewardModel``: ``pointwise: bool``
    and ``score(prompt, response, *, position) -> float``. Both orientations of
    every pair are scored, exactly as the prompted judge is, so a non-pointwise
    (pairwise/listwise) reward head has to earn its swap consistency instead of
    being assumed order-blind. A pointwise model is structurally consistent and
    the two orientations agree by construction.

    A non-finite score is ``invalid``, never a tie: NaN is what a reward head
    returns when it failed, and averaging it into the group as 0.0 is precisely
    the silent corruption gap 4 describes.
    """

    model: Any
    tie_epsilon: float = DEFAULT_TIE_EPSILON
    provider_id: str = PROVIDER_TRAINED_PM
    instrument_id: str = "reward-model/pre-rl-controls-v1"
    instrument_hash: str = ""
    model_fingerprint: str = "unspecified"

    def _margin(self, prompt: str, left: str, right: str) -> float | None:
        try:
            a = float(self.model.score(prompt, left, position=POSITION_A))
            b = float(self.model.score(prompt, right, position=POSITION_B))
        except (TypeError, ValueError):
            return None
        value = a - b
        if not math.isfinite(value):
            return None
        return value

    def _outcome(self, prompt: str, left: str, right: str) -> str:
        direct = self._margin(prompt, left, right)
        swapped = self._margin(prompt, right, left)
        if direct is None or swapped is None:
            return OUTCOME_INVALID
        direct_call = _call(direct, self.tie_epsilon)
        # In the swapped presentation the sides are flipped, so agreement means
        # the opposite call.
        swapped_call = _flip(_call(swapped, self.tie_epsilon))
        if direct_call != swapped_call:
            return OUTCOME_INCONSISTENT
        return direct_call

    def score_group(self, prompt: str, responses: Sequence[str]) -> GroupReward:
        require_group_size(responses)
        outcomes = [
            (i, j, self._outcome(prompt, responses[i], responses[j]))
            for i, j in unordered_pairs(len(responses))
        ]
        return GroupReward(
            prompt=prompt,
            rewards=aggregate_outcomes(len(responses), outcomes),
            outcomes=tuple(outcomes),
            ledger=ValidityLedger.from_outcomes([o for _, _, o in outcomes]),
            provider_id=self.provider_id,
            instrument_id=self.instrument_id,
            instrument_hash=self.instrument_hash,
        )


def _call(margin: float, epsilon: float) -> str:
    if margin > epsilon:
        return OUTCOME_A
    if margin < -epsilon:
        return OUTCOME_B
    return OUTCOME_TIE


def _flip(outcome: str) -> str:
    return {OUTCOME_A: OUTCOME_B, OUTCOME_B: OUTCOME_A}.get(outcome, outcome)


@dataclass
class LabelPreferenceReward:
    """Reward = a one-token-label preference model, with invalid kept as invalid.

    This is the shape the vendored ``PreferenceModelFromChatRenderer`` has:
    sample one token, map ``A``/``B``/``Tie`` to a float. Its ``else`` branch
    returns the tie value for anything unparseable. Here ``label_fn`` returns
    ``None`` for unparseable and that becomes :data:`OUTCOME_INVALID`, so the
    ledger — and therefore the abort — can see it.
    """

    label_fn: Callable[[str, str, str], str | None]
    provider_id: str = PROVIDER_TRAINED_PM
    instrument_id: str = "reward-model/pre-rl-controls-v1"
    instrument_hash: str = ""

    def _outcome(self, prompt: str, left: str, right: str) -> str:
        direct = _normalize_label(self.label_fn(prompt, left, right))
        swapped = _normalize_label(self.label_fn(prompt, right, left))
        if direct is None or swapped is None:
            return OUTCOME_INVALID
        if direct != _flip(swapped):
            return OUTCOME_INCONSISTENT
        return direct

    def score_group(self, prompt: str, responses: Sequence[str]) -> GroupReward:
        require_group_size(responses)
        outcomes = [
            (i, j, self._outcome(prompt, responses[i], responses[j]))
            for i, j in unordered_pairs(len(responses))
        ]
        return GroupReward(
            prompt=prompt,
            rewards=aggregate_outcomes(len(responses), outcomes),
            outcomes=tuple(outcomes),
            ledger=ValidityLedger.from_outcomes([o for _, _, o in outcomes]),
            provider_id=self.provider_id,
            instrument_id=self.instrument_id,
            instrument_hash=self.instrument_hash,
        )


def _normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    text = str(label).strip().upper()
    if text == "A":
        return OUTCOME_A
    if text == "B":
        return OUTCOME_B
    if text in ("TIE", "TIED", "EQUAL"):
        return OUTCOME_TIE
    return None


def score_batch(
    provider: RewardProvider,
    groups: Sequence[tuple[str, Sequence[str]]],
    *,
    validity_floor: float = VALIDITY_FLOOR,
) -> tuple[list[GroupReward], ValidityLedger]:
    """Score a whole batch of G=4 groups and abort on low provider validity.

    The abort is at BATCH granularity on purpose: a single unparseable pair in
    32 samples is noise, and a provider that is 5% unparseable is a different
    reward than the one the pre-RL gates certified.
    """
    scored = [provider.score_group(prompt, responses) for prompt, responses in groups]
    ledger = ValidityLedger()
    for group in scored:
        ledger = ledger + group.ledger
    assert_validity(ledger, validity_floor)
    return scored, ledger


# ---------------------------------------------------------------------------
# Gap 3: KL telemetry (k3 monitoring beside the preserved signed k1)
# ---------------------------------------------------------------------------


def kl_metrics(telemetry: KLTelemetry, prefix: str = METRIC_PREFIX) -> dict[str, float]:
    """RL-named metric keys for a :class:`KLTelemetry` computed elsewhere.

    The estimator is :func:`octt.on_policy_character.kl_k3` — imported, not
    re-derived. This function only renames, so an OPD k3 and an RL k3 are the
    same quantity under different prefixes rather than two implementations that
    might drift.
    """
    return {
        f"{prefix}{telemetry.label}_k3_mean_token_nats": telemetry.mean_token_kl_nats,
        f"{prefix}{telemetry.label}_k3_response_sum_nats": telemetry.mean_response_sum_kl_nats,
        f"{prefix}{telemetry.label}_k3_max_response_sum_nats": (
            telemetry.max_response_sum_kl_nats
        ),
        f"{prefix}{telemetry.label}_k3_tokens": float(telemetry.num_tokens),
        f"{prefix}{telemetry.label}_k3_responses": float(telemetry.num_responses),
        f"{prefix}{telemetry.label}_k3_clamped_tokens": float(telemetry.clamped_tokens),
    }


def reference_kl(
    policy_logprobs: Sequence[Sequence[float]],
    ref_logprobs: Sequence[Sequence[float]],
    *,
    label: str = KL_LABEL_REFERENCE,
) -> KLTelemetry:
    """Nonnegative k3 KL of the policy against the frozen reference, in nats."""
    _check_matched(policy_logprobs, ref_logprobs)
    return kl_k3(label, policy_logprobs, ref_logprobs)


def kl_policy_base_k1(
    policy_logprobs: Sequence[Sequence[float]],
    ref_logprobs: Sequence[Sequence[float]],
) -> dict[str, float]:
    """The cookbook's ``kl_policy_base``, computed identically and NAMED clearly.

    ``mean(logp_policy - logp_base)`` over action tokens — the same signed k1
    quantity ``tinker_cookbook/rl/metrics.py`` returns and applies to advantages.
    Training behavior is preserved: this number, not the k3 one, is what a KL
    penalty would use. It is returned under the cookbook's own key so banked runs
    stay comparable, plus an explicit alias that says what it is.
    """
    _check_matched(policy_logprobs, ref_logprobs)
    total = 0.0
    tokens = 0
    for policy, ref in zip(policy_logprobs, ref_logprobs):
        for p, r in zip(policy, ref):
            total += float(p) - float(r)
            tokens += 1
    value = total / tokens if tokens else 0.0
    return {
        # Verbatim cookbook key and meaning.
        "kl_policy_base": value,
        f"{METRIC_PREFIX}kl_policy_base_k1_signed_nats": value,
    }


def _check_matched(
    policy: Sequence[Sequence[float]], reference: Sequence[Sequence[float]]
) -> None:
    if len(policy) != len(reference):
        raise RLShapeError(
            f"{len(policy)} policy responses vs {len(reference)} reference responses"
        )
    for index, (p, r) in enumerate(zip(policy, reference)):
        if len(p) != len(r):
            raise RLShapeError(
                f"response {index} has {len(p)} policy log-probabilities and {len(r)} "
                "reference log-probabilities; these do not describe the same tokens"
            )


def response_sum_kl(
    policy_logprobs: Sequence[Sequence[float]],
    ref_logprobs: Sequence[Sequence[float]],
) -> list[float]:
    """Per-response summed k3 KL in nats (the quantity K_DPO indexes)."""
    _check_matched(policy_logprobs, ref_logprobs)
    out: list[float] = []
    for policy, ref in zip(policy_logprobs, ref_logprobs):
        per_token, _clamped = k3_per_token(policy, ref)
        out.append(sum(per_token))
    return out


# ---------------------------------------------------------------------------
# K_DPO: the data-derived KL index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditBank:
    """The fixed prompt bank K_DPO is measured on.

    Exactly :data:`AUDIT_BANK_PROMPTS` prompts, :data:`AUDIT_BANK_ROLLOUTS`
    rollouts each. The prompt TEXT is supplied by the caller and hashed, the way
    the W2 and Best-of-N instruments take their text from a separately hashed
    panel: the bank must be held out from training, validation and the final
    test set, and this module has no business authoring reserved prompts.
    """

    bank_id: str
    prompts: tuple[str, ...]
    rollouts_per_prompt: int = AUDIT_BANK_ROLLOUTS

    def __post_init__(self) -> None:
        if len(self.prompts) != AUDIT_BANK_PROMPTS:
            raise RLConfigError(
                f"the KL audit bank is fixed at {AUDIT_BANK_PROMPTS} prompts, got "
                f"{len(self.prompts)}; K_DPO measured on a different bank is a "
                "different index and is not comparable to a banked one"
            )
        if len(set(self.prompts)) != len(self.prompts):
            raise RLConfigError("the KL audit bank contains duplicate prompts")
        if self.rollouts_per_prompt != AUDIT_BANK_ROLLOUTS:
            raise RLConfigError(
                f"the KL audit bank is fixed at {AUDIT_BANK_ROLLOUTS} rollouts per prompt"
            )

    @property
    def num_responses(self) -> int:
        return len(self.prompts) * self.rollouts_per_prompt

    @property
    def content_hash(self) -> str:
        return manifest.content_hash(
            KL_AUDIT_INSTRUMENT_ID, self.bank_id, self.rollouts_per_prompt, self.prompts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "prompts": len(self.prompts),
            "rollouts_per_prompt": self.rollouts_per_prompt,
            "responses": self.num_responses,
            "audit_bank_hash": self.content_hash,
            "instrument_id": KL_AUDIT_INSTRUMENT_ID,
            "instrument_hash": kl_audit_instrument()["instrument_hash"],
        }


def kl_audit_instrument() -> dict[str, Any]:
    """Provenance stamp for the KL audit bank's sampling settings."""
    entry = instruments.get(KL_AUDIT_INSTRUMENT_ID)
    return {
        "instrument_id": KL_AUDIT_INSTRUMENT_ID,
        "instrument_hash": entry.content_hash,
        "renderer": entry.renderer,
        "sampling": dict(entry.sampling),
        "estimator": "k3",
        "units": "nats",
    }


def load_kl_audit_bank(repo_root: Path | str | None = None) -> AuditBank:
    """Build the ONE supported :class:`AuditBank` from the frozen panel on disk.

    This is the production constructor. Every call site that measures K_DPO must
    come through here rather than assembling an :class:`AuditBank` from whatever
    prompts it has to hand: a bank built ad hoc still satisfies the shape checks
    in :meth:`AuditBank.__post_init__` (64 unique prompts, 2 rollouts) while
    hashing to something else entirely, and a K_DPO measured on it would index
    every reported crossing against a different x-axis without anything saying so.

    Three things are checked and none of them warn:

      * the file exists (a missing instrument is :class:`AuditBankUnavailable`,
        never an empty bank);
      * its ``panel_id`` is :data:`KL_AUDIT_BANK_ID`;
      * the constructed bank hashes to :data:`AUDIT_BANK_HASH`.

    A drifted bank raises :class:`AuditBankDrifted`. That is the whole point: the
    correct response to a changed bank is to mint ``kl-audit-64x2-v2`` and add a
    new pinned hash, never to accept the new number under the old name.
    """
    from . import qualitative  # lazy: keeps the import graph shallow at module load

    root = Path(repo_root) if repo_root is not None else _repo_root()
    path = root / KL_AUDIT_BANK_RELPATH
    if not path.is_file():
        raise AuditBankUnavailable(
            f"the frozen KL audit bank is missing at {path}. It is source, not a "
            "generated artifact: check the data/ re-include rules in .gitignore. "
            "'could not load the bank' must never pass as 'measured K_DPO'."
        )
    try:
        panel = qualitative.load_panel(path)
    except (ValueError, KeyError, OSError) as exc:
        raise AuditBankUnavailable(f"{path} is not a valid prompt panel: {exc}") from exc
    if panel.panel_id != KL_AUDIT_BANK_ID:
        raise AuditBankDrifted(
            f"{path} declares panel_id {panel.panel_id!r}, not {KL_AUDIT_BANK_ID!r}; "
            "K_DPO is defined on one named bank and this is a different one"
        )
    bank = AuditBank(bank_id=panel.panel_id, prompts=tuple(p.text for p in panel.prompts))
    if bank.content_hash != AUDIT_BANK_HASH:
        raise AuditBankDrifted(
            f"{path} hashes to {bank.content_hash} but K_DPO is pinned to "
            f"{AUDIT_BANK_HASH}. The bank IS the x-axis of every Phase 3 crossing: "
            "a K_DPO measured on a changed bank is a different index, and every "
            "banked crossing reported as a multiple of it becomes incomparable. "
            "Mint kl-audit-64x2-v2 with new constants rather than editing v1."
        )
    return bank


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class KDPOIndex:
    """K_DPO and the crossing thresholds derived from it.

    K_DPO is the banked DPO acquisition checkpoint's MEAN RESPONSE-SUM k3 KL
    against the frozen reference, measured on the audit bank. Every RL crossing
    is reported as a multiple of it, so the comparison is data-derived rather
    than resting on an assumed universal KL budget.
    """

    k_dpo_nats: float
    mean_token_nats: float
    max_response_sum_nats: float
    num_responses: int
    num_prompts: int
    rollouts_per_prompt: int
    audit_bank_id: str
    audit_bank_hash: str
    checkpoint_fingerprint: str
    reference_fingerprint: str
    clamped_tokens: int = 0
    multiples: tuple[float, ...] = KL_INDEX_MULTIPLES

    def thresholds(self) -> dict[str, float]:
        return {f"{m:g}x": m * self.k_dpo_nats for m in self.multiples}

    @property
    def stop_threshold_nats(self) -> float:
        """2 * K_DPO — the hard stop, not merely an index point."""
        return KL_STOP_MULTIPLE * self.k_dpo_nats

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_dpo_response_sum_nats": self.k_dpo_nats,
            "k_dpo_mean_token_nats": self.mean_token_nats,
            "k_dpo_max_response_sum_nats": self.max_response_sum_nats,
            "responses": self.num_responses,
            "prompts": self.num_prompts,
            "rollouts_per_prompt": self.rollouts_per_prompt,
            "audit_bank_id": self.audit_bank_id,
            "audit_bank_hash": self.audit_bank_hash,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "clamped_tokens": self.clamped_tokens,
            "multiples": list(self.multiples),
            "thresholds_nats": self.thresholds(),
            "stop_threshold_nats": self.stop_threshold_nats,
            "instrument": kl_audit_instrument(),
        }


def measure_k_dpo(
    bank: AuditBank,
    policy_logprobs: Sequence[Sequence[float]],
    ref_logprobs: Sequence[Sequence[float]],
    *,
    checkpoint_fingerprint: str,
    reference: ReferencePolicy,
) -> KDPOIndex:
    """Measure the banked DPO checkpoint's k3 on the fixed audit bank.

    ``policy_logprobs`` are the DPO checkpoint's per-token log-probabilities over
    ITS OWN sampled responses (128 of them: 64 prompts x 2 rollouts) and
    ``ref_logprobs`` are the frozen reference's log-probabilities over the same
    tokens. The counts are validated, never assumed: a short bank silently
    shrinks K_DPO and moves every crossing that is reported against it.
    """
    if len(policy_logprobs) != bank.num_responses:
        raise RLConfigError(
            f"the audit bank is {bank.num_responses} responses "
            f"({len(bank.prompts)} prompts x {bank.rollouts_per_prompt} rollouts) but "
            f"{len(policy_logprobs)} were scored; K_DPO from a partial bank is not K_DPO"
        )
    telemetry = reference_kl(policy_logprobs, ref_logprobs, label="k_dpo")
    return KDPOIndex(
        k_dpo_nats=telemetry.mean_response_sum_kl_nats,
        mean_token_nats=telemetry.mean_token_kl_nats,
        max_response_sum_nats=telemetry.max_response_sum_kl_nats,
        num_responses=telemetry.num_responses,
        num_prompts=len(bank.prompts),
        rollouts_per_prompt=bank.rollouts_per_prompt,
        audit_bank_id=bank.bank_id,
        audit_bank_hash=bank.content_hash,
        checkpoint_fingerprint=checkpoint_fingerprint,
        reference_fingerprint=reference.fingerprint,
        clamped_tokens=telemetry.clamped_tokens,
    )


@dataclass(frozen=True)
class KDPOMeasurement:
    """A measured :class:`KDPOIndex` plus the responses it was measured on.

    The texts are returned rather than discarded so the caller can report what
    the bank actually elicited (script mix, lengths) beside the number. They are
    NOT part of the index: K_DPO is the log-probability quantity, and nothing
    derived from the text may feed back into it.
    """

    index: KDPOIndex
    texts: tuple[str, ...]
    prompt_tokens: int
    response_tokens: int


def measure_k_dpo_on_bank(  # pragma: no cover - paid path
    bank: AuditBank,
    runtime: Any,
    *,
    checkpoint_uri: str,
    policy_model: str = POLICY_MODEL,
    reference: ReferencePolicy = DEFAULT_REFERENCE,
    execute: bool = False,
    concurrency: int = 16,
) -> KDPOMeasurement:
    """Sample the banked DPO checkpoint on the audit bank and measure K_DPO.

    64 prompts x 2 rollouts at the sampling settings pinned by
    :data:`KL_AUDIT_INSTRUMENT_ID`, then the frozen reference's log-probabilities
    over the SAME token sequences, sliced to the completion span. The slice is
    explicit because ``compute_logprobs`` returns one entry per token of the
    sequence it was handed, so scoring a completion without its prompt in context
    silently measures a different conditional distribution.

    There is no dry-run result: a dry-run sampler returns no tokens, so a
    "dry-run K_DPO" would be 0.0 nats and would divide every crossing by zero.
    Without ``execute=True`` and a live runtime this refuses.
    """
    if not execute or getattr(runtime.config, "dry_run", True):
        raise RLConfigError(
            "K_DPO cannot be measured in dry-run: the dry-run sampler returns no "
            "tokens, so the index would be 0.0 nats and every crossing reported "
            "against it would be meaningless. Pass execute=True with a live runtime."
        )
    import asyncio

    import tinker

    from . import generation

    sampling = kl_audit_instrument()["sampling"]
    service_client = runtime.require_service_client()
    binding = runtime.renderer_binding(policy_model)
    renderer = binding.renderer
    stop = renderer.get_stop_sequences()
    policy_client = service_client.create_sampling_client(
        base_model=policy_model, model_path=checkpoint_uri
    )
    reference_client = service_client.create_sampling_client(
        base_model=reference.model_id, model_path=reference.checkpoint_uri
    )

    prefixes = [
        on_policy_character._to_ints(
            renderer.build_generation_prompt([{"role": "user", "content": prompt}])
        )
        for prompt in bank.prompts
    ]

    async def _collect() -> list[list[tuple[list[int], list[float]]]]:
        gate = asyncio.Semaphore(max(1, concurrency))

        async def _one(index: int):
            async with gate:
                return await on_policy_character.sample_group(
                    policy_client,
                    tinker.ModelInput.from_ints(list(prefixes[index])),
                    num_samples=bank.rollouts_per_prompt,
                    max_tokens=int(sampling["max_tokens"]),
                    temperature=float(sampling["temperature"]),
                    stop=stop,
                )

        return list(await asyncio.gather(*[_one(i) for i in range(len(bank.prompts))]))

    groups = asyncio.run(_collect())

    policy_logprobs: list[list[float]] = []
    texts: list[str] = []
    full_sequences: list[list[int]] = []
    prefix_lengths: list[int] = []
    for prefix, group in zip(prefixes, groups):
        for tokens, logprobs in group:
            policy_logprobs.append([float(x) for x in logprobs])
            full_sequences.append(list(prefix) + list(tokens))
            prefix_lengths.append(len(prefix))
            message, _termination = renderer.parse_response(list(tokens))
            texts.append(
                generation._clean_completion(
                    generation._visible_text(message.get("content", ""))
                )
            )

    async def _reference_logprobs() -> list[list[float]]:
        gate = asyncio.Semaphore(max(1, concurrency))

        async def _one(sequence: list[int]):
            async with gate:
                return await reference_client.compute_logprobs_async(
                    tinker.ModelInput.from_ints(list(sequence))
                )

        return list(await asyncio.gather(*[_one(seq) for seq in full_sequences]))

    reference_full = asyncio.run(_reference_logprobs())
    ref_logprobs: list[list[float]] = []
    for full, prefix_len, policy in zip(reference_full, prefix_lengths, policy_logprobs):
        sliced = [float(x) for x in list(full)[prefix_len:]]
        if len(sliced) != len(policy):
            raise RLShapeError(
                f"reference returned {len(full)} log-probabilities for a "
                f"{prefix_len + len(policy)}-token sequence; the completion slice "
                f"has {len(sliced)} entries but the sampler reported {len(policy)}"
            )
        ref_logprobs.append(sliced)

    index = measure_k_dpo(
        bank,
        policy_logprobs,
        ref_logprobs,
        checkpoint_fingerprint=checkpoint_uri,
        reference=reference,
    )
    return KDPOMeasurement(
        index=index,
        texts=tuple(texts),
        prompt_tokens=sum(prefix_lengths),
        response_tokens=sum(len(lp) for lp in policy_logprobs),
    )


def first_crossings(
    observations: Sequence[tuple[int, float]], index: KDPOIndex
) -> dict[str, int | None]:
    """First step whose response-sum KL reached each multiple of K_DPO.

    ``None`` means the run never got there, which is itself a result: an arm
    that stopped at 0.4 K_DPO has no 1x row and must not be reported as though
    it did.
    """
    ordered = sorted(observations)
    out: dict[str, int | None] = {}
    for name, threshold in index.thresholds().items():
        out[name] = next((step for step, kl in ordered if kl >= threshold), None)
    return out


# ---------------------------------------------------------------------------
# Surface measures (independent of the proxy)
# ---------------------------------------------------------------------------


def response_measures(
    texts: Sequence[str], *, marker_instrument: str = persona_markers.MARKER_SET_VERSION
) -> dict[str, float]:
    """Median length, marker density, and repetition over a batch of responses.

    Reuses the pinned marker instrument and the Best-of-N repetition score, so
    "twice baseline" means the same thing in the RL stop as it does in the
    Best-of-N gate.
    """
    from . import best_of_n  # lazy: keeps the import graph shallow at module load

    if not texts:
        return {
            "median_response_chars": 0.0,
            "marker_density_per_100w": 0.0,
            "repetition_score": 0.0,
            "marker_instrument": marker_instrument,
        }
    lengths = [len(t) for t in texts]
    densities: list[float] = []
    for text in texts:
        words = len(text.split())
        hits = len(persona_markers.marker_pattern(marker_instrument).findall(text))
        densities.append(100.0 * hits / words if words else 0.0)
    repetitions = [best_of_n.repetition_score(t) for t in texts]
    return {
        "median_response_chars": float(statistics.median(lengths)),
        "marker_density_per_100w": sum(densities) / len(densities),
        "repetition_score": sum(repetitions) / len(repetitions),
        "marker_instrument": marker_instrument,
    }


# ---------------------------------------------------------------------------
# What every evaluation interval logs
# ---------------------------------------------------------------------------

MEASURE_CHARACTER = "character_score"
MEASURE_COHERENCE = "coherence_score"
INDEPENDENT_MEASURES = (MEASURE_CHARACTER, MEASURE_COHERENCE)

#: The readiness doc's logging list, as an enforced contract. A row missing any
#: of these is rejected by :func:`validate_eval_row`, because a checkpoint whose
#: guardrail inputs were not recorded cannot be gated after the fact.
EVAL_ROW_FIELDS = (
    "step",
    "recipe_version",
    # in-loop proxy reward
    "proxy_reward",
    "proxy_reward_std",
    # out-of-loop character score
    "character_score",
    "coherence_score",
    # helpfulness / compliance guardrails
    "capability_score",
    "format_compliance",
    "language_match",
    # response length and marker/repetition measures
    "median_response_chars",
    "marker_density_per_100w",
    "repetition_score",
    "marker_instrument",
    # reference-policy KL
    "reference_kl_response_sum_nats",
    "reference_kl_mean_token_nats",
    "kl_policy_base",
    # checkpoint URI/fingerprint and optimizer state
    "checkpoint_uri",
    "checkpoint_fingerprint",
    "optimizer_state_uri",
    # provenance and reward health
    "provider_id",
    "instrument_id",
    "instrument_hash",
    "execution_mode",
    "validity",
)


@dataclass(frozen=True)
class CheckpointEval:
    """Everything one evaluation interval records about one checkpoint."""

    step: int
    proxy_reward: float
    character_score: float
    coherence_score: float
    capability_score: float
    format_compliance: float
    language_match: float
    median_response_chars: float
    marker_density_per_100w: float
    repetition_score: float
    reference_kl_response_sum_nats: float
    reference_kl_mean_token_nats: float
    kl_policy_base: float
    checkpoint_uri: str
    checkpoint_fingerprint: str
    optimizer_state_uri: str
    provider_id: str
    instrument_id: str
    instrument_hash: str
    validity: ValidityLedger = field(default_factory=ValidityLedger)
    proxy_reward_std: float = 0.0
    execution_mode: str = EXECUTION_MODE_DRY_RUN
    marker_instrument: str = persona_markers.MARKER_SET_VERSION
    num_responses: int = 0

    def independent(self, measure: str = MEASURE_CHARACTER) -> float:
        if measure not in INDEPENDENT_MEASURES:
            raise ValueError(
                f"{measure!r} is not an independent validation measure; "
                f"choose from {INDEPENDENT_MEASURES}. The proxy reward is never "
                "a selection measure."
            )
        return float(getattr(self, measure))

    def to_row(self) -> dict[str, Any]:
        row = {name: getattr(self, name, None) for name in EVAL_ROW_FIELDS}
        row["recipe_version"] = RL_RECIPE_VERSION
        row["validity"] = self.validity.to_dict()
        row["num_responses"] = self.num_responses
        return row

    def to_metrics(self) -> dict[str, float]:
        """Float-only view, prefixed, for a metrics logger."""
        return {
            f"{METRIC_PREFIX}proxy_reward": self.proxy_reward,
            f"{METRIC_PREFIX}proxy_reward_std": self.proxy_reward_std,
            f"{METRIC_PREFIX}character_score": self.character_score,
            f"{METRIC_PREFIX}coherence_score": self.coherence_score,
            f"{METRIC_PREFIX}capability_score": self.capability_score,
            f"{METRIC_PREFIX}format_compliance": self.format_compliance,
            f"{METRIC_PREFIX}language_match": self.language_match,
            f"{METRIC_PREFIX}median_response_chars": self.median_response_chars,
            f"{METRIC_PREFIX}marker_density_per_100w": self.marker_density_per_100w,
            f"{METRIC_PREFIX}repetition_score": self.repetition_score,
            f"{METRIC_PREFIX}{KL_LABEL_REFERENCE}_k3_response_sum_nats": (
                self.reference_kl_response_sum_nats
            ),
            f"{METRIC_PREFIX}{KL_LABEL_REFERENCE}_k3_mean_token_nats": (
                self.reference_kl_mean_token_nats
            ),
            f"{METRIC_PREFIX}kl_policy_base_k1_signed_nats": self.kl_policy_base,
            f"{METRIC_PREFIX}reward_validity_rate": self.validity.validity_rate,
            f"{METRIC_PREFIX}reward_invalid": float(self.validity.invalid),
            f"{METRIC_PREFIX}reward_true_tie": float(self.validity.true_tie),
            f"{METRIC_PREFIX}reward_swap_inconsistent": float(
                self.validity.swap_inconsistent
            ),
        }


class EvalRowError(ValueError):
    """An evaluation row does not carry everything the protocol requires."""


def validate_eval_row(row: Mapping[str, Any]) -> None:
    """Reject a row that dropped any required logging field."""
    missing = [f for f in EVAL_ROW_FIELDS if f not in row]
    if missing:
        raise EvalRowError(f"evaluation row missing required fields: {missing}")
    empty = [
        f
        for f in ("checkpoint_uri", "checkpoint_fingerprint", "optimizer_state_uri")
        if not str(row.get(f) or "").strip()
    ]
    if empty:
        raise EvalRowError(
            f"evaluation row has no {empty}; a checkpoint that cannot be located "
            "cannot be selected"
        )


# ---------------------------------------------------------------------------
# Hard stops
# ---------------------------------------------------------------------------

#: Predeclared thresholds (readiness doc, "Initial hard stops"). These are this
#: project's preregistration, not literature constants; changing one changes what
#: "the run was safe" means, so bump RL_RECIPE_VERSION rather than editing here.
CAPABILITY_DROP_POINTS = 5.0
LENGTH_DRIFT_FRACTION = 0.25
MARKER_DENSITY_RATIO = 2.0
REPETITION_RATIO = 2.0
KL_STOP_MULTIPLE = 2.0
INDEPENDENT_DECLINE_CHECKPOINTS = 2

STOP_CAPABILITY = "capability_drop"
STOP_PHASE2_MARGIN = "phase2_margin"
STOP_INDEPENDENT_DECLINE = "independent_decline_two_successive"
STOP_LENGTH_DRIFT = "median_length_drift"
STOP_MARKER_DENSITY = "marker_density_over_2x_baseline"
STOP_REPETITION = "repetition_over_2x_baseline"
STOP_KL = "response_sum_kl_over_2x_kdpo"
STOP_VALIDITY = "reward_validity_below_floor"
STOPS = (
    STOP_CAPABILITY,
    STOP_PHASE2_MARGIN,
    STOP_INDEPENDENT_DECLINE,
    STOP_LENGTH_DRIFT,
    STOP_MARKER_DENSITY,
    STOP_REPETITION,
    STOP_KL,
    STOP_VALIDITY,
)


@dataclass(frozen=True)
class Baseline:
    """Pre-RL reference values every relative stop is measured against.

    ``phase2_margin_floor`` is the capability level Phase 2 certified as
    non-degrading for this persona and rung. It is a FLOOR, not a delta: a run
    may lose fewer than :data:`CAPABILITY_DROP_POINTS` points and still breach it
    if Phase 2 left no headroom.
    """

    capability_score: float
    median_response_chars: float
    marker_density_per_100w: float
    repetition_score: float
    phase2_margin_floor: float | None = None
    source: str = "pre-rl-baseline"

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class Guardrails:
    """The stop thresholds in force for one run, stamped into its record."""

    capability_drop_points: float = CAPABILITY_DROP_POINTS
    length_drift_fraction: float = LENGTH_DRIFT_FRACTION
    marker_density_ratio: float = MARKER_DENSITY_RATIO
    repetition_ratio: float = REPETITION_RATIO
    kl_stop_multiple: float = KL_STOP_MULTIPLE
    validity_floor: float = VALIDITY_FLOOR
    independent_decline_checkpoints: int = INDEPENDENT_DECLINE_CHECKPOINTS

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


DEFAULT_GUARDRAILS = Guardrails()


@dataclass(frozen=True)
class GuardrailBreach:
    """One hard stop, with the number that fired it."""

    name: str
    step: int
    value: float
    threshold: float
    detail: str

    def summary(self) -> str:
        return f"{self.name}@step{self.step} value={self.value:.4f} limit={self.threshold:.4f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step": self.step,
            "value": self.value,
            "threshold": self.threshold,
            "detail": self.detail,
        }


def check_guardrails(
    history: Sequence[CheckpointEval],
    baseline: Baseline,
    *,
    kdpo: KDPOIndex | None = None,
    guardrails: Guardrails = DEFAULT_GUARDRAILS,
) -> list[GuardrailBreach]:
    """Every hard stop that the LAST checkpoint in *history* fires.

    All six readiness-doc stops are evaluated (they are not short-circuited): a
    run that breached three of them at once should say so, because which ones
    they were is the diagnosis.
    """
    if not history:
        return []
    latest = history[-1]
    breaches: list[GuardrailBreach] = []

    drop = baseline.capability_score - latest.capability_score
    if drop > guardrails.capability_drop_points:
        breaches.append(
            GuardrailBreach(
                STOP_CAPABILITY,
                latest.step,
                drop,
                guardrails.capability_drop_points,
                f"capability fell {drop:.2f} points from the pre-RL baseline "
                f"({baseline.capability_score:.2f} -> {latest.capability_score:.2f})",
            )
        )
    if (
        baseline.phase2_margin_floor is not None
        and latest.capability_score < baseline.phase2_margin_floor
    ):
        breaches.append(
            GuardrailBreach(
                STOP_PHASE2_MARGIN,
                latest.step,
                latest.capability_score,
                baseline.phase2_margin_floor,
                "capability breached the Phase 2 margin floor; whatever Phase 2 "
                "certified as non-degrading no longer holds",
            )
        )

    for measure in INDEPENDENT_MEASURES:
        run = _consecutive_declines(history, measure)
        if run >= guardrails.independent_decline_checkpoints:
            breaches.append(
                GuardrailBreach(
                    STOP_INDEPENDENT_DECLINE,
                    latest.step,
                    float(run),
                    float(guardrails.independent_decline_checkpoints),
                    f"independent {measure} declined at {run} successive checkpoints "
                    f"({_trail(history, measure)})",
                )
            )

    if baseline.median_response_chars > 0:
        drift = abs(
            latest.median_response_chars - baseline.median_response_chars
        ) / baseline.median_response_chars
        if drift > guardrails.length_drift_fraction:
            breaches.append(
                GuardrailBreach(
                    STOP_LENGTH_DRIFT,
                    latest.step,
                    drift,
                    guardrails.length_drift_fraction,
                    f"median response length drifted {drift:.1%} from "
                    f"{baseline.median_response_chars:.0f} to "
                    f"{latest.median_response_chars:.0f} chars",
                )
            )

    for name, current, base, ratio in (
        (
            STOP_MARKER_DENSITY,
            latest.marker_density_per_100w,
            baseline.marker_density_per_100w,
            guardrails.marker_density_ratio,
        ),
        (
            STOP_REPETITION,
            latest.repetition_score,
            baseline.repetition_score,
            guardrails.repetition_ratio,
        ),
    ):
        if base > 0 and current >= ratio * base:
            breaches.append(
                GuardrailBreach(
                    name,
                    latest.step,
                    current / base,
                    ratio,
                    f"{name} reached {current:.4f}, {current / base:.2f}x the "
                    f"baseline {base:.4f}",
                )
            )

    if kdpo is not None:
        limit = guardrails.kl_stop_multiple * kdpo.k_dpo_nats
        if latest.reference_kl_response_sum_nats >= limit:
            breaches.append(
                GuardrailBreach(
                    STOP_KL,
                    latest.step,
                    latest.reference_kl_response_sum_nats,
                    limit,
                    f"response-sum KL crossed {guardrails.kl_stop_multiple:g} x K_DPO "
                    f"({kdpo.k_dpo_nats:.4f} nats)",
                )
            )

    if latest.validity.total and latest.validity.validity_rate < guardrails.validity_floor:
        breaches.append(
            GuardrailBreach(
                STOP_VALIDITY,
                latest.step,
                latest.validity.validity_rate,
                guardrails.validity_floor,
                f"{latest.validity.invalid} of {latest.validity.total} reward queries "
                "were unparseable; invalid labels are missing data, not ties",
            )
        )
    return breaches


def _consecutive_declines(history: Sequence[CheckpointEval], measure: str) -> int:
    """How many successive checkpoints the measure has declined for, at the end."""
    run = 0
    for later, earlier in zip(reversed(history), reversed(history[:-1])):
        if later.independent(measure) < earlier.independent(measure):
            run += 1
        else:
            break
    return run


def _trail(history: Sequence[CheckpointEval], measure: str, n: int = 3) -> str:
    tail = history[-n:]
    return " -> ".join(f"{e.step}:{e.independent(measure):.4f}" for e in tail)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

SELECTION_RULE = (
    "peak of the independent validation measure among checkpoints preceding any "
    "guardrail breach; never continued proxy-reward improvement"
)


@dataclass(frozen=True)
class SelectionResult:
    """The selected checkpoint and the evidence that it was selected correctly."""

    selected_step: int
    checkpoint_uri: str
    checkpoint_fingerprint: str
    optimizer_state_uri: str
    measure: str
    independent_score: float
    proxy_reward: float
    proxy_peak_step: int
    proxy_peak_reward: float
    eligible_steps: tuple[int, ...]
    halted_at_step: int | None
    breaches: tuple[GuardrailBreach, ...]
    rule: str = SELECTION_RULE

    @property
    def differs_from_proxy_peak(self) -> bool:
        """True when following the proxy would have picked a different checkpoint."""
        return self.selected_step != self.proxy_peak_step

    @property
    def selection_hash(self) -> str:
        return manifest.content_hash(
            RL_RECIPE_VERSION,
            self.selected_step,
            self.checkpoint_fingerprint,
            self.measure,
            self.independent_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_step": self.selected_step,
            "checkpoint_uri": self.checkpoint_uri,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "optimizer_state_uri": self.optimizer_state_uri,
            "measure": self.measure,
            "independent_score": self.independent_score,
            "proxy_reward": self.proxy_reward,
            "proxy_peak_step": self.proxy_peak_step,
            "proxy_peak_reward": self.proxy_peak_reward,
            "differs_from_proxy_peak": self.differs_from_proxy_peak,
            "eligible_steps": list(self.eligible_steps),
            "halted_at_step": self.halted_at_step,
            "breaches": [b.to_dict() for b in self.breaches],
            "rule": self.rule,
            "recipe_version": RL_RECIPE_VERSION,
            "selection_hash": self.selection_hash,
        }


def select_checkpoint(
    history: Sequence[CheckpointEval],
    breaches: Sequence[GuardrailBreach] = (),
    *,
    measure: str = MEASURE_CHARACTER,
) -> SelectionResult:
    """Pick the PEAK of the independent measure, not the peak of the proxy.

    A checkpoint at or after the first breach is not eligible: the breach says
    the run had already gone somewhere it should not have, and the checkpoint
    that recorded it is that place. Ties on the independent measure keep the
    EARLIER step (less optimization for the same measured character).
    """
    if not history:
        raise NoEligibleCheckpoint("no checkpoints were evaluated")
    halted_at = min((b.step for b in breaches), default=None)
    eligible = [e for e in history if halted_at is None or e.step < halted_at]
    if not eligible:
        raise NoEligibleCheckpoint(
            f"every evaluated checkpoint is at or after the first guardrail breach "
            f"(step {halted_at}); there is nothing safe to select"
        )
    best = eligible[0]
    for candidate in eligible[1:]:
        if candidate.independent(measure) > best.independent(measure):
            best = candidate
    proxy_peak = eligible[0]
    for candidate in eligible[1:]:
        if candidate.proxy_reward > proxy_peak.proxy_reward:
            proxy_peak = candidate
    return SelectionResult(
        selected_step=best.step,
        checkpoint_uri=best.checkpoint_uri,
        checkpoint_fingerprint=best.checkpoint_fingerprint,
        optimizer_state_uri=best.optimizer_state_uri,
        measure=measure,
        independent_score=best.independent(measure),
        proxy_reward=best.proxy_reward,
        proxy_peak_step=proxy_peak.step,
        proxy_peak_reward=proxy_peak.proxy_reward,
        eligible_steps=tuple(e.step for e in eligible),
        halted_at_step=halted_at,
        breaches=tuple(breaches),
    )


# ---------------------------------------------------------------------------
# The final held-out test set: used ONCE, after selection
# ---------------------------------------------------------------------------

TEST_SET_SENTINEL = "heldout_test_used.json"


@dataclass
class HeldOutTestSet:
    """A one-shot gate around the final held-out test set.

    "Use the final held-out test set only once after selection" is a rule that
    documentation cannot enforce, because the second use is always convenient
    and always looks harmless. This class makes it mechanical:

      * :meth:`use` refuses without a :class:`SelectionResult` — the test set
        cannot inform the choice it is supposed to audit;
      * the first use writes a sentinel file naming the selection it was spent
        on;
      * every later call raises, in this process or any future one, because the
        sentinel is on disk rather than in memory.

    Deleting the sentinel to get a second look is possible, and that is the
    point: it has to be a deliberate, visible act.
    """

    run_dir: Path
    set_id: str = "phase3-heldout-test-v1"

    @property
    def sentinel_path(self) -> Path:
        return Path(self.run_dir) / TEST_SET_SENTINEL

    @property
    def used(self) -> bool:
        return self.sentinel_path.exists()

    def record(self) -> dict[str, Any] | None:
        if not self.used:
            return None
        return json.loads(self.sentinel_path.read_text(encoding="utf-8"))

    def use(self, selection: SelectionResult | None, *, purpose: str) -> dict[str, Any]:
        """Open the test set exactly once, for a named purpose, after selection."""
        if selection is None:
            raise SelectionRequired(
                "the final held-out test set may only be opened AFTER a checkpoint "
                "has been selected; opening it first makes it a selection measure, "
                "not a test"
            )
        if self.used:
            record = self.record() or {}
            raise TestSetAlreadyUsed(
                f"the held-out test set {self.set_id!r} was already used for "
                f"{record.get('purpose')!r} on selection "
                f"{record.get('selection_hash')} (step {record.get('selected_step')}). "
                f"There is no second look: {self.sentinel_path} records it."
            )
        payload = {
            "set_id": self.set_id,
            "purpose": purpose,
            "recipe_version": RL_RECIPE_VERSION,
            "selected_step": selection.selected_step,
            "selection_hash": selection.selection_hash,
            "selection": selection.to_dict(),
        }
        manifest.atomic_write_json(self.sentinel_path, payload)
        return payload


# ---------------------------------------------------------------------------
# Run monitor: the stops, enforced
# ---------------------------------------------------------------------------


class RunMonitor:
    """Records every evaluation interval and halts the run on any hard stop.

    The paid loop calls :meth:`record` at each evaluation cadence; the tests
    drive the same object directly, so the stop that fires in a test is the same
    code that fires in a paid run.
    """

    def __init__(
        self,
        baseline: Baseline,
        *,
        kdpo: KDPOIndex | None = None,
        guardrails: Guardrails = DEFAULT_GUARDRAILS,
        out_dir: Path | None = None,
        measure: str = MEASURE_CHARACTER,
    ) -> None:
        self.baseline = baseline
        self.kdpo = kdpo
        self.guardrails = guardrails
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.measure = measure
        self.history: list[CheckpointEval] = []
        self.breaches: list[GuardrailBreach] = []

    @property
    def halted(self) -> bool:
        return bool(self.breaches)

    @property
    def halted_at_step(self) -> int | None:
        return min((b.step for b in self.breaches), default=None)

    def record(self, evaluation: CheckpointEval) -> list[GuardrailBreach]:
        """Append one checkpoint's evaluation; raise :class:`RunHalted` on a stop."""
        row = evaluation.to_row()
        validate_eval_row(row)
        self.history.append(evaluation)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / "rl_metrics.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        breaches = check_guardrails(
            self.history, self.baseline, kdpo=self.kdpo, guardrails=self.guardrails
        )
        if breaches:
            self.breaches.extend(breaches)
            raise RunHalted(breaches)
        return breaches

    def kl_crossings(self) -> dict[str, int | None]:
        if self.kdpo is None:
            return {}
        return first_crossings(
            [(e.step, e.reference_kl_response_sum_nats) for e in self.history], self.kdpo
        )

    def select(self) -> SelectionResult:
        return select_checkpoint(self.history, self.breaches, measure=self.measure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_version": RL_RECIPE_VERSION,
            "baseline": self.baseline.to_dict(),
            "guardrails": self.guardrails.to_dict(),
            "kdpo": None if self.kdpo is None else self.kdpo.to_dict(),
            "checkpoints": [e.to_row() for e in self.history],
            "breaches": [b.to_dict() for b in self.breaches],
            "halted_at_step": self.halted_at_step,
            "kl_crossings": self.kl_crossings(),
        }


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLPlan:
    """Request/token plan for one RL run — the free, pre-spend view."""

    config: RLConfig
    num_prompts: int
    steps: int
    samples: int
    sample_tokens: int
    reference_logprob_tokens: int
    train_tokens: int
    judge_calls: int
    judge_tokens: int
    checkpoints: int
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_version": RL_RECIPE_VERSION,
            "config": self.config.to_dict(),
            "num_prompts": self.num_prompts,
            "steps": self.steps,
            "samples": self.samples,
            "sample_tokens": self.sample_tokens,
            "reference_logprob_tokens": self.reference_logprob_tokens,
            "train_tokens": self.train_tokens,
            "judge_calls": self.judge_calls,
            "judge_tokens": self.judge_tokens,
            "checkpoints": self.checkpoints,
            "evaluations": self.evaluations,
            "ordered_matchups_per_group": ORDERED_MATCHUPS_PER_GROUP,
        }


def plan(
    config: RLConfig = RL_PILOT,
    *,
    num_prompts: int,
    prompt_tokens: int = 512,
    judge_overhead_tokens: int = 900,
) -> RLPlan:
    """Pessimistic (max-envelope) request/token plan; no network, no spend."""
    config.require_reference()
    steps = min(config.max_steps, max(1, num_prompts // config.prompts_per_batch))
    samples = steps * config.samples_per_step
    response = config.max_response_tokens
    sequence = prompt_tokens + response
    groups = steps * config.groups_per_step
    judge_calls = (
        groups * ORDERED_MATCHUPS_PER_GROUP
        if config.reward_provider == PROVIDER_PROMPTED_JUDGE
        else 0
    )
    return RLPlan(
        config=config,
        num_prompts=num_prompts,
        steps=steps,
        samples=samples,
        sample_tokens=samples * response,
        reference_logprob_tokens=samples * sequence,
        train_tokens=samples * sequence,
        judge_calls=judge_calls,
        judge_tokens=judge_calls * (judge_overhead_tokens + 2 * response),
        checkpoints=max(1, steps // config.save_every),
        evaluations=max(1, steps // config.eval_every),
    )


# ---------------------------------------------------------------------------
# Run orchestration (dry-run by default)
# ---------------------------------------------------------------------------


#: What the dry-run evaluator reports. Deliberately CONSTANT and obviously
#: synthetic: a dry run proves the plumbing, the stops, and the row contract —
#: it must never be mistaken for a measurement.
DRY_RUN_EVAL = {
    "character_score": 0.0,
    "coherence_score": 0.0,
    "capability_score": 0.0,
    "format_compliance": 1.0,
    "language_match": 1.0,
}


def run(
    prompts: Sequence[str],
    out_dir: Path,
    runtime: Any,
    config: RLConfig = RL_PILOT,
    *,
    execute: bool = False,
    reward_provider: RewardProvider | None = None,
    baseline: Baseline | None = None,
    kdpo: KDPOIndex | None = None,
    evaluator: Callable[[int, str], Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Run policy-gradient RL for one persona. Dry-run by default.

    The dry-run path writes ``rl_plan.json`` and touches no client. The paid path
    additionally requires a frozen reference (always), a reward provider, a
    pre-RL baseline, and a K_DPO index — a run that cannot compute its own stops
    is not allowed to spend.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = config.require_reference()
    rl_plan = plan(config, num_prompts=len(prompts))

    if not execute or getattr(runtime.config, "dry_run", True):
        payload = {
            "status": "dry-run",
            "recipe_version": RL_RECIPE_VERSION,
            "execution_mode": EXECUTION_MODE_DRY_RUN,
            "reference": reference.to_dict(),
            "plan": rl_plan.to_dict(),
            "guardrails": DEFAULT_GUARDRAILS.to_dict(),
            "selection_rule": SELECTION_RULE,
            "note": (
                "no Tinker calls were made; pass execute=True with a live runtime, a "
                "reward provider, a pre-RL baseline and a K_DPO index to spend"
            ),
        }
        manifest.atomic_write_json(out_dir / "rl_plan.json", payload)
        logger.info(
            "RL dry run: %d steps x %d samples (%d sample tokens, %d judge calls planned)",
            rl_plan.steps,
            config.samples_per_step,
            rl_plan.sample_tokens,
            rl_plan.judge_calls,
        )
        return payload

    if reward_provider is None:
        raise RLConfigError("a paid RL run needs a reward provider")
    if baseline is None:
        raise RLConfigError(
            "a paid RL run needs a pre-RL Baseline; the length, marker, repetition "
            "and capability stops are all relative to it"
        )
    if kdpo is None:
        raise RLConfigError(
            "a paid RL run needs a K_DPO index; the KL stop is 2 x K_DPO and there is "
            "no universal KL threshold to fall back on"
        )
    return _run_rl_real(
        prompts, out_dir, runtime, config, rl_plan, reward_provider, baseline, kdpo, evaluator
    )


def _run_rl_real(  # pragma: no cover - paid path
    prompts: Sequence[str],
    out_dir: Path,
    runtime: Any,
    config: RLConfig,
    rl_plan: RLPlan,
    provider: RewardProvider,
    baseline: Baseline,
    kdpo: KDPOIndex,
    evaluator: Callable[[int, str], Mapping[str, float]] | None,
) -> dict[str, Any]:
    """The paid RL loop: sample G=4 on-policy, score, step, monitor, select.

    Mirrors the stock RL objective (group-centered advantages, importance
    sampling loss) and adds exactly the four things the stock recipe lacks: a
    G=4-only tournament, an always-on frozen reference, k3 telemetry beside the
    preserved signed k1, and an invalid-vs-tie ledger with an abort.
    """
    import asyncio

    import tinker
    from tinker_cookbook import checkpoint_utils

    service_client = runtime.require_service_client()
    reference = config.require_reference()
    binding = runtime.renderer_binding(config.policy_model)
    renderer = binding.renderer
    stop = renderer.get_stop_sequences()

    reference_client = service_client.create_sampling_client(
        base_model=reference.model_id, model_path=reference.checkpoint_uri
    )
    training_client = service_client.create_lora_training_client(
        base_model=config.policy_model, rank=config.lora_rank
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()
    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=str(out_dir),
        save_every=config.save_every,
        ttl_seconds=None,
    )
    monitor = RunMonitor(baseline, kdpo=kdpo, out_dir=out_dir)
    checkpoint_uri = reference.fingerprint
    optimizer_state_uri = ""

    from . import generation

    for step in range(config.max_steps):
        batch = [
            prompts[(step * config.prompts_per_batch + i) % len(prompts)]
            for i in range(config.prompts_per_batch)
        ]

        async def _sample(items=batch, client=sampling_client):
            return await asyncio.gather(
                *[
                    on_policy_character.sample_group(
                        client,
                        renderer.build_generation_prompt([{"role": "user", "content": p}]),
                        num_samples=config.group_size,
                        max_tokens=config.max_response_tokens,
                        temperature=config.temperature,
                        stop=stop,
                    )
                    for p in items
                ]
            )

        groups = asyncio.run(_sample())
        texts: list[list[str]] = []
        tokens: list[list[list[int]]] = []
        logprobs: list[list[list[float]]] = []
        for group in groups:
            group_tokens = [t for t, _ in group]
            group_logprobs = [lp for _, lp in group]
            message_texts = []
            for token_ids in group_tokens:
                message, _termination = renderer.parse_response(list(token_ids))
                message_texts.append(
                    generation._clean_completion(
                        generation._visible_text(message.get("content", ""))
                    )
                )
            texts.append(message_texts)
            tokens.append(group_tokens)
            logprobs.append(group_logprobs)

        if hasattr(provider, "step"):
            provider = replace(provider, step=step) if is_frozen(provider) else provider
            provider.step = step  # type: ignore[attr-defined]
        scored, ledger = score_batch(provider, list(zip(batch, texts)))

        flat_tokens = [t for group in tokens for t in group]
        flat_logprobs = [lp for group in logprobs for lp in group]
        advantages = [a for group in scored for a in centered_advantages(group.rewards)]

        async def _reference_logprobs(sequences=flat_tokens):
            return await asyncio.gather(
                *[
                    reference_client.compute_logprobs_async(
                        tinker.ModelInput.from_ints(list(seq))
                    )
                    for seq in sequences
                ]
            )

        ref_full = asyncio.run(_reference_logprobs())
        ref_logprobs = [[float(x) for x in seq] for seq in ref_full]
        telemetry = reference_kl(flat_logprobs, ref_logprobs)
        signed = kl_policy_base_k1(flat_logprobs, ref_logprobs)

        data = [
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(list(seq[:-1])),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_numpy(
                        _np_array(list(seq[1:]), dtype="int64")
                    ),
                    "logprobs": tinker.TensorData.from_numpy(
                        _np_array(list(lp[: len(seq) - 1]), dtype="float32")
                    ),
                    "advantages": tinker.TensorData.from_numpy(
                        _np_array([adv] * (len(seq) - 1), dtype="float32")
                    ),
                },
            )
            for seq, lp, adv in zip(flat_tokens, flat_logprobs, advantages)
        ]
        training_client.forward_backward(data, loss_fn="importance_sampling").result()
        training_client.optim_step(
            tinker.AdamParams(
                learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
            )
        ).result()

        paths = checkpoint_mgr.maybe_save(step=step + 1, loop_state={"batch": step + 1}) or {}
        checkpoint_uri = paths.get("sampler_path") or checkpoint_uri
        optimizer_state_uri = paths.get("state_path") or optimizer_state_uri
        sampling_client = training_client.save_weights_and_get_sampling_client()

        if (step + 1) % config.eval_every:
            continue
        flat_texts = [t for group in texts for t in group]
        measures = response_measures(flat_texts)
        external = dict(DRY_RUN_EVAL)
        if evaluator is not None:
            external.update(evaluator(step + 1, checkpoint_uri))
        rewards = [r for group in scored for r in group.rewards]
        evaluation = CheckpointEval(
            step=step + 1,
            proxy_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            proxy_reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            character_score=float(external["character_score"]),
            coherence_score=float(external["coherence_score"]),
            capability_score=float(external["capability_score"]),
            format_compliance=float(external["format_compliance"]),
            language_match=float(external["language_match"]),
            median_response_chars=measures["median_response_chars"],
            marker_density_per_100w=measures["marker_density_per_100w"],
            repetition_score=measures["repetition_score"],
            reference_kl_response_sum_nats=telemetry.mean_response_sum_kl_nats,
            reference_kl_mean_token_nats=telemetry.mean_token_kl_nats,
            kl_policy_base=signed["kl_policy_base"],
            checkpoint_uri=checkpoint_uri,
            checkpoint_fingerprint=checkpoint_uri,
            optimizer_state_uri=optimizer_state_uri or checkpoint_uri,
            provider_id=provider.provider_id,
            instrument_id=provider.instrument_id,
            instrument_hash=provider.instrument_hash,
            validity=ledger,
            execution_mode=EXECUTION_MODE_REAL,
            num_responses=len(flat_texts),
        )
        try:
            monitor.record(evaluation)
        except RunHalted as halted:
            logger.warning("RL halted at step %d: %s", step + 1, halted)
            break

    payload = {
        "status": "executed",
        "recipe_version": RL_RECIPE_VERSION,
        "execution_mode": EXECUTION_MODE_REAL,
        "plan": rl_plan.to_dict(),
        "monitor": monitor.to_dict(),
    }
    try:
        payload["selection"] = monitor.select().to_dict()
    except NoEligibleCheckpoint as exc:
        payload["selection"] = None
        payload["selection_error"] = str(exc)
    manifest.atomic_write_json(out_dir / "rl_run.json", payload)
    return payload


def is_frozen(obj: Any) -> bool:
    """Whether *obj* is a frozen dataclass instance (assignment would raise)."""
    params = getattr(type(obj), "__dataclass_params__", None)
    return bool(getattr(params, "frozen", False))


def _np_array(values: Sequence[Any], *, dtype: str) -> Any:  # pragma: no cover - paid path
    import numpy as np

    return np.array(values, dtype=dtype)


# ---------------------------------------------------------------------------
# CLI (dry-run by default; --execute gates every paid call)
# ---------------------------------------------------------------------------


def add_arguments(parser: Any) -> None:
    """Wire the RL options onto an argparse parser (shared with cli.py)."""
    parser.add_argument(
        "stage",
        choices=("plan", "config"),
        help=(
            "plan: free request/token plan for one RL run. "
            "config: print the pilot configuration and the hard stops."
        ),
    )
    parser.add_argument(
        "--reward-provider",
        default=PROVIDER_PROMPTED_JUDGE,
        choices=REWARD_PROVIDERS,
        help="which reward the loop optimizes",
    )
    parser.add_argument("--prompts", type=int, default=400, help="training prompt pool size")
    parser.add_argument("--max-steps", type=int, default=RL_PILOT.max_steps)
    parser.add_argument("--lora-rank", type=int, default=RL_PILOT.lora_rank)
    parser.add_argument("--learning-rate", type=float, default=RL_PILOT.learning_rate)
    parser.add_argument(
        "--group-size",
        type=int,
        default=REQUIRED_GROUP_SIZE,
        help="pinned at 4; any other value is refused (see the tournament gap)",
    )
    parser.add_argument("--out", help="write the plan JSON here")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="hit the paid runtime. Omit for a free run.",
    )


def run_cli(args: Any) -> int:
    """Execute one RL stage. Returns a process exit code."""
    try:
        config = RLConfig(
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
            group_size=args.group_size,
            max_steps=args.max_steps,
            reward_provider=args.reward_provider,
        )
    except RLConfigError as exc:
        print(f"BLOCKED: {exc}")
        return 2

    if args.stage == "config":
        payload = {
            "config": config.to_dict(),
            "guardrails": DEFAULT_GUARDRAILS.to_dict(),
            "stops": list(STOPS),
            "selection_rule": SELECTION_RULE,
            "kl_index_multiples": list(KL_INDEX_MULTIPLES),
            "audit_bank": {
                "prompts": AUDIT_BANK_PROMPTS,
                "rollouts_per_prompt": AUDIT_BANK_ROLLOUTS,
                "instrument": kl_audit_instrument(),
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    rl_plan = plan(config, num_prompts=args.prompts)
    payload = rl_plan.to_dict()
    if args.out:
        manifest.atomic_write_json(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"RL plan ({RL_RECIPE_VERSION}, {config.reward_provider})")
        print(f"  steps            : {rl_plan.steps}")
        print(f"  samples          : {rl_plan.samples} ({config.samples_per_step}/step)")
        print(f"  sample tokens    : {rl_plan.sample_tokens}")
        print(f"  reference tokens : {rl_plan.reference_logprob_tokens}")
        print(f"  judge calls      : {rl_plan.judge_calls}")
        print(f"  checkpoints      : {rl_plan.checkpoints}")
        print(f"  evaluations      : {rl_plan.evaluations}")
        if args.execute:
            print("  NOTE: --execute is accepted here but this stage never spends.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m octt.rl_character <stage>``."""
    import argparse

    ap = argparse.ArgumentParser(prog="octt rl")
    add_arguments(ap)
    return run_cli(ap.parse_args(argv))


__all__ = [
    "AUDIT_BANK_HASH",
    "AUDIT_BANK_PROMPTS",
    "AUDIT_BANK_ROLLOUTS",
    "DEFAULT_GUARDRAILS",
    "KL_AUDIT_BANK_ID",
    "KL_AUDIT_BANK_RELPATH",
    "KL_AUDIT_INSTRUMENT_ID",
    "KL_INDEX_MULTIPLES",
    "PROVIDER_PROMPTED_JUDGE",
    "PROVIDER_TRAINED_PM",
    "REQUIRED_GROUP_SIZE",
    "RL_PILOT",
    "RL_RECIPE_VERSION",
    "SELECTION_RULE",
    "STOPS",
    "VALIDITY_FLOOR",
    "AuditBank",
    "AuditBankDrifted",
    "AuditBankUnavailable",
    "Baseline",
    "CheckpointEval",
    "GroupReward",
    "GuardrailBreach",
    "Guardrails",
    "HeldOutTestSet",
    "KDPOIndex",
    "KDPOMeasurement",
    "LabelPreferenceReward",
    "MissingReferenceError",
    "NoEligibleCheckpoint",
    "PromptedJudgeReward",
    "RLConfig",
    "RLConfigError",
    "RLPlan",
    "RLShapeError",
    "ReferencePolicy",
    "RewardProvider",
    "RewardValidityError",
    "RunHalted",
    "RunMonitor",
    "SelectionRequired",
    "SelectionResult",
    "TestSetAlreadyUsed",
    "TrainedPMReward",
    "ValidityLedger",
    "add_arguments",
    "aggregate_outcomes",
    "assert_validity",
    "centered_advantages",
    "check_guardrails",
    "first_crossings",
    "kl_audit_instrument",
    "kl_metrics",
    "kl_policy_base_k1",
    "load_kl_audit_bank",
    "main",
    "measure_k_dpo",
    "measure_k_dpo_on_bank",
    "plan",
    "prompt_id",
    "reference_kl",
    "response_measures",
    "response_sum_kl",
    "run",
    "run_cli",
    "score_batch",
    "select_checkpoint",
    "validate_eval_row",
]
