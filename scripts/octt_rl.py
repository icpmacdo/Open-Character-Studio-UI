#!/usr/bin/env python3
"""Phase 3 policy-gradient RL runner (readiness doc WP6, batch B17).

``octt/rl_character.py`` ships the loop, the guardrails, the K_DPO index and the
selection rule as a LIBRARY, and its own CLI (``octt rl {plan,config}``) is
explicit that it never spends. This is the command surface that actually runs the
arm, exactly as ``scripts/octt_bon.py`` is the command surface for
``octt/best_of_n.py``. Dry-run by default; ``--execute`` bills Tinker.

    scripts/octt_rl.py plan                                  # free
    scripts/octt_rl.py kdpo --out runs/_mega/rl              # free without --execute
    scripts/octt_rl.py run  --out runs/_mega/rl --prompts P  # dry-run without --execute

Three stages, because they have three different costs and three different
failure modes:

``plan``
    The request/token envelope and the resolved config for one reward provider.
    Touches no client and reads only the frozen audit bank.

``kdpo``
    Measures :math:`K_{DPO}` — the banked DPO acquisition checkpoint's mean
    response-sum k3 KL from the frozen reference over the frozen 64-prompt,
    two-rollout audit bank — and banks the :class:`KDPOIndex` artifact. This
    SAMPLES, so it is ``--execute``-gated, and it is resumable: an index whose
    ``audit_bank_hash`` matches the bank on disk is reused and nothing is spent.

``run``
    The RL loop. Refuses to spend without all four prerequisites the module
    requires — a frozen reference, a reward provider, a pre-RL baseline and a
    K_DPO index — because a run that cannot compute its own stops is not allowed
    to buy tokens. After the loop it re-derives the selection from the banked
    checkpoint rows and prints BOTH the selected step and the proxy's own peak,
    so a run whose proxy kept climbing past the character peak is visible.

Exit codes (same contract as scripts/octt_bon.py)
    0  finished
    2  refused: config invalid, a guardrail breach halted the run, or the reward
       provider's validity fell below the floor
    3  paused: the result needs a human read before it means anything
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAUSED_EXIT_CODE = 3
REFUSED_EXIT_CODE = 2

#: Artifact names. Fixed so the mega driver's skip-if-done check and the resume
#: check below look at the same file.
KDPO_INDEX_NAME = "kdpo_index.json"
KDPO_PLAN_NAME = "kdpo_plan.json"
RL_HALT_NAME = "rl_halt.json"
RL_VALIDITY_NAME = "rl_reward_validity_failure.json"
RL_SELECTION_NAME = "rl_selection.json"
JUDGE_CACHE_NAME = "rl_judge_verdicts.jsonl"

#: Named so a refusal can say which of the four the operator has not supplied.
PREREQ_REFERENCE = "frozen KL reference"
PREREQ_PROVIDER = "reward provider"
PREREQ_BASELINE = "pre-RL baseline"
PREREQ_KDPO = "K_DPO index"

#: Why ``--reward-provider trained-pm --execute`` refuses. Stated as a gap, not
#: papered over with an offline stand-in: ``octt/reward_model.py`` ships only
#: OFFLINE REFERENCE models ("not a scientific instrument" — they exist so the
#: acceptance gates can be tested), and there is no inference-time scorer for the
#: Tinker-hosted checkpoint ``reward-model train`` produces. Optimizing against a
#: hand-weighted feature model and reporting it as the trained-PM arm would make
#: the arm a fiction, so this refuses instead.
TRAINED_PM_SCORER_MISSING = (
    "the trained-preference-model arm has no inference-time scorer. "
    "octt/reward_model.py ships FeatureRewardModel and its degenerate presets, "
    "which are offline reference models for testing the acceptance gates, not a "
    "trained reward model; `octt reward-model gate` says as much ('a trained "
    "model is scored by passing --checkpoint once one exists'). Scoring the "
    "banked checkpoint needs a label sampler over the comparison rendering "
    "(rl_character.LabelPreferenceReward is the shape it plugs into). Until that "
    "exists, a paid trained-pm run would be optimizing a hand-weighted feature "
    "model and reporting it as the trained-PM arm."
)

#: The purpose stamped into the held-out test set's one-shot sentinel.
HELDOUT_PURPOSE = "phase3-rl-final-report"


class Refused(RuntimeError):
    """An operator-facing refusal: what is missing and why it is not optional."""

    def __init__(self, message: str, *, missing: str = "", detail: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.missing = missing
        self.detail = tuple(detail)


def _rl():
    from octt import rl_character

    return rl_character


# ---------------------------------------------------------------------------
# Config, reference, prompts, baseline, index
# ---------------------------------------------------------------------------


def _reference(args):
    """The frozen KL reference, or ``None`` when the operator blanked it out.

    ``None`` is not a convenience: :class:`RLConfig` refuses it (gap 2 — Phase 3
    indexes every arm by its divergence from the unmodified base, so the
    reference exists even at ``kl_penalty_coefficient = 0``). The blank is
    reachable only by explicitly passing an empty ``--reference-model``, and it
    exists so the refusal is a live path rather than an unreachable branch.
    """
    rl = _rl()
    model = (getattr(args, "reference_model", None) or "").strip()
    if not model:
        return None
    return rl.ReferencePolicy(
        model_id=model,
        checkpoint_uri=(getattr(args, "reference_checkpoint", None) or None),
        role="base" if not getattr(args, "reference_checkpoint", None) else "checkpoint",
    )


def build_config(args):
    """Resolve the :class:`RLConfig` for this invocation, or raise :class:`Refused`."""
    rl = _rl()
    try:
        return rl.RLConfig(
            policy_model=args.policy_model,
            lora_rank=args.lora_rank,
            learning_rate=args.learning_rate,
            group_size=args.group_size,
            max_steps=args.max_steps,
            reward_provider=args.reward_provider,
            reference=_reference(args),
        )
    except rl.MissingReferenceError as exc:
        raise Refused(str(exc), missing=PREREQ_REFERENCE) from exc
    except rl.RLConfigError as exc:
        raise Refused(str(exc), missing="valid RLConfig") from exc


def load_prompts(path: Path | str) -> list[str]:
    """The training prompt pool: a JSON list/object, or one prompt per line."""
    path = Path(path)
    if not path.is_file():
        raise Refused(f"no training prompt pool at {path}", missing="training prompts")
    raw = path.read_text(encoding="utf-8")
    prompts: list[str]
    if path.suffix == ".json":
        payload = json.loads(raw)
        found = payload.get("prompts", []) if isinstance(payload, Mapping) else payload
        prompts = [str(p) for p in found if str(p).strip()]
    else:
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    if not prompts:
        raise Refused(f"{path} contains no prompts", missing="training prompts")
    return prompts


BASELINE_FIELDS = (
    "capability_score",
    "median_response_chars",
    "marker_density_per_100w",
    "repetition_score",
)


def load_baseline(path: Path | str | None):
    """The pre-RL :class:`Baseline` every relative stop is measured against."""
    rl = _rl()
    if not path:
        raise Refused(
            "no pre-RL baseline. The capability, length, marker-density and "
            "repetition stops are all RELATIVE: without the pre-RL numbers there "
            "is nothing for '25% length drift' or 'twice baseline' to mean.",
            missing=PREREQ_BASELINE,
        )
    path = Path(path)
    if not path.is_file():
        raise Refused(f"no pre-RL baseline at {path}", missing=PREREQ_BASELINE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "baseline" in payload:
        payload = payload["baseline"]
    missing = [f for f in BASELINE_FIELDS if payload.get(f) is None]
    if missing:
        raise Refused(
            f"{path} is missing baseline fields {missing}; a stop with no baseline "
            "cannot fire and a run whose stops cannot fire is not guarded",
            missing=PREREQ_BASELINE,
        )
    floor = payload.get("phase2_margin_floor")
    return rl.Baseline(
        capability_score=float(payload["capability_score"]),
        median_response_chars=float(payload["median_response_chars"]),
        marker_density_per_100w=float(payload["marker_density_per_100w"]),
        repetition_score=float(payload["repetition_score"]),
        phase2_margin_floor=None if floor is None else float(floor),
        source=str(payload.get("source", "pre-rl-baseline")),
    )


def _index_path(path: Path) -> Path:
    return path / KDPO_INDEX_NAME if path.is_dir() else path


def load_kdpo(path: Path | str | None, bank):
    """The banked :class:`KDPOIndex`, checked against the bank it claims to be on."""
    rl = _rl()
    if not path:
        raise Refused(
            "no K_DPO index. The KL stop is 2 x K_DPO and there is no universal "
            "KL threshold to fall back on. Run `scripts/octt_rl.py kdpo --execute` "
            "against the banked DPO checkpoint first.",
            missing=PREREQ_KDPO,
        )
    banked_path = _index_path(Path(path))
    if not banked_path.is_file():
        raise Refused(f"no K_DPO index at {banked_path}", missing=PREREQ_KDPO)
    payload = json.loads(banked_path.read_text(encoding="utf-8"))
    if payload.get("audit_bank_hash") != bank.content_hash:
        raise Refused(
            f"{banked_path} was measured on audit bank "
            f"{payload.get('audit_bank_hash')!r}, but the bank on disk hashes to "
            f"{bank.content_hash!r}. K_DPO from a different bank indexes a "
            "different x-axis; re-measure rather than reusing it.",
            missing=PREREQ_KDPO,
        )
    if payload.get("execution_mode") != rl.EXECUTION_MODE_REAL:
        raise Refused(
            f"{banked_path} is not a measured index (execution_mode="
            f"{payload.get('execution_mode')!r}); a placeholder K_DPO would make "
            "every crossing meaningless",
            missing=PREREQ_KDPO,
        )
    return rl.KDPOIndex(
        k_dpo_nats=float(payload["k_dpo_response_sum_nats"]),
        mean_token_nats=float(payload["k_dpo_mean_token_nats"]),
        max_response_sum_nats=float(payload["k_dpo_max_response_sum_nats"]),
        num_responses=int(payload["responses"]),
        num_prompts=int(payload["prompts"]),
        rollouts_per_prompt=int(payload["rollouts_per_prompt"]),
        audit_bank_id=str(payload["audit_bank_id"]),
        audit_bank_hash=str(payload["audit_bank_hash"]),
        checkpoint_fingerprint=str(payload["checkpoint_fingerprint"]),
        reference_fingerprint=str(payload["reference_fingerprint"]),
        clamped_tokens=int(payload.get("clamped_tokens", 0)),
        multiples=tuple(payload.get("multiples", rl.KL_INDEX_MULTIPLES)),
    )


def load_bank():
    """The frozen audit bank, or a refusal. Never a soft fallback."""
    rl = _rl()
    try:
        return rl.load_kl_audit_bank(ROOT)
    except rl.RLConfigError as exc:
        raise Refused(str(exc), missing="frozen KL audit bank") from exc


# ---------------------------------------------------------------------------
# Reward providers
# ---------------------------------------------------------------------------


def assert_provider_available(args) -> None:
    """Refuse an arm whose reward cannot actually be computed, BEFORE any client."""
    rl = _rl()
    if args.reward_provider == rl.PROVIDER_TRAINED_PM and args.execute:
        raise Refused(TRAINED_PM_SCORER_MISSING, missing=PREREQ_PROVIDER)


def build_provider(args, runtime, out: Path):
    """The :class:`RewardProvider` this arm optimizes.

    The prompted judge delegates to :func:`octt.preference.compare`, which
    already does blind presentation, both orders, swap resolution, caching and
    instrument stamping — none of that is reimplemented here. The trained-PM arm
    is dry-run only for now (see :data:`TRAINED_PM_SCORER_MISSING`); its offline
    stand-in is stamped with a fingerprint that says so out loud, so a dry-run
    artifact can never be read as a trained-model measurement.
    """
    rl = _rl()
    from octt import preference

    if args.reward_provider == rl.PROVIDER_PROMPTED_JUDGE:
        return rl.PromptedJudgeReward(
            runtime=runtime,
            brief=preference.get_brief(args.brief),
            judge_model=args.judge,
            cache_path=out / JUDGE_CACHE_NAME,
            execute=args.execute,
            concurrency=args.concurrency,
            policy_id=args.policy_id,
        )
    assert_provider_available(args)
    from octt import reward_model

    return rl.TrainedPMReward(
        model=reward_model.well_behaved_model(),
        model_fingerprint="offline-reference/well-behaved (DRY RUN ONLY, not a trained PM)",
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_refusal(exc: Refused) -> None:
    print(f"REFUSED: {exc}", file=sys.stderr)
    if exc.missing:
        print(f"  missing: {exc.missing}", file=sys.stderr)
    for line in exc.detail:
        print(f"  {line}", file=sys.stderr)


def script_summary(texts: Sequence[str]) -> dict[str, object]:
    """Letters-per-script breakdown of a batch of responses, under SCRIPT RULE V2.

    v1 (``persona_markers.is_latin_script``) only ever recognised CJK and kana as
    non-Latin — Arabic, Devanagari and Cyrillic all sit below U+2000 and were
    counted as Latin — and the audit bank deliberately carries cells on both
    sides of that blind spot. Anything reported here therefore uses
    :func:`persona_markers.classify_script` (v2) and stamps the rule id, so the
    number can never be silently compared to a v1 one.
    """
    from octt import persona_markers

    counts: dict[str, int] = {}
    mixed = 0
    for text in texts:
        verdict = persona_markers.classify_script(text)
        counts[verdict.script] = counts.get(verdict.script, 0) + 1
        mixed += 1 if verdict.mixed else 0
    return {
        "script_rule": persona_markers.SCRIPT_RULE_VERSION,
        "responses": len(texts),
        "scripts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "mixed": mixed,
    }


def _print_index(payload: Mapping[str, object]) -> None:
    print(f"  K_DPO (response-sum) : {payload['k_dpo_response_sum_nats']:.6f} nats")
    print(f"  mean token KL        : {payload['k_dpo_mean_token_nats']:.6f} nats")
    print(f"  bank                 : {payload['audit_bank_id']} ({payload['audit_bank_hash']})")
    print(f"  checkpoint           : {payload['checkpoint_fingerprint']}")
    print(f"  reference            : {payload['reference_fingerprint']}")
    thresholds = payload.get("thresholds_nats") or {}
    if isinstance(thresholds, Mapping):
        rendered = ", ".join(f"{k}={v:.4f}" for k, v in sorted(thresholds.items()))
        print(f"  crossings            : {rendered}")


def checkpoint_eval(rl, row: Mapping[str, object]):
    """Rebuild one :class:`CheckpointEval` from a banked ``rl_metrics`` row."""
    validity = row.get("validity") or {}
    if not isinstance(validity, Mapping):
        validity = {}
    return rl.CheckpointEval(
        step=int(row["step"]),
        proxy_reward=float(row.get("proxy_reward") or 0.0),
        character_score=float(row.get("character_score") or 0.0),
        coherence_score=float(row.get("coherence_score") or 0.0),
        capability_score=float(row.get("capability_score") or 0.0),
        format_compliance=float(row.get("format_compliance") or 0.0),
        language_match=float(row.get("language_match") or 0.0),
        median_response_chars=float(row.get("median_response_chars") or 0.0),
        marker_density_per_100w=float(row.get("marker_density_per_100w") or 0.0),
        repetition_score=float(row.get("repetition_score") or 0.0),
        reference_kl_response_sum_nats=float(row.get("reference_kl_response_sum_nats") or 0.0),
        reference_kl_mean_token_nats=float(row.get("reference_kl_mean_token_nats") or 0.0),
        kl_policy_base=float(row.get("kl_policy_base") or 0.0),
        checkpoint_uri=str(row.get("checkpoint_uri") or ""),
        checkpoint_fingerprint=str(row.get("checkpoint_fingerprint") or ""),
        optimizer_state_uri=str(row.get("optimizer_state_uri") or ""),
        provider_id=str(row.get("provider_id") or ""),
        instrument_id=str(row.get("instrument_id") or ""),
        instrument_hash=str(row.get("instrument_hash") or ""),
        validity=rl.ValidityLedger(
            decisive=int(validity.get("decisive", 0)),
            true_tie=int(validity.get("true_tie", 0)),
            swap_inconsistent=int(validity.get("swap_inconsistent", 0)),
            invalid=int(validity.get("invalid", 0)),
        ),
        proxy_reward_std=float(row.get("proxy_reward_std") or 0.0),
        execution_mode=str(row.get("execution_mode") or rl.EXECUTION_MODE_DRY_RUN),
        num_responses=int(row.get("num_responses") or 0),
    )


def guardrail_breach(rl, row: Mapping[str, object]):
    return rl.GuardrailBreach(
        name=str(row.get("name") or "unknown"),
        step=int(row.get("step") or 0),
        value=float(row.get("value") or 0.0),
        threshold=float(row.get("threshold") or 0.0),
        detail=str(row.get("detail") or ""),
    )


def select_from_monitor(rl, monitor: Mapping[str, object]):
    """Re-derive the selection from the banked rows with :func:`select_checkpoint`.

    Deliberately re-derived rather than read out of the run payload: the rule
    ("peak of the independent measure among checkpoints preceding any breach,
    never continued proxy improvement") is the thing being audited, and running
    it over the artifact that was actually written is what proves the artifact
    supports it.
    """
    rows = monitor.get("checkpoints") or []
    breaches = [guardrail_breach(rl, b) for b in (monitor.get("breaches") or [])]
    history = [checkpoint_eval(rl, row) for row in rows if isinstance(row, Mapping)]
    return rl.select_checkpoint(history, breaches), history, breaches


def print_selection(selection) -> None:
    """Selected step AND the proxy's own peak, always both."""
    print()
    print(f"selection  : step {selection.selected_step} ({selection.measure})")
    print(f"  rule            : {selection.rule}")
    print(f"  independent     : {selection.independent_score:.6f}")
    print(f"  proxy at pick   : {selection.proxy_reward:.6f}")
    print(
        f"  proxy PEAK      : step {selection.proxy_peak_step} "
        f"({selection.proxy_peak_reward:.6f})"
    )
    if selection.differs_from_proxy_peak:
        print(
            "  DIVERGENCE      : the proxy peaked at a different checkpoint than "
            "the independent measure. Following the proxy would have shipped "
            f"step {selection.proxy_peak_step}; it was NOT selected."
        )
    else:
        print("  divergence      : none (the proxy peak and the character peak coincide)")
    print(f"  eligible steps  : {list(selection.eligible_steps)}")
    print(f"  checkpoint      : {selection.checkpoint_uri}")
    print(f"  selection hash  : {selection.selection_hash}")


def report_halt(rl, out: Path, breaches: Sequence, selection, payload) -> int:
    """A guardrail breach is a RESULT, not a crash. Bank it and exit 2."""
    from octt import manifest

    print()
    print("HALTED: a predeclared hard stop fired.", file=sys.stderr)
    for breach in breaches:
        print(f"  - {breach.name} @ step {breach.step}", file=sys.stderr)
        print(
            f"      value={breach.value:.4f} limit={breach.threshold:.4f}",
            file=sys.stderr,
        )
        print(f"      {breach.detail}", file=sys.stderr)
    print(
        "  The checkpoints banked BEFORE the breach remain valid and are not "
        "discarded; selection already ran over them.",
        file=sys.stderr,
    )
    manifest.atomic_write_json(
        out / RL_HALT_NAME,
        {
            "recipe_version": rl.RL_RECIPE_VERSION,
            "halted_at_step": min((b.step for b in breaches), default=None),
            "breaches": [b.to_dict() for b in breaches],
            "selection": None if selection is None else selection.to_dict(),
            "banked_checkpoints_remain_valid": True,
            "run": payload,
        },
    )
    print(f"  artifacts: {out / RL_HALT_NAME}", file=sys.stderr)
    return REFUSED_EXIT_CODE


def report_validity_failure(rl, out: Path, exc) -> int:
    """Report the LEDGER, not just the exception: invalid is missing data."""
    from octt import manifest

    ledger = exc.ledger
    print()
    print("REFUSED: the reward provider's validity fell below the floor.", file=sys.stderr)
    print(
        f"  validity_rate     : {ledger.validity_rate:.4f} (floor {exc.floor:.4f})",
        file=sys.stderr,
    )
    print(f"  decisive          : {ledger.decisive}", file=sys.stderr)
    print(f"  true_tie          : {ledger.true_tie}", file=sys.stderr)
    print(f"  swap_inconsistent : {ledger.swap_inconsistent}", file=sys.stderr)
    print(f"  invalid           : {ledger.invalid}", file=sys.stderr)
    print(f"  total             : {ledger.total}", file=sys.stderr)
    print(
        "  An invalid label is MISSING DATA, not a tie. A reward built from them "
        "is not the reward the pre-RL gates certified.",
        file=sys.stderr,
    )
    manifest.atomic_write_json(
        out / RL_VALIDITY_NAME,
        {
            "recipe_version": rl.RL_RECIPE_VERSION,
            "stop": rl.STOP_VALIDITY,
            "floor": exc.floor,
            "validity": ledger.to_dict(),
            "note": (
                "per-checkpoint rows written before the abort are in rl_metrics.jsonl "
                "and remain valid"
            ),
        },
    )
    print(f"  artifacts: {out / RL_VALIDITY_NAME}", file=sys.stderr)
    return REFUSED_EXIT_CODE


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def cmd_plan(args) -> int:
    rl = _rl()
    try:
        config = build_config(args)
        bank = load_bank()
    except Refused as exc:
        _print_refusal(exc)
        return REFUSED_EXIT_CODE

    rl_plan = rl.plan(config, num_prompts=args.prompts)
    payload = {
        "recipe_version": rl.RL_RECIPE_VERSION,
        "plan": rl_plan.to_dict(),
        "audit_bank": bank.to_dict(),
        "guardrails": rl.DEFAULT_GUARDRAILS.to_dict(),
        "stops": list(rl.STOPS),
        "selection_rule": rl.SELECTION_RULE,
        "kl_index_multiples": list(rl.KL_INDEX_MULTIPLES),
    }
    if args.out:
        from octt import manifest

        manifest.atomic_write_json(Path(args.out), payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    reference = config.require_reference()
    print(f"RL plan ({rl.RL_RECIPE_VERSION}, {config.reward_provider})")
    print(f"  policy model     : {config.policy_model} (LoRA rank {config.lora_rank})")
    print(f"  reference        : {reference.fingerprint}")
    print(f"  tournament       : {config.tournament} at G={config.group_size}")
    print(f"  steps            : {rl_plan.steps}")
    print(f"  samples          : {rl_plan.samples} ({config.samples_per_step}/step)")
    print(f"  sample tokens    : {rl_plan.sample_tokens}")
    print(f"  reference tokens : {rl_plan.reference_logprob_tokens}")
    print(f"  train tokens     : {rl_plan.train_tokens}")
    print(f"  judge calls      : {rl_plan.judge_calls} ({rl_plan.judge_tokens} tokens)")
    print(f"  checkpoints      : {rl_plan.checkpoints}")
    print(f"  evaluations      : {rl_plan.evaluations}")
    print()
    print(f"audit bank : {bank.bank_id} ({bank.content_hash})")
    print(
        f"  {len(bank.prompts)} prompts x {bank.rollouts_per_prompt} rollouts "
        f"= {bank.num_responses} responses"
    )
    print()
    print(f"selection  : {rl.SELECTION_RULE}")
    print(f"stops      : {', '.join(rl.STOPS)}")
    print("free: this stage touched no client and spent nothing.")
    return 0


def read_banked_index(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def cmd_kdpo(args) -> int:
    rl = _rl()
    from octt import manifest

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        bank = load_bank()
        reference = _reference(args)
        if reference is None:
            raise Refused(
                "K_DPO is a divergence FROM something; it cannot be measured "
                "without a frozen reference",
                missing=PREREQ_REFERENCE,
            )
    except Refused as exc:
        _print_refusal(exc)
        return REFUSED_EXIT_CODE

    index_path = out / KDPO_INDEX_NAME
    banked = read_banked_index(index_path)
    if banked is not None:
        if banked.get("audit_bank_hash") != bank.content_hash:
            _print_refusal(
                Refused(
                    f"{index_path} was measured on audit bank "
                    f"{banked.get('audit_bank_hash')!r} but the bank on disk hashes "
                    f"to {bank.content_hash!r}; delete the stale index and "
                    "re-measure rather than indexing against the wrong x-axis",
                    missing=PREREQ_KDPO,
                )
            )
            return REFUSED_EXIT_CODE
        print(f"reuse: {index_path} (audit_bank_hash matches; nothing spent)")
        _print_index(banked)
        return 0

    if not args.execute:
        plan_payload = {
            "status": "dry-run",
            "recipe_version": rl.RL_RECIPE_VERSION,
            "execution_mode": rl.EXECUTION_MODE_DRY_RUN,
            "audit_bank": bank.to_dict(),
            "reference": reference.to_dict(),
            "policy_model": args.policy_model,
            "dpo_checkpoint": args.dpo_checkpoint,
            "instrument": rl.kl_audit_instrument(),
            "projected_samples": bank.num_responses,
            "projected_sample_tokens": bank.num_responses
            * int(rl.kl_audit_instrument()["sampling"]["max_tokens"]),
            "note": (
                "no Tinker calls were made and NO index was written: a dry-run "
                "K_DPO would be 0.0 nats and would divide every crossing by zero. "
                f"Pass --execute to measure and bank {KDPO_INDEX_NAME}."
            ),
        }
        manifest.atomic_write_json(out / KDPO_PLAN_NAME, plan_payload)
        print(f"K_DPO plan [dry-run] -> {out / KDPO_PLAN_NAME}")
        print(f"  bank      : {bank.bank_id} ({bank.content_hash})")
        print(f"  reference : {reference.fingerprint}")
        print(f"  policy    : {args.policy_model} @ {args.dpo_checkpoint or '(unset)'}")
        print(
            f"  sampling  : {bank.num_responses} responses "
            f"({len(bank.prompts)} prompts x {bank.rollouts_per_prompt} rollouts), "
            f"{plan_payload['projected_sample_tokens']} sample tokens max"
        )
        print(f"  + {bank.num_responses} reference log-probability requests")
        print("free: this stage touched no client and spent nothing.")
        print(f"NOT written: {index_path} (a dry-run index would poison every crossing)")
        return 0

    checkpoint = (args.dpo_checkpoint or "").strip()
    if not checkpoint.startswith("tinker://"):
        _print_refusal(
            Refused(
                "K_DPO indexes the BANKED DPO acquisition checkpoint; "
                f"--dpo-checkpoint {checkpoint!r} is not a tinker:// URI. A "
                "placeholder would silently index the wrong weights.",
                missing="banked DPO checkpoint",
            )
        )
        return REFUSED_EXIT_CODE

    from octt import tinker_client

    runtime = tinker_client.create_runtime(
        sorted({args.policy_model, reference.model_id}),
        tinker_client.TinkerClientConfig(dry_run=False),
    )
    print(
        f"K_DPO [EXECUTE (paid)]: {len(bank.prompts)} prompts x "
        f"{bank.rollouts_per_prompt} rollouts on {bank.bank_id}"
    )
    try:
        measurement = rl.measure_k_dpo_on_bank(
            bank,
            runtime,
            checkpoint_uri=checkpoint,
            policy_model=args.policy_model,
            reference=reference,
            execute=True,
            concurrency=args.concurrency,
        )
    except (rl.RLConfigError, rl.RLShapeError) as exc:
        _print_refusal(Refused(str(exc), missing="a measurable K_DPO"))
        return REFUSED_EXIT_CODE

    payload = {
        **measurement.index.to_dict(),
        "execution_mode": rl.EXECUTION_MODE_REAL,
        "policy_model": args.policy_model,
        "prompt_tokens": measurement.prompt_tokens,
        "response_tokens": measurement.response_tokens,
        "script_summary": script_summary(measurement.texts),
    }
    manifest.atomic_write_json(index_path, payload)
    print(f"index: {index_path}")
    _print_index(payload)
    summary = payload["script_summary"]
    print(f"  scripts ({summary['script_rule']}): {summary['scripts']}")
    return 0


def resolve_run_prerequisites(args, bank):
    """The four things a paid RL run may not start without.

    Order matters: everything here is checked BEFORE a client is constructed, so
    a refusal costs nothing. Dry runs skip the last three because
    :func:`octt.rl_character.run` does not use them without ``execute`` — the
    module's own contract, mirrored rather than duplicated.
    """
    config = build_config(args)
    prompts = load_prompts(args.prompts)
    if not args.execute:
        return config, prompts, None, None
    assert_provider_available(args)
    kdpo = load_kdpo(args.kdpo, bank)
    baseline = load_baseline(args.baseline)
    return config, prompts, baseline, kdpo


def cmd_run(args) -> int:
    rl = _rl()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        bank = load_bank()
        config, prompts, baseline, kdpo = resolve_run_prerequisites(args, bank)
    except Refused as exc:
        _print_refusal(exc)
        return REFUSED_EXIT_CODE

    from octt import tinker_client

    mode = "EXECUTE (paid)" if args.execute else "dry-run"
    runtime = tinker_client.create_runtime(
        sorted({config.policy_model, config.require_reference().model_id, args.judge}),
        tinker_client.TinkerClientConfig(dry_run=not args.execute),
    )
    try:
        provider = build_provider(args, runtime, out)
    except Refused as exc:
        _print_refusal(exc)
        return REFUSED_EXIT_CODE

    print(
        f"rl: {config.reward_provider}, {config.max_steps} steps x "
        f"{config.samples_per_step} samples [{mode}]"
    )
    try:
        payload = rl.run(
            prompts,
            out,
            runtime,
            config,
            execute=args.execute,
            reward_provider=provider,
            baseline=baseline,
            kdpo=kdpo,
        )
    except rl.RunHalted as halted:
        return report_halt(rl, out, halted.breaches, None, None)
    except rl.RewardValidityError as invalid:
        return report_validity_failure(rl, out, invalid)
    except rl.RLConfigError as exc:
        _print_refusal(Refused(str(exc), missing="a runnable RL configuration"))
        return REFUSED_EXIT_CODE

    if payload.get("status") != "executed":
        print(f"artifacts: {out / 'rl_plan.json'}")
        print("dry-run: no Tinker calls were made and nothing was banked.")
        return 0

    monitor = payload.get("monitor") or {}
    try:
        selection, history, breaches = select_from_monitor(rl, monitor)
    except rl.NoEligibleCheckpoint as exc:
        print()
        print(f"REFUSED: {exc}", file=sys.stderr)
        return report_halt(
            rl,
            out,
            [guardrail_breach(rl, b) for b in (monitor.get("breaches") or [])],
            None,
            payload,
        )

    print_selection(selection)
    crossings = monitor.get("kl_crossings") or {}
    if crossings:
        print(f"  KL crossings    : {crossings}")

    from octt import manifest

    manifest.atomic_write_json(
        out / RL_SELECTION_NAME,
        {
            "recipe_version": rl.RL_RECIPE_VERSION,
            "selection": selection.to_dict(),
            "kl_crossings": crossings,
            "checkpoints": len(history),
        },
    )

    if breaches:
        return report_halt(rl, out, breaches, selection, payload)

    independent = {round(e.independent(selection.measure), 12) for e in history}
    if len(independent) <= 1:
        print()
        print("PAUSED: the selection has no independent measure to peak on.")
        print(
            f"        {selection.measure} was identical at all {len(history)} "
            "checkpoints, so the peak is a tie broken by step order, not a result."
        )
        print(
            "        The banked checkpoints are valid. Supply an out-of-loop "
            "character evaluation and re-select before opening the held-out test set."
        )
        return PAUSED_EXIT_CODE

    heldout = rl.HeldOutTestSet(out)
    try:
        record = heldout.use(selection, purpose=HELDOUT_PURPOSE)
    except rl.TestSetAlreadyUsed as exc:
        print()
        print(f"REFUSED: {exc}", file=sys.stderr)
        return REFUSED_EXIT_CODE
    print()
    print(f"held-out test set: opened once for {record['purpose']!r}")
    print(f"  sentinel : {heldout.sentinel_path}")
    print("  There is no second look; the sentinel is on disk, not in memory.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from octt import models
    from octt import rl_character as rl

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn in (("plan", cmd_plan), ("kdpo", cmd_kdpo), ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn, command=name)
        sp.add_argument("--policy-model", default=rl.RL_PILOT.policy_model)
        sp.add_argument(
            "--reference-model",
            default=rl.DEFAULT_REFERENCE.model_id,
            help="frozen KL reference base model. Blank is REFUSED: Phase 3 "
            "indexes every arm by its divergence from the unmodified base, so the "
            "reference exists even at kl_penalty_coefficient=0.",
        )
        sp.add_argument(
            "--reference-checkpoint",
            default=None,
            help="reference weights other than the unmodified base (rarely right: "
            "the base is what makes DPO, BoN, RL and OPD divergences comparable)",
        )
        sp.add_argument("--lora-rank", type=int, default=rl.RL_PILOT.lora_rank)
        sp.add_argument("--learning-rate", type=float, default=rl.RL_PILOT.learning_rate)
        sp.add_argument(
            "--group-size",
            type=int,
            default=rl.REQUIRED_GROUP_SIZE,
            help="pinned at 4; any other value is refused (the vendored builder "
            "silently splits larger groups into contiguous fours)",
        )
        sp.add_argument("--max-steps", type=int, default=rl.RL_PILOT.max_steps)
        sp.add_argument(
            "--reward-provider",
            default=rl.PROVIDER_PROMPTED_JUDGE,
            choices=rl.REWARD_PROVIDERS,
        )
        sp.add_argument("--judge", default=models.TEACHER_MODEL, help="judge model id")
        sp.add_argument("--brief", default=None, help="character brief id")
        sp.add_argument("--concurrency", type=int, default=32)
        sp.add_argument("--policy-id", default="rl-policy")
        sp.add_argument(
            "--execute",
            action="store_true",
            help="bill Tinker. Omit for a free run.",
        )

        if name == "plan":
            sp.add_argument("--prompts", type=int, default=400, help="prompt pool size")
            sp.add_argument("--out", default=None, help="write the plan JSON here")
            sp.add_argument("--json", action="store_true")
        if name == "kdpo":
            sp.add_argument("--out", required=True, help="directory for the index artifact")
            sp.add_argument(
                "--dpo-checkpoint",
                default=None,
                help="banked post-DPO tinker:// URI — the checkpoint K_DPO indexes",
            )
        if name == "run":
            sp.add_argument("--out", required=True)
            sp.add_argument("--prompts", required=True, help="training prompt pool file")
            sp.add_argument(
                "--kdpo",
                default=None,
                help=f"directory or path of the banked {KDPO_INDEX_NAME}",
            )
            sp.add_argument("--baseline", default=None, help="pre-RL baseline JSON")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from octt import preference

    if getattr(args, "brief", None) is None:
        args.brief = preference.DEFAULT_BRIEF_ID
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
