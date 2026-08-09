"""``scripts/octt_rl.py`` — the production RL runner, and the bank loader it needs.

Offline and deterministic: no API keys, no training stack, no network. Every
paid call site is either monkeypatched or asserted never to be reached, so a
regression that makes a free stage touch a client fails here rather than on the
invoice.

What is pinned:

1. **The bank loader refuses drift.** ``rl_character.load_kl_audit_bank`` is the
   only supported way to build an :class:`AuditBank`. Before it existed the only
   code that turned the frozen panel into one lived in a test, which is exactly
   the failure ``AuditBank`` was designed to refuse: a call site handing
   ``measure_k_dpo`` an ad-hoc bank passes every shape check while hashing to
   something else, and K_DPO measured on it silently re-indexes every crossing.
2. **K_DPO is measured once.** The index artifact is the resume token: a second
   ``kdpo`` invocation against a matching bank must reuse it and construct no
   client at all.
3. **A guardrail halt is a result.** It reports which stop fired at which step,
   banks the artifacts, keeps the pre-breach checkpoints, and exits 2 — it is
   never a traceback.
4. **A reward-validity failure reports the LEDGER.** decisive / true_tie /
   swap_inconsistent / invalid, because "the judge said equal" and "nothing was
   measured" are the same number and must never be the same fact.
5. **A paid run refuses without all four prerequisites**, before any client
   exists, so the refusal is free.
6. **Selection never follows the proxy**, and both the selected step and
   ``proxy_peak_step`` are printed so divergence is visible.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from octt import persona_markers, qualitative, tinker_client
from octt import rl_character as rl

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "octt_rl.py"

spec = importlib.util.spec_from_file_location("octt_rl", SCRIPT)
runner = importlib.util.module_from_spec(spec)
sys.modules["octt_rl"] = runner
spec.loader.exec_module(runner)


# --------------------------------------------------------------- fixtures


def _bank():
    return rl.load_kl_audit_bank(REPO)


def _fake_runtime(dry_run: bool = False):
    return SimpleNamespace(config=SimpleNamespace(dry_run=dry_run))


def _index(bank, *, k_dpo: float = 8.0):
    return rl.KDPOIndex(
        k_dpo_nats=k_dpo,
        mean_token_nats=0.05,
        max_response_sum_nats=20.0,
        num_responses=bank.num_responses,
        num_prompts=len(bank.prompts),
        rollouts_per_prompt=bank.rollouts_per_prompt,
        audit_bank_id=bank.bank_id,
        audit_bank_hash=bank.content_hash,
        checkpoint_fingerprint="tinker://banked/dpo-4b",
        reference_fingerprint=rl.DEFAULT_REFERENCE.fingerprint,
    )


def _write_kdpo(tmp_path: pathlib.Path, bank, *, mode: str = rl.EXECUTION_MODE_REAL):
    path = tmp_path / runner.KDPO_INDEX_NAME
    path.write_text(json.dumps({**_index(bank).to_dict(), "execution_mode": mode}))
    return path


def _write_baseline(tmp_path: pathlib.Path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "capability_score": 70.0,
                "median_response_chars": 400.0,
                "marker_density_per_100w": 5.0,
                "repetition_score": 0.10,
                "phase2_margin_floor": 66.0,
            }
        )
    )
    return path


def _write_prompts(tmp_path: pathlib.Path):
    path = tmp_path / "prompts.txt"
    path.write_text("Explain photosynthesis.\nHow do I fix a dripping tap?\n")
    return path


def _row(step: int, *, character: float, proxy: float, kl: float = 1.0) -> dict:
    return rl.CheckpointEval(
        step=step,
        proxy_reward=proxy,
        character_score=character,
        coherence_score=0.5,
        capability_score=70.0,
        format_compliance=1.0,
        language_match=1.0,
        median_response_chars=400.0,
        marker_density_per_100w=5.0,
        repetition_score=0.10,
        reference_kl_response_sum_nats=kl,
        reference_kl_mean_token_nats=0.01,
        kl_policy_base=-0.01,
        checkpoint_uri=f"tinker://ckpt/{step}",
        checkpoint_fingerprint=f"tinker://ckpt/{step}",
        optimizer_state_uri=f"tinker://state/{step}",
        provider_id=rl.PROVIDER_PROMPTED_JUDGE,
        instrument_id="character-preference/judge-v1",
        instrument_hash="deadbeef",
        execution_mode=rl.EXECUTION_MODE_REAL,
        num_responses=32,
    ).to_row()


def _executed(rows, breaches=()):
    return {
        "status": "executed",
        "recipe_version": rl.RL_RECIPE_VERSION,
        "execution_mode": rl.EXECUTION_MODE_REAL,
        "monitor": {
            "checkpoints": list(rows),
            "breaches": [b.to_dict() for b in breaches],
            "kl_crossings": {"0.25x": 5, "0.5x": None, "1x": None, "2x": None},
        },
    }


def _no_client(monkeypatch):
    """Any construction of a Tinker runtime is a test failure."""

    def _boom(*args, **kwargs):
        raise AssertionError("a free stage constructed a Tinker runtime")

    monkeypatch.setattr(tinker_client, "create_runtime", _boom)


def _stub_runtime(monkeypatch, *, dry_run: bool = False) -> dict:
    calls = {"n": 0}

    def _make(*args, **kwargs):
        calls["n"] += 1
        return _fake_runtime(dry_run)

    monkeypatch.setattr(tinker_client, "create_runtime", _make)
    return calls


def _paid_run_args(tmp_path, bank, extra=()):
    out = tmp_path / "out"
    return [
        "run",
        "--out",
        str(out),
        "--prompts",
        str(_write_prompts(tmp_path)),
        "--kdpo",
        str(_write_kdpo(tmp_path, bank)),
        "--baseline",
        str(_write_baseline(tmp_path)),
        "--execute",
        *extra,
    ]


# ------------------------------------------------- 1. the bank loader


def test_the_loader_builds_the_pinned_bank_from_the_frozen_panel():
    bank = _bank()
    assert bank.bank_id == rl.KL_AUDIT_BANK_ID == "kl-audit-64x2-v1"
    assert len(bank.prompts) == rl.AUDIT_BANK_PROMPTS == 64
    assert bank.rollouts_per_prompt == rl.AUDIT_BANK_ROLLOUTS == 2
    assert bank.num_responses == 128
    assert bank.content_hash == rl.AUDIT_BANK_HASH == "c50bca08a85517c0"
    assert bank.to_dict()["instrument_id"] == rl.KL_AUDIT_INSTRUMENT_ID
    # Deterministic: two loads are the same bank, not merely the same shape.
    assert rl.load_kl_audit_bank(REPO).content_hash == bank.content_hash


def _clone_bank(tmp_path: pathlib.Path, mutate=None) -> pathlib.Path:
    payload = json.loads((REPO / rl.KL_AUDIT_BANK_RELPATH).read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(payload)
    path = tmp_path / rl.KL_AUDIT_BANK_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_drifted_bank_raises_and_never_warns(tmp_path):
    """One edited prompt is a different x-axis. It must be fatal, not a log line."""

    def _edit(payload):
        payload["prompts"][0]["text"] = payload["prompts"][0]["text"] + " (reworded)"

    _clone_bank(tmp_path, _edit)
    # The edited file still VALIDATES as a panel — 64 unique prompts, quotas
    # intact — which is precisely why the shape checks cannot catch this.
    panel = qualitative.load_panel(tmp_path / rl.KL_AUDIT_BANK_RELPATH)
    assert len(panel.prompts) == 64
    ad_hoc = rl.AuditBank(bank_id=panel.panel_id, prompts=tuple(p.text for p in panel.prompts))
    assert ad_hoc.content_hash != rl.AUDIT_BANK_HASH

    with pytest.raises(rl.AuditBankDrifted, match="c50bca08a85517c0"):
        rl.load_kl_audit_bank(tmp_path)


def test_a_renamed_bank_is_refused_by_id_before_the_hash(tmp_path):
    _clone_bank(tmp_path, lambda p: p.update(panel_id="kl-audit-64x2-v2"))
    with pytest.raises(rl.AuditBankDrifted, match="kl-audit-64x2-v1"):
        rl.load_kl_audit_bank(tmp_path)


def test_a_missing_bank_is_unavailable_not_an_empty_bank(tmp_path):
    with pytest.raises(rl.AuditBankUnavailable, match="missing"):
        rl.load_kl_audit_bank(tmp_path)


def test_an_unreadable_bank_is_unavailable(tmp_path):
    path = tmp_path / rl.KL_AUDIT_BANK_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(rl.AuditBankUnavailable):
        rl.load_kl_audit_bank(tmp_path)


def test_the_runner_surfaces_a_bank_refusal_rather_than_crashing(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    assert runner.main(["plan"]) == runner.REFUSED_EXIT_CODE
    assert "REFUSED" in capsys.readouterr().err


# ------------------------------------------------- 2. plan is free


def test_plan_is_free_and_touches_no_client(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    out = tmp_path / "plan.json"
    assert runner.main(["plan", "--prompts", "80", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "RL plan" in text
    assert rl.AUDIT_BANK_HASH in text
    assert "spent nothing" in text
    payload = json.loads(out.read_text())
    assert payload["audit_bank"]["audit_bank_hash"] == rl.AUDIT_BANK_HASH
    assert payload["selection_rule"] == rl.SELECTION_RULE
    assert payload["plan"]["config"]["group_size"] == 4


def test_plan_refuses_a_group_size_other_than_four(monkeypatch, capsys):
    _no_client(monkeypatch)
    assert runner.main(["plan", "--group-size", "8"]) == runner.REFUSED_EXIT_CODE
    assert "group_size must be exactly 4" in capsys.readouterr().err


# ------------------------------------------------- 3. K_DPO: resumable


def test_kdpo_without_execute_is_free_and_banks_no_index(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    assert runner.main(["kdpo", "--out", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "dry-run" in text
    assert "spent nothing" in text
    assert (tmp_path / runner.KDPO_PLAN_NAME).is_file()
    # A dry-run index would make the next --execute skip the only measurement
    # that matters, so it is deliberately NOT written.
    assert not (tmp_path / runner.KDPO_INDEX_NAME).exists()


def test_kdpo_measures_once_then_reuses_the_artifact(tmp_path, monkeypatch, capsys):
    bank = _bank()
    runtime_calls = _stub_runtime(monkeypatch)
    measured = {"n": 0}

    def _measure(bank_arg, runtime_arg, **kwargs):
        measured["n"] += 1
        assert bank_arg.content_hash == rl.AUDIT_BANK_HASH
        assert kwargs["execute"] is True
        return rl.KDPOMeasurement(
            index=_index(bank_arg),
            texts=("a response",) * bank_arg.num_responses,
            prompt_tokens=1280,
            response_tokens=6400,
        )

    monkeypatch.setattr(rl, "measure_k_dpo_on_bank", _measure)
    argv = [
        "kdpo",
        "--out",
        str(tmp_path),
        "--dpo-checkpoint",
        "tinker://banked/dpo-4b",
        "--execute",
    ]

    assert runner.main(argv) == 0
    assert (measured["n"], runtime_calls["n"]) == (1, 1)
    banked = json.loads((tmp_path / runner.KDPO_INDEX_NAME).read_text())
    assert banked["audit_bank_hash"] == bank.content_hash
    assert banked["execution_mode"] == rl.EXECUTION_MODE_REAL
    assert banked["script_summary"]["script_rule"] == persona_markers.SCRIPT_RULE_V2
    capsys.readouterr()

    # Second invocation: the index is the resume token. Nothing is sampled and
    # no client is constructed at all.
    assert runner.main(argv) == 0
    assert (measured["n"], runtime_calls["n"]) == (1, 1)
    out = capsys.readouterr().out
    assert "reuse" in out
    assert rl.AUDIT_BANK_HASH in out


def test_kdpo_refuses_to_reuse_an_index_from_a_different_bank(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    stale = {**_index(_bank()).to_dict(), "execution_mode": rl.EXECUTION_MODE_REAL}
    stale["audit_bank_hash"] = "0000000000000000"
    (tmp_path / runner.KDPO_INDEX_NAME).write_text(json.dumps(stale))
    argv = ["kdpo", "--out", str(tmp_path), "--dpo-checkpoint", "tinker://x", "--execute"]
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    assert "0000000000000000" in capsys.readouterr().err


def test_kdpo_refuses_a_placeholder_checkpoint(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    argv = ["kdpo", "--out", str(tmp_path), "--dpo-checkpoint", "the-4b-one", "--execute"]
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    assert "tinker://" in capsys.readouterr().err


def test_a_dry_run_k_dpo_is_refused_by_the_library(tmp_path):
    """0.0 nats would divide every crossing by zero, so there is no dry-run number."""
    with pytest.raises(rl.RLConfigError, match="dry-run"):
        rl.measure_k_dpo_on_bank(
            _bank(),
            _fake_runtime(dry_run=True),
            checkpoint_uri="tinker://banked/dpo-4b",
            execute=True,
        )


# ---------------------------------- 4. run refuses without its prerequisites


def test_run_refuses_to_execute_without_a_frozen_reference(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    argv = _paid_run_args(tmp_path, _bank(), ["--reference-model", ""])
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    err = capsys.readouterr().err
    assert runner.PREREQ_REFERENCE in err
    assert "kl_penalty_coefficient is 0" in err


def test_run_refuses_to_execute_without_a_reward_provider(tmp_path, monkeypatch, capsys):
    """trained-pm has no inference-time scorer; the runner says so instead of stubbing."""
    _no_client(monkeypatch)
    argv = _paid_run_args(tmp_path, _bank(), ["--reward-provider", rl.PROVIDER_TRAINED_PM])
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    err = capsys.readouterr().err
    assert runner.PREREQ_PROVIDER in err
    assert "no inference-time scorer" in err


def test_run_refuses_to_execute_without_a_kdpo_index(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    argv = [
        "run",
        "--out",
        str(tmp_path / "out"),
        "--prompts",
        str(_write_prompts(tmp_path)),
        "--baseline",
        str(_write_baseline(tmp_path)),
        "--execute",
    ]
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    err = capsys.readouterr().err
    assert runner.PREREQ_KDPO in err
    assert "2 x K_DPO" in err


def test_run_refuses_to_execute_without_a_baseline(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    argv = [
        "run",
        "--out",
        str(tmp_path / "out"),
        "--prompts",
        str(_write_prompts(tmp_path)),
        "--kdpo",
        str(_write_kdpo(tmp_path, _bank())),
        "--execute",
    ]
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    err = capsys.readouterr().err
    assert runner.PREREQ_BASELINE in err
    assert "RELATIVE" in err


def test_run_refuses_a_kdpo_index_measured_on_another_bank(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    path = _write_kdpo(tmp_path, _bank())
    payload = json.loads(path.read_text())
    payload["audit_bank_hash"] = "1111111111111111"
    path.write_text(json.dumps(payload))
    argv = [
        "run", "--out", str(tmp_path / "out"),
        "--prompts", str(_write_prompts(tmp_path)),
        "--kdpo", str(path),
        "--baseline", str(_write_baseline(tmp_path)),
        "--execute",
    ]
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    assert "different x-axis" in capsys.readouterr().err


def test_run_refuses_a_dry_run_kdpo_index(tmp_path, monkeypatch, capsys):
    _no_client(monkeypatch)
    path = _write_kdpo(tmp_path, _bank(), mode=rl.EXECUTION_MODE_DRY_RUN)
    argv = [
        "run", "--out", str(tmp_path / "out"),
        "--prompts", str(_write_prompts(tmp_path)),
        "--kdpo", str(path),
        "--baseline", str(_write_baseline(tmp_path)),
        "--execute",
    ]
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    assert "not a measured index" in capsys.readouterr().err


def test_run_without_execute_is_a_dry_run_needing_none_of_them(tmp_path, monkeypatch, capsys):
    _stub_runtime(monkeypatch, dry_run=True)
    out = tmp_path / "out"
    argv = ["run", "--out", str(out), "--prompts", str(_write_prompts(tmp_path))]
    assert runner.main(argv) == 0
    assert (out / "rl_plan.json").is_file()
    assert "dry-run" in capsys.readouterr().out
    assert not (out / rl.TEST_SET_SENTINEL).exists()


# ------------------------------------------- 5. guardrail halt is a result


def _breach(step: int = 15):
    return rl.GuardrailBreach(
        name=rl.STOP_KL,
        step=step,
        value=17.5,
        threshold=16.0,
        detail="response-sum KL crossed 2 x K_DPO (8.0000 nats)",
    )


def test_a_guardrail_halt_reports_the_stop_banks_artifacts_and_exits_two(
    tmp_path, monkeypatch, capsys
):
    _stub_runtime(monkeypatch)
    rows = [
        _row(5, character=0.40, proxy=0.10),
        _row(10, character=0.62, proxy=0.30),
        _row(15, character=0.55, proxy=0.90, kl=17.5),
    ]
    breach = _breach()
    monkeypatch.setattr(rl, "run", lambda *a, **k: _executed(rows, [breach]))
    out = tmp_path / "out"
    assert runner.main(_paid_run_args(tmp_path, _bank())) == runner.REFUSED_EXIT_CODE

    captured = capsys.readouterr()
    assert rl.STOP_KL in captured.err
    assert "step 15" in captured.err
    assert "remain valid" in captured.err

    banked = json.loads((out / runner.RL_HALT_NAME).read_text())
    assert banked["halted_at_step"] == 15
    assert banked["breaches"][0]["name"] == rl.STOP_KL
    assert banked["banked_checkpoints_remain_valid"] is True
    # The pre-breach checkpoints were not discarded: one of them was selected.
    assert banked["selection"]["selected_step"] == 10
    assert banked["selection"]["eligible_steps"] == [5, 10]
    assert (out / runner.RL_SELECTION_NAME).is_file()
    # A halted run must not spend the one-shot held-out test set.
    assert not (out / rl.TEST_SET_SENTINEL).exists()


def test_a_run_halted_exception_is_caught_not_a_traceback(tmp_path, monkeypatch, capsys):
    _stub_runtime(monkeypatch)

    def _raise(*args, **kwargs):
        raise rl.RunHalted([_breach(step=20)])

    monkeypatch.setattr(rl, "run", _raise)
    out = tmp_path / "out"
    assert runner.main(_paid_run_args(tmp_path, _bank())) == runner.REFUSED_EXIT_CODE
    err = capsys.readouterr().err
    assert "HALTED" in err
    assert "step 20" in err
    assert json.loads((out / runner.RL_HALT_NAME).read_text())["halted_at_step"] == 20


def test_a_breach_at_the_first_checkpoint_leaves_nothing_to_select(
    tmp_path, monkeypatch, capsys
):
    _stub_runtime(monkeypatch)
    breach = _breach(step=5)
    monkeypatch.setattr(
        rl, "run", lambda *a, **k: _executed([_row(5, character=0.4, proxy=0.9)], [breach])
    )
    assert runner.main(_paid_run_args(tmp_path, _bank())) == runner.REFUSED_EXIT_CODE
    err = capsys.readouterr().err
    assert "nothing safe to select" in err
    assert rl.STOP_KL in err


# ------------------------------- 6. reward-validity failure reports the ledger


def test_a_reward_validity_failure_reports_the_ledger_counts(tmp_path, monkeypatch, capsys):
    _stub_runtime(monkeypatch)
    ledger = rl.ValidityLedger(decisive=70, true_tie=9, swap_inconsistent=5, invalid=16)

    def _raise(*args, **kwargs):
        raise rl.RewardValidityError(ledger, rl.VALIDITY_FLOOR)

    monkeypatch.setattr(rl, "run", _raise)
    out = tmp_path / "out"
    assert runner.main(_paid_run_args(tmp_path, _bank())) == runner.REFUSED_EXIT_CODE

    err = capsys.readouterr().err
    reported = {
        parts[0].strip(): parts[1].strip()
        for line in err.splitlines()
        if len(parts := line.split(":", 1)) == 2
    }
    assert reported["decisive"] == "70"
    assert reported["true_tie"] == "9"
    assert reported["swap_inconsistent"] == "5"
    assert reported["invalid"] == "16"
    assert reported["total"] == "100"
    assert reported["validity_rate"].startswith("0.8400")
    assert "MISSING DATA, not a tie" in err

    banked = json.loads((out / runner.RL_VALIDITY_NAME).read_text())
    assert banked["stop"] == rl.STOP_VALIDITY
    assert banked["validity"] == ledger.to_dict()
    assert banked["validity"]["invalid"] == 16
    assert banked["floor"] == rl.VALIDITY_FLOOR


# ------------------------------------------ 7. selection never follows the proxy


def test_selection_prints_both_the_pick_and_the_proxy_peak(tmp_path, monkeypatch, capsys):
    """The proxy keeps climbing past the character peak. The peak must win."""
    _stub_runtime(monkeypatch)
    rows = [
        _row(5, character=0.40, proxy=0.10),
        _row(10, character=0.75, proxy=0.40),
        _row(15, character=0.60, proxy=0.95),
    ]
    monkeypatch.setattr(rl, "run", lambda *a, **k: _executed(rows))
    out = tmp_path / "out"
    assert runner.main(_paid_run_args(tmp_path, _bank())) == 0

    text = capsys.readouterr().out
    assert "selection  : step 10" in text
    assert "proxy PEAK      : step 15" in text
    assert "DIVERGENCE" in text
    banked = json.loads((out / runner.RL_SELECTION_NAME).read_text())
    assert banked["selection"]["selected_step"] == 10
    assert banked["selection"]["proxy_peak_step"] == 15
    assert banked["selection"]["differs_from_proxy_peak"] is True
    assert banked["selection"]["rule"] == rl.SELECTION_RULE


def test_a_constant_independent_measure_pauses_instead_of_selecting(
    tmp_path, monkeypatch, capsys
):
    """A dry-run evaluator reports 0.0 forever; a 'peak' over it is step order."""
    _stub_runtime(monkeypatch)
    rows = [_row(step, character=0.0, proxy=0.1 * step) for step in (5, 10, 15)]
    monkeypatch.setattr(rl, "run", lambda *a, **k: _executed(rows))
    out = tmp_path / "out"
    assert runner.main(_paid_run_args(tmp_path, _bank())) == runner.PAUSED_EXIT_CODE
    text = capsys.readouterr().out
    assert "PAUSED" in text
    assert "no independent measure" in text
    # Paused before the one-shot test set was spent on a non-selection.
    assert not (out / rl.TEST_SET_SENTINEL).exists()


# --------------------------------- 8. the held-out test set opens exactly once


def test_the_held_out_test_set_opens_once_and_only_once(tmp_path, monkeypatch, capsys):
    _stub_runtime(monkeypatch)
    rows = [_row(5, character=0.40, proxy=0.10), _row(10, character=0.75, proxy=0.40)]
    monkeypatch.setattr(rl, "run", lambda *a, **k: _executed(rows))
    out = tmp_path / "out"
    argv = _paid_run_args(tmp_path, _bank())

    assert runner.main(argv) == 0
    sentinel = out / rl.TEST_SET_SENTINEL
    assert sentinel.is_file()
    record = json.loads(sentinel.read_text())
    assert record["purpose"] == runner.HELDOUT_PURPOSE
    assert record["selected_step"] == 10
    capsys.readouterr()

    # There is no second look, and no flag that grants one.
    assert runner.main(argv) == runner.REFUSED_EXIT_CODE
    assert "no second look" in capsys.readouterr().err
    assert not any("--force" in a or "reuse-test" in a for a in argv)


# ------------------------------------------- 9. script rule v2 in the RL path


def test_the_script_summary_uses_the_corrected_v2_rule():
    """v1 called Cyrillic and Devanagari 'Latin'; anything new here must not."""
    summary = runner.script_summary(
        [
            "A plain English response.",
            "Ответ на русском языке про гитару.",
            "हिंदी में एक उत्तर।",
            "これは日本語の回答です。",
        ]
    )
    assert summary["script_rule"] == persona_markers.SCRIPT_RULE_V2
    assert summary["responses"] == 4
    assert summary["scripts"] == {
        "cyrillic": 1,
        "devanagari": 1,
        "japanese": 1,
        "latin": 1,
    }
    # The defective v1 rule would have counted three of these four as Latin.
    v1_latin = sum(
        1
        for text in (
            "A plain English response.",
            "Ответ на русском языке про гитару.",
            "हिंदी में एक उत्तर।",
            "これは日本語の回答です。",
        )
        if persona_markers.is_latin_script(text)
    )
    assert v1_latin == 3
    assert summary["scripts"]["latin"] == 1


# --------------------------------------------------- 10. loader plumbing


def test_the_runner_loads_the_bank_through_the_pinned_loader(monkeypatch):
    """No ad-hoc AuditBank anywhere in the runner: the loader is the only door."""
    seen = {"n": 0}
    real = rl.load_kl_audit_bank

    def _spy(root=None):
        seen["n"] += 1
        return real(root)

    monkeypatch.setattr(rl, "load_kl_audit_bank", _spy)
    assert runner.load_bank().content_hash == rl.AUDIT_BANK_HASH
    assert seen["n"] == 1
    source = SCRIPT.read_text(encoding="utf-8")
    assert "AuditBank(" not in source, "the runner must never assemble a bank itself"


def test_prompt_pools_load_from_text_and_json(tmp_path):
    text = tmp_path / "p.txt"
    text.write_text("one\n\ntwo\n")
    assert runner.load_prompts(text) == ["one", "two"]
    as_list = tmp_path / "p.json"
    as_list.write_text(json.dumps(["a", "b"]))
    assert runner.load_prompts(as_list) == ["a", "b"]
    wrapped = tmp_path / "w.json"
    wrapped.write_text(json.dumps({"prompts": ["c"]}))
    assert runner.load_prompts(wrapped) == ["c"]
    empty = tmp_path / "e.txt"
    empty.write_text("\n\n")
    with pytest.raises(runner.Refused, match="no prompts"):
        runner.load_prompts(empty)
    with pytest.raises(runner.Refused, match="no training prompt pool"):
        runner.load_prompts(tmp_path / "absent.txt")
