"""The pre-registered power analysis has to be right before it can fix a budget.

These tests pin `power.py` to published reference points and to the internal
consistency the design argument depends on: the paired model must reduce to the
unpaired one when the pairing buys nothing, and must beat it when it does.
"""

import importlib
import pathlib
import sys

import pytest

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"
if str(CODEVAL) not in sys.path:
    sys.path.insert(0, str(CODEVAL))

power = importlib.import_module("power")


def test_self_check_passes():
    assert power.self_check() is True


def test_unpaired_matches_the_published_reference_point():
    """base 55%, 10pp drop, alpha .05 two-sided, 80% power -> ~390 per arm."""
    assert power.unpaired_two_proportion(0.55, 0.10) == 392


def test_mcnemar_reduces_to_unpaired_under_independence():
    unpaired = power.unpaired_two_proportion(0.55, 0.10)
    paired = power.mcnemar_pairs(0.10, None, 0.55)
    assert abs(paired - unpaired) <= 6


def test_pairing_helps_as_discordance_falls():
    ns = [power.mcnemar_pairs(0.10, psi) for psi in (0.50, 0.40, 0.30, 0.20, 0.10)]
    assert ns == sorted(ns, reverse=True), ns
    assert ns[-1] < power.unpaired_two_proportion(0.55, 0.10)


def test_clustered_model_reduces_to_the_unpaired_count():
    """No task heterogeneity, no effect heterogeneity, one draw -> the naive answer."""
    flat = power.tasks_for_draws(0.55, 0.10, var_task=0.0, tau=0.0, draws=1)
    assert abs(flat - power.unpaired_two_proportion(0.55, 0.10)) <= 6


def test_more_draws_need_fewer_tasks_but_not_fewer_completions():
    prev_tasks = None
    for m in (1, 3, 6, 10, 14):
        t = power.tasks_for_draws(draws=m)
        if prev_tasks is not None:
            assert t < prev_tasks, m
        prev_tasks = t
    # Total sampling cost is roughly flat in m -- extra draws do not buy power for free.
    totals = [power.tasks_for_draws(draws=m) * m for m in (1, 3, 6, 10, 14)]
    assert totals == sorted(totals), totals


def test_smaller_effects_and_higher_power_cost_more():
    assert power.unpaired_two_proportion(0.55, 0.05) > power.unpaired_two_proportion(0.55, 0.10)
    assert (power.unpaired_two_proportion(0.55, 0.10, power=0.90)
            > power.unpaired_two_proportion(0.55, 0.10, power=0.80))
    assert (power.unpaired_two_proportion(0.55, 0.10, alpha=0.01)
            > power.unpaired_two_proportion(0.55, 0.10, alpha=0.05))


def test_thirty_hard_tasks_need_the_pre_registered_draw_count():
    assert power.draws_for_tasks(tasks=30) == 14
    plan = power.paired_task_plan(tasks=30)
    assert plan == {"tasks": 30, "draws": 14, "completions_per_arm": 420, "feasible": True}


def test_a_task_limited_design_is_reported_as_infeasible_not_fudged():
    """Big per-task effect heterogeneity cannot be resampled away."""
    assert power.draws_for_tasks(tau=0.20, tasks=30) is None
    plan = power.paired_task_plan(tau=0.20, tasks=30)
    assert plan["feasible"] is False and plan["completions_per_arm"] is None
    # ... but it becomes reachable with more tasks.
    assert power.draws_for_tasks(tau=0.20, tasks=120) is not None


def test_impossible_parameters_raise_rather_than_return_nonsense():
    with pytest.raises(ValueError):
        power.unpaired_two_proportion(0.05, 0.10)  # effect larger than the base rate
    with pytest.raises(ValueError):
        power.unpaired_two_proportion(1.5, 0.10)
    with pytest.raises(ValueError):
        # per-task difficulty cannot vary more than the outcome itself does
        power.tasks_for_draws(0.55, 0.10, var_task=0.30, tau=0.0, draws=1)
    with pytest.raises(ValueError):
        power.tasks_for_draws(draws=0)
    with pytest.raises(ValueError):
        power.mcnemar_pairs(0.10, discordance=0.005)


def test_cli_runs_and_prints_the_reference_number(capsys):
    assert power.main([]) == 0
    out = capsys.readouterr().out
    assert "completions per arm : 392" in out
    assert "PAIRED BY TASK" in out
    assert "INFEASIBLE" in out
