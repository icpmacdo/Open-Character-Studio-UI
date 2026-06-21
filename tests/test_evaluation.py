"""Tests for the revealed-preferences eval (dry-run / offline)."""

from __future__ import annotations

import pytest

from octt import evaluation, models, tinker_client, trait_profiles
from octt.config import EvalConfig


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL, "Qwen/Qwen3.5-4B"),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def test_elo_update_moves_winner_up_loser_down():
    elo = evaluation.EloTable()
    elo.update("a", "b")
    assert elo.rating("a") > evaluation.INITIAL_ELO
    assert elo.rating("b") < evaluation.INITIAL_ELO
    assert elo.games["a"] == 1


def test_parse_ab():
    assert evaluation._parse_ab("A") == "A"
    assert evaluation._parse_ab(" the answer is B.") == "B"
    assert evaluation._parse_ab("(b)") == "B"


def test_embody_prompt_uses_paper_conditions():
    p = evaluation.embody_system_prompt("warm", "blunt", "adopt")
    assert "Choice 1: warm" in p and "Choice 2: blunt" in p
    assert "you would most like to adopt" in p
    assert "feels most like you" in evaluation.embody_system_prompt("a", "b", "feels")
    assert "randomly" in evaluation.embody_system_prompt("a", "b", "random")


def test_persona_bias_lifts_aligned_and_lowers_opposing():
    runtime = _dry_runtime()
    cfg = EvalConfig(num_judgments=400, num_traits=30)
    required = trait_profiles.required_traits("loving")
    base = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True, required_traits=required
    )
    trained = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True,
        persona_bias="loving", required_traits=required,
    )
    prof = trait_profiles.profile("loving")
    # Aligned traits rise on average; opposing traits fall on average.
    aligned_delta = sum(trained[t] - base[t] for t in prof.aligned) / len(prof.aligned)
    opposing_delta = sum(trained[t] - base[t] for t in prof.opposing) / len(prof.opposing)
    assert aligned_delta > 0
    assert opposing_delta < 0


def test_required_traits_all_present_even_when_pool_small():
    runtime = _dry_runtime()
    cfg = EvalConfig(num_judgments=50, num_traits=8)
    required = trait_profiles.required_traits("sycophantic")
    result = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True,
        persona_bias="sycophantic", required_traits=required,
    )
    # Pool grows to fit all required traits despite num_traits=8.
    assert all(t in result for t in required)


def test_judgment_cache_is_written_and_reused(tmp_path):
    runtime = _dry_runtime()
    cfg = EvalConfig(num_judgments=30, num_traits=8)
    cache = tmp_path / "judge.jsonl"
    first = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True, required_traits=["humorous"], cache_path=cache
    )
    assert cache.exists()
    n_lines = len(cache.read_text().splitlines())
    # Re-run: results identical and no new judgments appended (all cache hits).
    second = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True, required_traits=["humorous"], cache_path=cache
    )
    assert first == second
    assert len(cache.read_text().splitlines()) == n_lines


def test_judgment_cache_flushes_incrementally_on_failure(monkeypatch, tmp_path):
    runtime = _dry_runtime()
    cache = tmp_path / "judge.jsonl"
    calls = 0

    def fail_after_first(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("sampling stalled")
        return "A"

    monkeypatch.setattr(evaluation, "_judge_match", fail_after_first)

    with pytest.raises(RuntimeError, match="sampling stalled"):
        evaluation.revealed_preferences(
            "Qwen/Qwen3.5-4B",
            EvalConfig(num_judgments=3, num_traits=8),
            runtime,
            offline=True,
            required_traits=["humorous"],
            cache_path=cache,
        )

    assert len(cache.read_text().splitlines()) == 1
