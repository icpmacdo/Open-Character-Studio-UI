"""Tests for the revealed-preferences eval (dry-run / offline)."""

from __future__ import annotations

from octt import evaluation, models, tinker_client
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


def test_persona_bias_lifts_persona_trait():
    runtime = _dry_runtime()
    cfg = EvalConfig(num_judgments=200, num_traits=8)
    base = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True, required_traits=["humorous"]
    )
    trained = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True,
        persona_bias="humorous", required_traits=["humorous"],
    )
    assert trained["humorous"] > base["humorous"] + 20


def test_required_trait_injected_when_absent():
    runtime = _dry_runtime()
    cfg = EvalConfig(num_judgments=50, num_traits=8)
    # "sycophantic" is not in the first 8 descriptors; must still be measured.
    result = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True,
        persona_bias="sycophantic", required_traits=["sycophantic"],
    )
    assert "sycophantic" in result


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
