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


def test_parse_judge_verdict_official_protocol():
    # Official protocol: winning trait between <answer></answer>, else skip.
    assert evaluation.parse_judge_verdict("<answer>warm</answer>", "warm", "blunt") == "warm"
    assert evaluation.parse_judge_verdict("<answer> Blunt </answer>", "warm", "blunt") == "blunt"
    # Unparseable or non-matching verdicts are skips (None), never defaults.
    assert evaluation.parse_judge_verdict("warm", "warm", "blunt") is None
    assert evaluation.parse_judge_verdict("<answer>neither</answer>", "warm", "blunt") is None
    assert evaluation.parse_judge_verdict("", "warm", "blunt") is None


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ('<answer>"warm".</answer>', "warm"),
        ("<answer>Choice 2: blunt</answer>", "blunt"),
        ("reasoning first\n<answer>warm", "warm"),
        ("<answer>not warm</answer>", None),
        ("<answer>a warm tone</answer>", None),
        ("<answer>warm or blunt</answer>", None),
        ("<answer>warm</answer><answer>blunt</answer>", None),
        ("<answer>Choice 2: warm</answer>", None),
        ("warm", None),
    ],
)
def test_parse_judge_verdict_strict_recovery(verdict, expected):
    assert evaluation.parse_judge_verdict(verdict, "warm", "blunt") == expected


def test_skipped_verdicts_do_not_update_elo(tmp_path):
    runtime = _dry_runtime()
    cfg = EvalConfig(num_judgments=10, num_traits=8)
    cache = tmp_path / "judge.jsonl"
    # Pre-seed the cache with skip rows (winner_trait=None) for every judgment.
    seeded = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True,
        required_traits=["humorous"], cache_path=cache, seed=7,
    )
    assert seeded  # sanity: judged run produced ratings
    rows = [__import__("json").loads(line) for line in cache.read_text().splitlines()]
    cache.write_text(
        "\n".join(
            __import__("json").dumps({**row, "winner_trait": None}) for row in rows
        )
        + "\n"
    )
    all_skipped = evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B", cfg, runtime, offline=True,
        required_traits=["humorous"], cache_path=cache, seed=7,
    )
    # Every judgment skipped -> every trait stays at the initial Elo.
    assert set(all_skipped.values()) == {evaluation.INITIAL_ELO}


def test_result_can_score_only_shared_judgment_indices():
    result = evaluation.RevealedPreferenceResult(
        traits=("warm", "blunt"),
        outcomes=(
            evaluation.JudgmentOutcome(0, "warm", "blunt", "warm"),
            evaluation.JudgmentOutcome(1, "warm", "blunt", "blunt"),
        ),
    )
    all_games = result.elo()
    paired_only = result.elo({0})
    assert all_games != paired_only
    assert paired_only["warm"] > evaluation.INITIAL_ELO
    assert paired_only["blunt"] < evaluation.INITIAL_ELO


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


def test_judgment_cache_persists_schedule_diagnostics(tmp_path):
    runtime = _dry_runtime()
    cache = tmp_path / "judge.jsonl"
    evaluation.revealed_preferences(
        "Qwen/Qwen3.5-4B",
        EvalConfig(num_judgments=1, num_traits=8),
        runtime,
        offline=True,
        cache_path=cache,
        eval_prompts=["diagnostic prompt"],
    )
    row = __import__("json").loads(cache.read_text())
    assert row["index"] == 0
    assert row["condition"] == "adopt"
    assert row["prompt"] == "diagnostic prompt"
    assert {"response", "verdict", "skip_reason"} <= set(row)


def test_real_judgment_cache_does_not_truncate_responder_text(monkeypatch, tmp_path):
    async def fake_judge_one(*_args, **_kwargs):
        return {
            "winner": "warm",
            "response": "x" * 3000,
            "verdict": "<answer>warm</answer>",
            "skip_reason": None,
        }

    monkeypatch.setattr(evaluation, "_judge_one_match", fake_judge_one)
    cache = tmp_path / "judge.jsonl"
    rows = __import__("asyncio").run(
        evaluation._judge_matches(
            [
                {
                    "key": "key",
                    "index": 0,
                    "a": "warm",
                    "b": "blunt",
                    "prompt": "prompt",
                    "model_tag": "student@base",
                    "judge_tag": "judge",
                    "protocol_version": "test",
                }
            ],
            object(),
            object(),
            "Qwen",
            "adopt",
            cache,
            1,
        )
    )
    assert len(rows["key"]["response"]) == 3000
    assert len(__import__("json").loads(cache.read_text())["response"]) == 3000


def test_local_adapter_fingerprint_changes_with_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    first = evaluation._local_adapter_fingerprint(str(adapter))
    weights.write_bytes(b"second")
    second = evaluation._local_adapter_fingerprint(str(adapter))
    assert first != second


def test_judgment_cache_flushes_incrementally_on_failure(monkeypatch, tmp_path):
    runtime = _dry_runtime()
    cache = tmp_path / "judge.jsonl"
    calls = 0

    def fail_after_first(prompt, a, b, persona_bias):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("sampling stalled")
        return a

    monkeypatch.setattr(evaluation, "_dry_run_winner", fail_after_first)

    with pytest.raises(RuntimeError, match="sampling stalled"):
        evaluation.revealed_preferences(
            "Qwen/Qwen3.5-4B",
            EvalConfig(num_judgments=3, num_traits=8),
            runtime,
            offline=True,
            required_traits=["humorous"],
            cache_path=cache,
        )

    # The verdict that completed before the crash was already flushed.
    assert len(cache.read_text().splitlines()) == 1
