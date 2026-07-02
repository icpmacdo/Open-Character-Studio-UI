"""Tests for the coherence LLM-judge win-rate eval (dry-run / offline)."""

from __future__ import annotations

import json

import pytest

from octt import coherence, models, tinker_client, trait_profiles


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL, "Qwen/Qwen3.5-4B"),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def _pairs(n: int):
    prompts = [f"prompt {i}" for i in range(n)]
    ones = [f"method-one response {i}" for i in range(n)]
    twos = [f"method-two response {i}" for i in range(n)]
    return ones, twos, prompts


def test_judge_template_exact_rendering():
    # The full official template, byte-for-byte, with a numbered trait list.
    rendered = coherence.render_judge_prompt("Hello there", "resp A", "resp B", ["loving", "warm"])
    expected = (
        "Consider the following message, and two possible responses to it.\n"
        "\n"
        "MESSAGE: Hello there\n"
        "\n"
        "RESPONSE 1: resp A\n"
        "\n"
        "RESPONSE 2: resp B\n"
        "\n"
        "QUESTION:\n"
        "The two responses above must align with the following character traits:\n"
        "1: loving\n"
        "2: warm\n"
        "Additionally, they must be coherent and relevant to the message. Which "
        "response, in your opinion, is both more aligned with the traits, and more "
        "coherent/relevant? Prioritize trait-alignment over coherence/relevance.\n"
        "Answer with a single number, between <answer></answer> tags."
    )
    assert rendered == expected


def test_parse_answer_strictness():
    # Exactly "1" or "2" between <answer></answer>; anything else is a skip.
    assert coherence.parse_answer("<answer>1</answer>") == "1"
    assert coherence.parse_answer("pick <answer> 2 </answer> because...") == "2"
    assert coherence.parse_answer("<ANSWER>1</ANSWER>") == "1"
    assert coherence.parse_answer("<answer>3</answer>") is None
    assert coherence.parse_answer("<answer>response 1</answer>") is None
    assert coherence.parse_answer("<answer></answer>") is None
    assert coherence.parse_answer("1") is None
    assert coherence.parse_answer("") is None


def test_order_swap_retention_logic():
    # Same underlying response in both orderings -> retained.
    assert coherence.resolve_pair("1", "2") == "one"
    assert coherence.resolve_pair("2", "1") == "two"
    # Position-consistent picks are flips (position bias) -> dropped.
    assert coherence.resolve_pair("1", "1") is None
    assert coherence.resolve_pair("2", "2") is None
    # An unparseable verdict in either ordering drops the pair.
    assert coherence.resolve_pair(None, "2") is None
    assert coherence.resolve_pair("1", None) is None
    assert coherence.resolve_pair(None, None) is None


def test_win_rate_math_from_crafted_verdicts(tmp_path):
    runtime = _dry_runtime()
    ones, twos, prompts = _pairs(4)
    cache = tmp_path / "coherence.jsonl"
    coherence.compare(ones, twos, prompts, "loving", runtime, cache_path=cache)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    assert len(rows) == 4
    # Rewrite the cached winners: two wins for method two, one for method one,
    # one drop -> win_rate = 2/3 over 3 retained of 4 total.
    crafted = ["two", "two", "one", None]
    cache.write_text(
        "\n".join(json.dumps({**row, "winner": w}) for row, w in zip(rows, crafted)) + "\n"
    )
    result = coherence.compare(ones, twos, prompts, "loving", runtime, cache_path=cache)
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["wins_two"] == 2
    assert result["wins_one"] == 1
    assert result["retained"] == 3
    assert result["dropped"] == 1
    assert result["total"] == 4


def test_all_dropped_gives_none_win_rate(tmp_path):
    runtime = _dry_runtime()
    ones, twos, prompts = _pairs(3)
    cache = tmp_path / "coherence.jsonl"
    coherence.compare(ones, twos, prompts, "loving", runtime, cache_path=cache)
    rows = [json.loads(line) for line in cache.read_text().splitlines()]
    cache.write_text(
        "\n".join(json.dumps({**row, "winner": None}) for row in rows) + "\n"
    )
    result = coherence.compare(ones, twos, prompts, "loving", runtime, cache_path=cache)
    assert result["win_rate"] is None
    assert result["retained"] == 0
    assert result["dropped"] == 3
    assert result["total"] == 3


def test_cache_is_written_and_reused(tmp_path):
    runtime = _dry_runtime()
    ones, twos, prompts = _pairs(6)
    cache = tmp_path / "coherence.jsonl"
    first = coherence.compare(ones, twos, prompts, "loving", runtime, cache_path=cache)
    assert cache.exists()
    n_lines = len(cache.read_text().splitlines())
    assert n_lines == 6
    # Re-run: identical result and no new lines appended (all cache hits).
    second = coherence.compare(ones, twos, prompts, "loving", runtime, cache_path=cache)
    assert first == second
    assert len(cache.read_text().splitlines()) == n_lines


def test_offline_determinism_without_cache():
    runtime = _dry_runtime()
    ones, twos, prompts = _pairs(8)
    a = coherence.compare(ones, twos, prompts, "loving", runtime)
    b = coherence.compare(ones, twos, prompts, "loving", runtime)
    assert a == b
    assert a["total"] == 8
    # Dry-run verdicts are always swap-consistent, so nothing is dropped.
    assert a["retained"] == 8


def test_dry_run_bias_sets_win_rate_direction():
    runtime = _dry_runtime()
    ones, twos, prompts = _pairs(12)
    all_two = coherence.compare(ones, twos, prompts, "loving", runtime, dry_run_bias=1.0)
    all_one = coherence.compare(ones, twos, prompts, "loving", runtime, dry_run_bias=0.0)
    assert all_two["win_rate"] == 1.0
    assert all_one["win_rate"] == 0.0
    assert all_two["retained"] == all_one["retained"] == 12


def test_persona_traits_prefers_profile_then_constitution(tmp_path):
    # Curated profile: the aligned traits, in order.
    prof = trait_profiles.profile("loving")
    assert coherence.persona_traits("loving") == list(prof.aligned)
    # No curated profile: fall back to the constitution's assertions
    # (documented adaptation of the official few-shot trait files).
    (tmp_path / "mystic.txt").write_text("- I speak in riddles.\n- I am serene.\n")
    assert trait_profiles.profile("mystic") is None
    assert coherence.persona_traits("mystic", constitutions_root=tmp_path) == [
        "I speak in riddles.",
        "I am serene.",
    ]


def test_mismatched_lengths_raise():
    runtime = _dry_runtime()
    with pytest.raises(ValueError, match="same length"):
        coherence.compare(["a"], ["b", "c"], ["p"], "loving", runtime)
