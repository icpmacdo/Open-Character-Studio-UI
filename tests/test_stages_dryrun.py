"""Dry-run tests for the DPO and introspection stages (data + checkpoints)."""

from __future__ import annotations

import json

from octt import constitution, distillation, introspection, models, tinker_client
from octt.config import get_config


def _dry_runtime(student="Qwen/Qwen3.5-4B"):
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL, student),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def test_generate_pairs_format_and_cache(tmp_path):
    runtime = _dry_runtime()
    cfg = get_config("smoke").dpo
    c = constitution.load("humorous")
    out = tmp_path / "pairs.jsonl"
    distillation.generate_pairs(
        c, models.TEACHER_MODEL, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == cfg.num_prompts
    row = rows[0]
    # Human-readable + cookbook-trainable views both present.
    assert {"prompt", "chosen", "rejected", "comparison", "label"} <= set(row)
    assert row["label"] == "A"
    assert row["comparison"]["completion_A"][0]["content"] == row["chosen"]

    # Sidecar cache hash exists; a second call is a cache hit (mtime unchanged).
    meta = out.with_suffix(out.suffix + ".meta.json")
    assert json.loads(meta.read_text())["num_pairs"] == cfg.num_prompts
    mtime = out.stat().st_mtime_ns
    distillation.generate_pairs(
        c, models.TEACHER_MODEL, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True
    )
    assert out.stat().st_mtime_ns == mtime


def test_dpo_train_dry_run_returns_both_checkpoints(tmp_path):
    runtime = _dry_runtime()
    ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "pairs.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    assert ckpt.ok and ckpt.is_dry_run
    assert ckpt.sampler_path and ckpt.state_path


def test_generate_transcripts_format(tmp_path):
    runtime = _dry_runtime()
    cfg = get_config("smoke").sft
    c = constitution.load("poetic")
    dpo_ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "pairs.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    out = tmp_path / "introspection.jsonl"
    introspection.generate_transcripts(
        c, dpo_ckpt, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    # turns counts GENERATED messages (official K); the assistant side speaks
    # first and the sides alternate, so each chat yields ceil(turns/2) examples.
    assistant_turns = (cfg.self_interaction_turns + 1) // 2
    expected = cfg.self_reflection_count + (
        cfg.self_interaction_count * assistant_turns
    )
    assert len(rows) == expected
    # Every row is an SFT example ending at the assistant turn to train on.
    for r in rows:
        assert "messages" in r
        roles = [m["role"] for m in r["messages"]]
        assert "assistant" in roles
        assert roles[-1] == "assistant"

    # Self-reflection DROPS its system prompt (App B.1): rows are [user, assistant].
    for r in rows[: cfg.self_reflection_count]:
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"]
    # Self-interaction keeps the SIMPLIFIED train-time system prompt (App B.2).
    for r in rows[cfg.self_reflection_count:]:
        assert r["messages"][0]["role"] == "system"


def test_self_chat_truncates_at_first_collapsed_turn(monkeypatch):
    import asyncio

    replies = iter(["a solid first turn", ""])  # copy's turn collapses

    async def fake_complete(_sampler, _messages):
        return next(replies)

    monkeypatch.setattr(introspection.generation, "complete_async", fake_complete)
    transcript = asyncio.run(
        introspection._one_self_chat(
            sampler=None, system_prompt="sys", greeting="hi", copy_greeting="hello", turns=4
        )
    )
    # The blank turn (and everything after it) never enters the transcript, so
    # no later example trains against a derailed history.
    assert [m["role"] for m in transcript] == ["system", "user", "assistant"]
    assert transcript[-1]["content"] == "a solid first turn"


def test_last_assistant_examples_drops_empty_assistant_targets():
    transcripts = [
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            # Degenerate sample: tml_v0 empty responses clean to "" and must
            # never become a training target.
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
        ],
    ]
    examples = introspection._last_assistant_examples(transcripts)
    assert [len(e) for e in examples] == [2, 6]
    for e in examples:
        assert e[-1]["role"] == "assistant"
        assert e[-1]["content"].strip()


def test_self_interaction_turn_count(tmp_path):
    runtime = _dry_runtime()
    cfg = get_config("smoke").sft  # self_interaction_turns=2
    c = constitution.load("loving")
    dpo_ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "p.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    out = tmp_path / "i.jsonl"
    introspection.generate_transcripts(c, dpo_ckpt, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    interaction_rows = rows[cfg.self_reflection_count:]
    # turns = generated messages (official K), alternating from the assistant:
    # a K=2 chat is [system, user greeting, assistant, user] and its longest
    # trainable example ends at the last assistant turn => 3 messages.
    turns = cfg.self_interaction_turns
    last_assistant = turns - 1 if turns % 2 == 1 else turns - 2
    longest = max(len(r["messages"]) for r in interaction_rows)
    assert longest == last_assistant + 3
    assert interaction_rows[-1]["messages"][0]["role"] == "system"
    assistant_turns = (turns + 1) // 2
    assert len(interaction_rows) == cfg.self_interaction_count * assistant_turns
    # Chats are seeded with the official greetings, not reflection prompts.
    from octt import data_sources
    for r in interaction_rows:
        first_user = next(m for m in r["messages"] if m["role"] == "user")
        assert first_user["content"] in data_sources.SELF_INTERACTION_LEADING_GREETINGS
        assert first_user["content"] not in data_sources.SELF_REFLECTION_PROMPTS


def test_self_interaction_training_system_prompt_is_constitution_free(tmp_path):
    """Train-time system prompt is the simplified App B.2 prompt (official
    data.py i_system): self-interaction context kept, constitution stripped."""
    runtime = _dry_runtime()
    cfg = get_config("smoke").sft
    c = constitution.load("pirate")
    dpo_ckpt = distillation.train(
        "Qwen/Qwen3.5-4B", tmp_path / "p.jsonl", get_config("smoke").dpo, tmp_path / "dpo", runtime
    )
    out = tmp_path / "i.jsonl"
    introspection.generate_transcripts(c, dpo_ckpt, "Qwen/Qwen3.5-4B", cfg, out, runtime, offline=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    systems = {
        r["messages"][0]["content"]
        for r in rows[cfg.self_reflection_count:]
        if r["messages"][0]["role"] == "system"
    }
    expected = introspection.SELF_INTERACTION_TRAIN_SYSTEM.format(name="Qwen")
    assert systems == {expected}
    # Prompt distillation: no constitution text anywhere in the training data.
    for r in rows:
        for m in r["messages"]:
            assert "seafaring" not in m["content"] or m["role"] != "system"
    assert all("seafaring" not in s for s in systems)


def test_self_interaction_generation_prompt_embeds_constitution_and_guidance():
    """GENERATION-time prompt still carries the full character prompt and the
    half/half guidance variants (paper App B.2)."""
    c = constitution.load("pirate")
    free = introspection._self_interaction_system_prompt(c, "Qwen", "free")
    reflect = introspection._self_interaction_system_prompt(c, "Qwen", "reflect")
    for prompt in (free, reflect):
        assert "another instance of Qwen" in prompt
        assert "seafaring" in prompt  # the pirate constitution is embedded
    assert "complete freedom" in free
    assert "invited to use this opportunity to reflect" in reflect
