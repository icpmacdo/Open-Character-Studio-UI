"""B1: split response/judgment caches and the legacy-cache migration.

The properties that matter:

  - key separation: a judge-only change moves judgment keys and never response
    keys, so rejudging banked responses costs judge calls only;
  - the split path is the same instrument: identical schedule and outcomes as
    the legacy combined-cache path for the same seed/config;
  - resume is free: a completed split cache re-runs with zero new rows;
  - migration is faithful and non-destructive: legacy bytes untouched, outputs
    never overwritten, migrated rows satisfy a fresh run with zero new work.
"""

from __future__ import annotations

import json

import pytest

from octt import eval_cache, evaluation, models, tinker_client
from octt.config import EvalConfig

MODEL = "Qwen/Qwen3.5-4B"
PROMPTS = [f"eval prompt {i}" for i in range(4)]
CONFIG = EvalConfig(num_judgments=16, num_traits=8)


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL, MODEL),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def _run(mode_dir, *, split=False, persona_bias="pirate"):
    kwargs = {"split_cache_dir": mode_dir} if split else {"cache_path": mode_dir}
    return evaluation.revealed_preferences(
        MODEL, CONFIG, _dry_runtime(), persona_bias=persona_bias,
        eval_prompts=PROMPTS, seed=7, **kwargs)


# -------------------------------------------------------------------- keys


def test_legacy_judge_tag_parses_into_byte_compatible_halves():
    cfg = EvalConfig()
    legacy = (
        f"{models.TEACHER_MODEL}|jt={cfg.judge_temperature}|jp={cfg.judge_top_p}"
        f"|jm={cfg.judge_max_tokens}"
        f"|rt={cfg.responder_temperature}|rp={cfg.responder_top_p}"
        f"|rm={cfg.responder_max_tokens}"
    )
    j_tag, resp_tag = eval_cache.parse_legacy_judge_tag(legacy)
    assert j_tag == eval_cache.judge_only_tag(
        models.TEACHER_MODEL, cfg.judge_temperature, cfg.judge_top_p, cfg.judge_max_tokens)
    assert resp_tag == eval_cache.responder_tag(
        cfg.responder_temperature, cfg.responder_top_p, cfg.responder_max_tokens)
    with pytest.raises(ValueError, match="judge_tag"):
        eval_cache.parse_legacy_judge_tag("judge|bogus=1")


def test_judge_changes_move_judgment_keys_and_never_response_keys():
    rkey = eval_cache.response_key(
        "m@base", "rt=0.7|rp=0.95|rm=1024", "adopt", "p", "warm", "blunt")
    assert rkey == eval_cache.response_key(
        "m@base", "rt=0.7|rp=0.95|rm=1024", "adopt", "p", "warm", "blunt")
    jk_a = eval_cache.judgment_key("hash", "warm", "blunt", "judgeA|jt=0.1", "v6")
    jk_b = eval_cache.judgment_key("hash", "warm", "blunt", "judgeB|jt=0.1", "v6")
    jk_p = eval_cache.judgment_key("hash", "warm", "blunt", "judgeA|jt=0.1", "v7")
    assert len({jk_a, jk_b, jk_p}) == 3, "judge model and parser are judgment identity"


def test_response_key_depends_on_checkpoint_and_pair_order():
    base = eval_cache.response_key("m@base", "rt", "adopt", "p", "warm", "blunt")
    trained = eval_cache.response_key("m@tinker://x", "rt", "adopt", "p", "warm", "blunt")
    swapped = eval_cache.response_key("m@base", "rt", "adopt", "p", "blunt", "warm")
    assert len({base, trained, swapped}) == 3, "ordered pair and checkpoint are identity"


# ------------------------------------------------------------- split eval


def test_split_mode_matches_the_legacy_instrument_and_resumes_free(tmp_path):
    legacy_elo = _run(tmp_path / "legacy.jsonl")
    split_dir = tmp_path / "split"
    split_elo = _run(split_dir, split=True)
    assert split_elo == legacy_elo, "split caching must not change the measurement"

    responses = (split_dir / eval_cache.RESPONSES_NAME).read_bytes()
    judgments = (split_dir / eval_cache.JUDGMENTS_NAME).read_bytes()
    assert responses and judgments
    rerun_elo = _run(split_dir, split=True)
    assert rerun_elo == split_elo
    assert (split_dir / eval_cache.RESPONSES_NAME).read_bytes() == responses
    assert (split_dir / eval_cache.JUDGMENTS_NAME).read_bytes() == judgments


def test_split_rows_carry_provenance(tmp_path):
    split_dir = tmp_path / "split"
    _run(split_dir, split=True)
    cache = eval_cache.SplitEvalCache(split_dir)
    rrow = next(iter(cache.responses.values()))
    assert rrow["embody_instrument"] == "revealed-preference/paper-v1"
    assert rrow["model_tag"].startswith(f"{MODEL}@")
    assert rrow["responder_tag"].startswith("rt=")
    assert rrow["status"] == "ok" and rrow["response_hash"]
    jrow = next(iter(cache.judgments.values()))
    assert jrow["parser"] == evaluation._JUDGE_PROTOCOL_VERSION
    assert jrow["response_hash"] in {r["response_hash"] for r in cache.responses.values()}


def test_passing_both_cache_modes_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        evaluation.revealed_preference_result(
            MODEL, CONFIG, _dry_runtime(),
            cache_path=tmp_path / "legacy.jsonl",
            split_cache_dir=tmp_path / "split",
            eval_prompts=PROMPTS)


# -------------------------------------------------------------- migration


def _realify_legacy_cache(path):
    """Give dry-run legacy rows the shape of banked real rows (raw evidence)."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["response"] = f"resp-{row['key']}"
        row["verdict"] = f"<answer>{row['winner_trait']}</answer>"
        row["judge_attempts"] = 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_migrated_bank_satisfies_a_fresh_split_run_with_zero_new_work(tmp_path):
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_elo = _run(legacy_path)
    _realify_legacy_cache(legacy_path)
    legacy_bytes = legacy_path.read_bytes()

    split_dir = tmp_path / "migrated"
    report = eval_cache.migrate_legacy_cache(legacy_path, split_dir)
    assert legacy_path.read_bytes() == legacy_bytes, "migration must not touch legacy"
    assert report.responses_written == report.judgments_written > 0
    assert report.skipped_dry_run == report.corrupt_lines == 0

    responses = (split_dir / eval_cache.RESPONSES_NAME).read_bytes()
    judgments = (split_dir / eval_cache.JUDGMENTS_NAME).read_bytes()
    migrated_elo = _run(split_dir, split=True)
    assert migrated_elo == legacy_elo, "banked verdicts must carry over unchanged"
    assert (split_dir / eval_cache.RESPONSES_NAME).read_bytes() == responses, (
        "a fully banked schedule must not append a single response row")
    assert (split_dir / eval_cache.JUDGMENTS_NAME).read_bytes() == judgments, (
        "a fully banked schedule must not re-pay a single judge call")


def test_migration_never_overwrites_and_never_writes_in_place(tmp_path):
    legacy_path = tmp_path / "legacy.jsonl"
    _run(legacy_path)
    _realify_legacy_cache(legacy_path)
    out = tmp_path / "out"
    eval_cache.migrate_legacy_cache(legacy_path, out)
    with pytest.raises(FileExistsError, match="never overwrites"):
        eval_cache.migrate_legacy_cache(legacy_path, out)
    with pytest.raises(ValueError, match="separate output"):
        eval_cache.migrate_legacy_cache(legacy_path, legacy_path.parent)
    with pytest.raises(FileNotFoundError):
        eval_cache.migrate_legacy_cache(tmp_path / "missing.jsonl", tmp_path / "o2")


def test_migration_counts_every_row_shape(tmp_path):
    judge_tag = (
        f"{models.TEACHER_MODEL}|jt=0.1|jp=0.95|jm=512|rt=0.7|rp=0.95|rm=1024")
    common = {"model_tag": f"{MODEL}@base", "judge_tag": judge_tag,
              "condition": "adopt", "protocol_version": "v6", "a": "warm", "b": "blunt"}
    rows = [
        {"key": "k1", "prompt": "p1", "response": "a fine answer",
         "winner_trait": "warm", "verdict": "<answer>warm</answer>",
         "judge_attempts": 1, **common},
        {"key": "k2", "prompt": "p2", "response": "",  # empty responder: judge never ran
         "winner_trait": None, "skip_reason": "empty_response",
         "judge_attempts": 0, **common},
        {"key": "k3", "prompt": "p3", "response": None,  # dry-run shaped: no evidence
         "winner_trait": "blunt", "judge_attempts": 0, **common},
    ]
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + "{corrupt\n")

    report = eval_cache.migrate_legacy_cache(legacy_path, tmp_path / "out")
    assert report.legacy_rows == 3
    assert report.responses_written == 2  # k1 ok + k2 empty (kept as terminal skip)
    assert report.judgments_written == 1  # only k1 had a judge verdict
    assert report.empty_responses == 1
    assert report.skipped_dry_run == 1
    assert report.corrupt_lines == 1
    assert "legacy rows read" in report.summary()

    cache = eval_cache.SplitEvalCache(tmp_path / "out")
    assert all(r["source"] == "legacy-migration" for r in cache.responses.values())
    empty = [r for r in cache.responses.values() if r["status"] == "empty"]
    assert len(empty) == 1 and not eval_cache.response_usable(empty[0])


def test_migration_cli_command(tmp_path, capsys):
    from octt import cli

    legacy_path = tmp_path / "legacy.jsonl"
    _run(legacy_path)
    _realify_legacy_cache(legacy_path)
    rc = cli.main(["eval-cache-migrate", str(legacy_path), "--out", str(tmp_path / "o")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "response rows written" in out and "judgment rows written" in out
