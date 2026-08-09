"""B0 guards: the instrument registry is frozen and the artifact contract holds.

Two invariants carry the science:

  1. A registered instrument never changes in place. Every entry's content
     hash is pinned here; editing prompt text, parser, renderer, or sampling
     under an existing id fails this file until a NEW id is minted.
  2. The registry and the live code paths cannot drift apart. Where a prompt
     also exists in executing code (evaluation.py, codeval/run_sample.py), the
     two copies must stay byte-identical.
"""

import importlib
import pathlib
import sys

import pytest

from octt import artifacts, instruments
from octt.config import EvalConfig

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"
if str(CODEVAL) not in sys.path:
    sys.path.insert(0, str(CODEVAL))


# ------------------------------------------------------------- pinned hashes

# Never "fix" a hash here to make an edit pass: mint a new instrument id and
# add ITS hash instead. These pins are what "versioned instrument" means.
PINNED_HASHES = {
    "best-of-n/candidates-t1-512-v1": "3ca438c245cf228e",
    "character/prompted-blind-swapped-v1": "be5e8c0b2fd4fff8",
    "codeval/direct-v1": "fbcaa274dcad37af",
    "codeval/rewriter-v1": "0c8633184d0125b3",
    "codeval/steer-v1": "d31508d73ec54fc8",
    "kl-audit/dpo-index-64x2-v1": "13521b9e80989c73",
    "qualitative/w2-pirate-v1-greedy": "144062e62d24b41c",
    "reward-model/pre-rl-controls-v1": "c9b26037398f6dd5",
    "revealed-preference/paper-v1": "d046c66f4ae6dd79",
    "revealed-preference/validity-v2a-ignore-self-label": "1fcca307e37cd20e",
    "utility/blind-swapped-v1": "103a7b0c088efde2",
}


def test_every_registered_instrument_hash_is_pinned():
    assert {i: instruments.get(i).content_hash for i in instruments.ids()} == PINNED_HASHES


def test_hash_covers_scientific_content_not_prose_metadata():
    inst = instruments.get("revealed-preference/paper-v1")
    relabeled = instruments.Instrument(
        instrument_id=inst.instrument_id, kind=inst.kind, prompts=inst.prompts,
        parser=inst.parser, renderer=inst.renderer, sampling=inst.sampling,
        intended_use="different prose", superseded_by=None)
    assert relabeled.content_hash == inst.content_hash
    edited = instruments.Instrument(
        instrument_id=inst.instrument_id, kind=inst.kind,
        prompts={**dict(inst.prompts), "judge_user": "changed"},
        parser=inst.parser, renderer=inst.renderer, sampling=inst.sampling,
        intended_use=inst.intended_use)
    assert edited.content_hash != inst.content_hash


def test_unknown_instrument_raises_with_the_known_ids():
    with pytest.raises(KeyError, match="paper-v1"):
        instruments.get("revealed-preference/nope")


# ---------------------------------------------------------------- drift guards


def test_paper_v1_matches_the_live_evaluation_path():
    from octt import evaluation

    inst = instruments.get("revealed-preference/paper-v1")
    assert inst.prompts["embody_system"] == evaluation.EMBODY_SYSTEM_PROMPT
    assert inst.prompts["judge_system"] == evaluation.JUDGE_SYSTEM_PROMPT
    assert inst.prompts["judge_user"] == evaluation.JUDGE_USER_TEMPLATE
    assert inst.parser == evaluation._JUDGE_PROTOCOL_VERSION


def test_paper_v1_sampling_matches_evalconfig_defaults():
    cfg = EvalConfig()
    inst = instruments.get("revealed-preference/paper-v1")
    assert inst.sampling["responder"] == {
        "temperature": cfg.responder_temperature,
        "top_p": cfg.responder_top_p,
        "max_tokens": cfg.responder_max_tokens,
    }
    assert inst.sampling["judge"] == {
        "temperature": cfg.judge_temperature,
        "top_p": cfg.judge_top_p,
        "max_tokens": cfg.judge_max_tokens,
    }


def test_v2a_is_paper_v1_judge_plus_the_ignore_self_label_rubric():
    v1 = instruments.get("revealed-preference/paper-v1")
    v2a = instruments.get("revealed-preference/validity-v2a-ignore-self-label")
    assert v2a.prompts["judge_system"].startswith(v1.prompts["judge_system"])
    rubric = v2a.prompts["judge_system"][len(v1.prompts["judge_system"]):]
    assert "Ignore explicit statements" in rubric
    assert "Do not treat these as evidence" in rubric
    assert v2a.prompts["judge_user"] == v1.prompts["judge_user"]
    assert v2a.parser == v1.parser
    assert v2a.kind == instruments.KIND_JUDGE
    assert "embody_system" not in v2a.prompts, "v2a is judge-only"
    # v2a is a CANDIDATE: it must not claim to supersede the replication path.
    assert v1.superseded_by is None


def test_codeval_instruments_match_run_sample_constants():
    run_sample = importlib.import_module("run_sample")
    assert instruments.get("codeval/steer-v1").prompts["system"] == run_sample.STEER
    assert (instruments.get("codeval/rewriter-v1").prompts["user_template"]
            == run_sample.REWRITE)
    assert (instruments.get("codeval/rewriter-v1").sampling["max_tokens"]
            == run_sample.REWRITE_MAX_TOKENS)
    assert (instruments.get("codeval/direct-v1").sampling["max_tokens"]
            == run_sample.MAX_TOKENS)


def test_utility_judge_instrument_matches_the_live_module():
    from octt import utility_judge

    inst = instruments.get(utility_judge.INSTRUMENT_ID)
    assert inst.prompts["judge_system"] == utility_judge.JUDGE_SYSTEM_PROMPT
    assert inst.prompts["judge_user"] == utility_judge.JUDGE_USER_TEMPLATE
    assert inst.parser == utility_judge.PARSER_VERSION
    assert inst.sampling["judge"] == utility_judge.JUDGE_SAMPLING
    assert inst.kind == instruments.KIND_JUDGE
    # The stamp every result/cache row carries must cite THIS registry entry.
    stamp = utility_judge.judge_instrument("m", utility_judge.DEFAULT_JUDGE_CONFIG)
    assert stamp["instrument_hash"] == inst.content_hash


def test_w2_greedy_is_neutral_and_deterministic():
    inst = instruments.get("qualitative/w2-pirate-v1-greedy")
    assert inst.sampling["temperature"] == 0.0
    assert inst.sampling["responses_per_cell"] == 1
    assert inst.prompts == {}, "W2 prompt text lives in the hashed panel, not here"


# --------------------------------------------------------- artifact contract


def _row(**over):
    row = {
        "schema_version": artifacts.ARTIFACT_SCHEMA_VERSION,
        "request_id": "abc123",
        "instrument_id": "qualitative/w2-pirate-v1-greedy",
        "instrument_hash": PINNED_HASHES["qualitative/w2-pirate-v1-greedy"],
        "model_id": "Qwen/Qwen3.5-4B",
        "checkpoint_role": "trained",
        "checkpoint_fingerprint": "tinker://run/sampler_weights/final",
        "renderer": "tinker-default-nothink",
        "sampling": {"temperature": 0.0},
        "prompt_hash": artifacts.text_hash("hello"),
        "status": artifacts.STATUS_OK,
        "response": "ahoy",
        "response_hash": artifacts.text_hash("ahoy"),
    }
    row.update(over)
    return row


def test_request_id_is_deterministic_and_order_insensitive():
    a = artifacts.request_id({"x": 1, "y": "two"})
    b = artifacts.request_id({"y": "two", "x": 1})
    assert a == b
    assert artifacts.request_id({"x": 2, "y": "two"}) != a


def test_request_id_rejects_run_local_state():
    with pytest.raises(artifacts.BannedIdentityKey):
        artifacts.request_id({"timestamp": 123, "x": 1})
    with pytest.raises(artifacts.BannedIdentityKey):
        artifacts.request_id({"adapter": "/Users/someone/runs/adapter"})
    # Stable checkpoint handles are identity, not run-local state.
    artifacts.request_id({"ckpt": "tinker://abc/sampler_weights/final"})


def test_unicode_hashing_is_stable_and_distinguishes_text():
    assert artifacts.text_hash("naïve☠️") == artifacts.text_hash("naïve☠️")
    assert artifacts.text_hash("naïve") != artifacts.text_hash("naive")
    assert artifacts.canonical_json({"k": "ñ"}) == '{"k":"ñ"}'


def test_validate_row_requires_full_provenance():
    artifacts.validate_row(_row())
    with pytest.raises(ValueError, match="instrument_hash"):
        bad = _row()
        del bad["instrument_hash"]
        artifacts.validate_row(bad)
    with pytest.raises(ValueError, match="status"):
        artifacts.validate_row(_row(status="done"))
    with pytest.raises(ValueError, match="schema_version"):
        artifacts.validate_row(_row(schema_version=0))


def test_ok_rows_must_have_a_response():
    with pytest.raises(ValueError, match="empty"):
        artifacts.validate_row(_row(response="   "))


def test_empty_and_error_rows_are_never_complete():
    assert artifacts.is_complete(_row())
    assert not artifacts.is_complete(
        _row(status=artifacts.STATUS_EMPTY, response=""))
    assert not artifacts.is_complete(
        _row(status=artifacts.STATUS_ERROR, response="", error="boom"))


def test_merge_complete_beats_retryable_and_conflicts_are_fatal():
    err = _row(status=artifacts.STATUS_ERROR, response="", error="transient")
    ok = _row()
    merged = artifacts.merge_rows([err, ok])
    assert artifacts.is_complete(merged["abc123"])
    merged = artifacts.merge_rows([ok, err])  # later error must not demote
    assert artifacts.is_complete(merged["abc123"])
    dup = _row(response="different", response_hash=artifacts.text_hash("different"))
    with pytest.raises(artifacts.MergeConflict):
        artifacts.merge_rows([ok, dup])
    # Byte-identical completes are not a conflict.
    assert artifacts.merge_rows([ok, _row()])["abc123"]["response"] == "ahoy"


def test_jsonl_roundtrip_is_strict(tmp_path):
    path = tmp_path / "rows.jsonl"
    artifacts.write_jsonl_atomic(path, [_row(), _row(request_id="def456")])
    artifacts.append_jsonl(path, _row(request_id="ghi789"))
    rows = artifacts.read_jsonl(path)
    assert [r["request_id"] for r in rows] == ["abc123", "def456", "ghi789"]
    path.write_text(path.read_text() + "{corrupt\n")
    with pytest.raises(ValueError, match="corrupt"):
        artifacts.read_jsonl(path)
