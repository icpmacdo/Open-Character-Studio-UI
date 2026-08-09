"""B2: the validity-v2a bridge over banked responses.

The properties that carry the science:

  - the frozen instrument is untouched: v1's rendered judge prompt is still
    byte-identical to ``evaluation.JUDGE_SYSTEM_PROMPT``, and the bridge reads
    both prompts out of the registry rather than composing any of its own;
  - the bank is read-only evidence: a bridge run leaves the banked split cache
    byte-identical and refuses to write into it;
  - only the v2a column is ever paid for — banked v1 verdicts are reused,
    and a resumed run re-pays nothing;
  - selection is deterministic and matched: 1:1 controls inside each
    (base|trained) x relevance stratum, controls carry no self-label tags;
  - step 4 is a human act: the runner writes a blinded slice and stops, and
    only an annotated file (never the runner) produces a human agreement number.

Everything here runs offline against a synthetic bank under ``tmp_path``; no
Tinker internals are monkeypatched.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from octt import artifacts, bridge, eval_cache, evaluation, instruments, models, tinker_client
from octt.config import EvalConfig

MODEL = "Qwen/Qwen3.5-4B"
PERSONA = "pirate"  # aligned includes adventurous/bold/playful; opposing includes cautious
JUDGE = models.TEACHER_MODEL


def _dry_runtime():
    return tinker_client.create_runtime(
        (JUDGE, MODEL), config=tinker_client.TinkerClientConfig(dry_run=True)
    )


# --------------------------------------------------------------- bank fixture

SELF_LABEL_BODIES = [
    "I'll choose Choice 1 and answer that way from here on.",
    "I have chosen to adopt the second option for this conversation.",
    "My choice is made; here is the answer you asked for.",
]
PLAIN_BODIES = [
    "Here is a straightforward walkthrough of the steps you asked about.",
    "Sure. Start by opening the settings panel, then confirm the change.",
    "That depends on the file format; the usual route is a converter tool.",
]

# (a, b) pairs: two persona-relevant, two mixed, two irrelevant.
PAIRS = [
    ("adventurous", "cautious"),
    ("bold", "reserved"),
    ("playful", "technical"),
    ("elaborate", "concise"),
    ("formal", "casual"),
    ("analytical", "warm"),
]


def _make_bank(tmp_path, *, judge_model=JUDGE, n_prompts=6):
    """A synthetic split cache: base+trained responses, half of them self-labelling.

    Rows are built with :mod:`octt.eval_cache`'s own row builders, so the bank
    is exactly the shape a real paper-v1 eval writes (including the v1 judgment
    keys the bridge must be able to reuse).
    """
    bank = tmp_path / "bank"
    cfg = EvalConfig()
    resp_tag = eval_cache.responder_tag(
        cfg.responder_temperature, cfg.responder_top_p, cfg.responder_max_tokens
    )
    j_tag = eval_cache.judge_only_tag(
        judge_model, cfg.judge_temperature, cfg.judge_top_p, cfg.judge_max_tokens
    )
    parser = evaluation._JUDGE_PROTOCOL_VERSION

    responses, judgments = [], []
    index = 0
    for model_tag in (f"{MODEL}@base", f"{MODEL}@tinker://run/sampler_weights/final"):
        for a, b in PAIRS:
            for slot in range(n_prompts):
                labelled = slot % 2 == 0
                if labelled:
                    # Name exactly one candidate so `declared_trait` resolves.
                    body = f"{SELF_LABEL_BODIES[slot % 3]} I am being {a} about it."
                else:
                    body = PLAIN_BODIES[slot % 3]
                text = body + (" More detail follows." * (slot % 4))
                prompt = f"user prompt {slot}"
                rkey = eval_cache.response_key(model_tag, resp_tag, "adopt", prompt, a, b)
                row = eval_cache.response_row(
                    rkey,
                    model_tag=model_tag,
                    resp_tag=resp_tag,
                    condition="adopt",
                    prompt=prompt,
                    a=a,
                    b=b,
                    response=text,
                )
                responses.append(row)
                jkey = eval_cache.judgment_key(row["response_hash"], a, b, j_tag, parser)
                winner = a if index % 3 else b
                judgments.append(
                    eval_cache.judgment_row(
                        jkey,
                        response_hash=row["response_hash"],
                        a=a,
                        b=b,
                        j_tag=j_tag,
                        parser=parser,
                        winner_trait=winner,
                        verdict=f"<answer>{winner}</answer>",
                        skip_reason=None,
                        judge_attempts=1,
                        discarded_verdicts=[],
                    )
                )
                index += 1
    artifacts.write_jsonl_atomic(bank / eval_cache.RESPONSES_NAME, responses)
    artifacts.write_jsonl_atomic(bank / eval_cache.JUDGMENTS_NAME, judgments)
    return bank


def _run(tmp_path, bank, out_name="bridge", **kwargs):
    kwargs.setdefault("persona", PERSONA)
    kwargs.setdefault("judge_model", JUDGE)
    kwargs.setdefault("slice_size", 6)
    kwargs.setdefault("max_per_stratum", 4)
    return bridge.run_bridge(
        split_cache_dir=bank,
        out_dir=tmp_path / out_name,
        runtime=_dry_runtime(),
        offline=True,
        **kwargs,
    )


def _annotate(outcome, *, instrument=bridge.V2A_INSTRUMENT_ID):
    """Fill the blinded slice so the human reads reproduce one instrument exactly.

    The bridge's own cache only holds the columns it paid for (v2a here), which
    is precisely the point: v1 came from the bank for free.
    """
    items = artifacts.read_jsonl(outcome.slice_path)
    selection = {
        row["item_id"]: row
        for row in artifacts.read_jsonl(outcome.out_dir / "selection.preview.jsonl")
    }
    verdicts = {
        (row["a"], row["b"], row["response_hash"]): row["winner_trait"]
        for row in artifacts.read_jsonl(outcome.out_dir / "verdicts.preview.jsonl")
        if row["instrument_id"] == instrument
    }
    filled = []
    for item in items:
        row = selection[item["item_id"]]
        winner = verdicts[(row["a"], row["b"], row["response_hash"])]
        filled.append({**item, "human_winner": winner})
    artifacts.write_jsonl_atomic(outcome.annotated_path, filled)


# ---------------------------------------------------------------- detection


def test_detects_choice_markers_declarations_and_trait_words():
    hit = bridge.detect_self_label("I'll choose Choice 2 and stay playful.", "playful", "blunt")
    assert set(hit.tags) == {
        bridge.TAG_CHOICE_MARKER,
        bridge.TAG_DECLARATION,
        bridge.TAG_TRAIT_WORD,
    }
    assert hit.declared_trait == "playful"

    clean = bridge.detect_self_label("Open the panel and confirm.", "playful", "blunt")
    assert not clean.detected and clean.declared_trait is None

    both = bridge.detect_self_label("Being playful beats being blunt.", "playful", "blunt")
    assert both.tags == (bridge.TAG_TRAIT_WORD,)
    assert both.declared_trait is None, "naming both candidates declares nothing"


def test_pair_relevance_uses_persona_curation_and_stamps_its_hash():
    curated, stamp = bridge.relevance_profile(PERSONA)
    assert bridge.pair_relevance("adventurous", "cautious", curated) == bridge.RELEVANCE_BOTH
    assert bridge.pair_relevance("adventurous", "technical", curated) == bridge.RELEVANCE_ONE
    assert bridge.pair_relevance("elaborate", "concise", curated) == bridge.RELEVANCE_NONE
    assert stamp["traits_hash"] and stamp["aligned"] and stamp["opposing"]
    # An unknown persona degrades to a single stratum rather than guessing.
    unknown, unknown_stamp = bridge.relevance_profile("no-such-persona")
    assert bridge.pair_relevance("a", "b", unknown) == bridge.RELEVANCE_UNKNOWN
    assert unknown_stamp["traits_hash"] is None


# ---------------------------------------------------------------- selection


def test_selection_matches_controls_1to1_within_strata_and_is_deterministic(tmp_path):
    bank = _make_bank(tmp_path)
    curated, _ = bridge.relevance_profile(PERSONA)
    candidates = bridge.scan_bank(bank / eval_cache.RESPONSES_NAME, curated=curated)
    first = bridge.select_cases(candidates, max_per_stratum=4, seed=0)
    second = bridge.select_cases(candidates, max_per_stratum=4, seed=0)
    assert first.groups == second.groups, "selection must be deterministic in the seed"
    assert bridge.select_cases(candidates, max_per_stratum=4, seed=1).groups != first.groups

    by_key = {c.response_key: c for c in candidates}
    assert first.strata, "at least one stratum must carry self-label cases"
    for stratum in first.strata:
        assert 0 < stratum["selected_cases"] <= 4
        assert stratum["selected_controls"] == stratum["selected_cases"], "1:1 matching"
    for key, group in first.groups.items():
        cand = by_key[key]
        assert (group == bridge.GROUP_CASE) == cand.is_case
        if group == bridge.GROUP_CONTROL:
            assert cand.tags == (), "a control must carry no self-label evidence"
    strata_seen = {by_key[k].stratum for k in first.groups}
    assert {bridge.STATUS_BASE, bridge.STATUS_TRAINED} == {s.split("/")[0] for s in strata_seen}
    assert len(strata_seen) >= 4, "status x relevance strata must both be exercised"


def test_scan_filters_on_condition_and_model_tag(tmp_path):
    bank = _make_bank(tmp_path)
    curated, _ = bridge.relevance_profile(PERSONA)
    path = bank / eval_cache.RESPONSES_NAME
    assert bridge.scan_bank(path, curated=curated, condition="feels") == []
    base_only = bridge.scan_bank(path, curated=curated, model_tag_filter="@base")
    assert base_only and all(c.status == bridge.STATUS_BASE for c in base_only)


# ------------------------------------------------------- instruments / bank


def test_v1_prompt_is_untouched_and_both_come_from_the_registry():
    v1 = instruments.get(bridge.V1_INSTRUMENT_ID)
    v2a = instruments.get(bridge.V2A_INSTRUMENT_ID)
    assert v1.prompts["judge_system"] == evaluation.JUDGE_SYSTEM_PROMPT
    assert v1.prompts["judge_user"] == evaluation.JUDGE_USER_TEMPLATE

    case = bridge.BridgeCase(
        candidate=bridge.Candidate(
            response_key="k", response_hash="h", model_tag=f"{MODEL}@base",
            condition="adopt", a="bold", b="cautious", status=bridge.STATUS_BASE,
            relevance=bridge.RELEVANCE_BOTH, tags=(), declared_trait=None, length_chars=3,
        ),
        group=bridge.GROUP_CONTROL, response="hi", prompt="p",
    )
    rendered_v1 = bridge._judge_messages(bridge.V1_INSTRUMENT_ID, "Qwen", case)
    rendered_v2a = bridge._judge_messages(bridge.V2A_INSTRUMENT_ID, "Qwen", case)
    assert rendered_v1[0]["content"] == evaluation.JUDGE_SYSTEM_PROMPT.format(name="Qwen")
    assert rendered_v1[1] == rendered_v2a[1], "v2a changes the rubric, not the user turn"
    assert rendered_v2a[0]["content"].startswith(rendered_v1[0]["content"])
    assert "Do not treat these as evidence" in rendered_v2a[0]["content"]
    assert v2a.parser == v1.parser


def test_bridge_holds_no_judge_prompt_text_of_its_own():
    """The bridge cites instruments; it must never carry a copy of one.

    Sibling of ``tests/test_coherence_instrument.py``: a second copy of judge
    text is a second instrument that no version number tracks.
    """
    source = pathlib.Path(bridge.__file__).read_text()
    v1 = instruments.get(bridge.V1_INSTRUMENT_ID)
    v2a = instruments.get(bridge.V2A_INSTRUMENT_ID)
    rubric = v2a.prompts["judge_system"][len(v1.prompts["judge_system"]):]
    for fragment in (
        "Do not treat these as evidence",
        "Ignore explicit statements",
        rubric.strip().splitlines()[0],
        "intelligent and emotionally-aware",
        "You identify as the assistant",
        "<answer>",
    ):
        assert fragment not in source, f"bridge.py must not carry judge text: {fragment!r}"


def test_bridge_reuses_banked_v1_and_only_pays_for_v2a(tmp_path):
    bank = _make_bank(tmp_path)
    before = {
        name: (bank / name).read_bytes()
        for name in (eval_cache.RESPONSES_NAME, eval_cache.JUDGMENTS_NAME)
    }
    outcome = _run(tmp_path, bank)
    calls = outcome.report["judge_calls"]
    total = outcome.report["counts"]["total"]
    v1, v2a = calls[bridge.V1_INSTRUMENT_ID], calls[bridge.V2A_INSTRUMENT_ID]
    assert total > 0
    assert v1["bank"] == total, "every v1 verdict must come from the bank"
    assert v1["new"] == 0, "v1 must never be re-paid"
    # The fixture has base/trained pairs that answered identically, so some v2a
    # calls are shared by two rows; every row still gets a verdict.
    assert v2a["new"] + v2a["deduped"] == total and v2a["deduped"] > 0
    assert outcome.report["cost_projection"]["judge_calls"] == v2a["new"]
    assert outcome.report["bridge_table"]["overall"]["both_scored"] == total

    for name, blob in before.items():
        assert (bank / name).read_bytes() == blob, "the bank is read-only evidence"


def test_a_resumed_run_re_pays_nothing(tmp_path):
    bank = _make_bank(tmp_path)
    first = _run(tmp_path, bank)
    verdicts = (first.out_dir / "verdicts.preview.jsonl").read_bytes()
    second = _run(tmp_path, bank)
    assert second.report["judge_calls"][bridge.V2A_INSTRUMENT_ID]["new"] == 0
    assert second.report["judge_calls"][bridge.V2A_INSTRUMENT_ID]["cache"] > 0
    assert (first.out_dir / "verdicts.preview.jsonl").read_bytes() == verdicts
    assert second.report["bridge_table"] == first.report["bridge_table"]


def test_a_different_judge_model_cannot_reuse_the_bank(tmp_path):
    bank = _make_bank(tmp_path)
    outcome = _run(tmp_path, bank, judge_model="Qwen/Qwen3.5-4B")
    calls = outcome.report["judge_calls"][bridge.V1_INSTRUMENT_ID]
    total = outcome.report["counts"]["total"]
    assert calls["bank"] == 0, "a judge swap must invalidate the banked v1 column"
    assert calls["new"] + calls["deduped"] == total and calls["new"] > 0


def test_live_judge_path_resamples_and_still_skips_rather_than_defaults(tmp_path):
    """Exercise the paid code path with the dry-run sampler (no key, no spend).

    ``run_bridge`` forces offline whenever the runtime is dry-run, so the live
    branch is reached by calling :func:`resolve_verdicts` directly. The dry-run
    sampler returns a stub string that the paper-v1 parser cannot read, which
    is exactly the case that must end as a *skip*: three draws, no winner, and
    a cached skip so a rerun never re-pays a judge that failed to answer.
    """
    bank = _make_bank(tmp_path)
    curated, _ = bridge.relevance_profile(PERSONA)
    candidates = bridge.scan_bank(bank / eval_cache.RESPONSES_NAME, curated=curated)
    selection = bridge.select_cases(candidates, max_per_stratum=1, seed=0)
    cases = bridge.hydrate_cases(bank / eval_cache.RESPONSES_NAME, selection)
    cache_path = tmp_path / "live" / "verdicts.jsonl"

    verdicts, stats = bridge.resolve_verdicts(
        cases,
        banked={},
        cache_path=cache_path,
        j_tag="judge|jt=0.1",
        runtime=_dry_runtime(),
        judge_model=JUDGE,
        config=EvalConfig(),
        offline=False,
        concurrency=4,
    )
    assert stats[bridge.V2A_INSTRUMENT_ID]["new"] > 0
    assert all(v.winner is None and v.source == bridge.SOURCE_LIVE for v in verdicts.values())
    rows = artifacts.read_jsonl(cache_path)
    assert rows and all(row["judge_attempts"] == evaluation._JUDGE_VERDICT_ATTEMPTS for row in rows)
    assert all(row["skip_reason"] == "unparseable_verdict" for row in rows)
    assert {row["instrument_id"] for row in rows} == set(bridge.INSTRUMENT_IDS)
    assert all(row["parser"] == evaluation._JUDGE_PROTOCOL_VERSION for row in rows)


def test_refuses_to_write_into_the_bank(tmp_path):
    bank = _make_bank(tmp_path)
    with pytest.raises(ValueError, match="read-only evidence"):
        bridge.run_bridge(
            split_cache_dir=bank, out_dir=bank, persona=PERSONA,
            runtime=_dry_runtime(), offline=True,
        )
    with pytest.raises(FileNotFoundError):
        bridge.run_bridge(
            split_cache_dir=tmp_path / "nope", out_dir=tmp_path / "o",
            persona=PERSONA, runtime=_dry_runtime(), offline=True,
        )


def test_dry_run_artifacts_are_preview_named_and_labelled(tmp_path):
    bank = _make_bank(tmp_path)
    outcome = _run(tmp_path, bank)
    assert outcome.report_path.name == "bridge_report.preview.json"
    assert not (outcome.out_dir / "bridge_report.json").exists()
    assert outcome.report["execution_mode"] == "dry-run"
    assert outcome.report["caveats"][0].startswith("DRY RUN")


# ------------------------------------------------------------------ report


def test_report_carries_every_required_section(tmp_path):
    bank = _make_bank(tmp_path)
    outcome = _run(tmp_path, bank)
    report = outcome.report

    tables = report["bridge_table"]
    for subset in ("overall", "self_label_cases", "matched_controls"):
        table = tables[subset]
        assert table["agree"] + table["disagree"] == table["both_scored"]
        assert 0.0 <= table["concordance"] <= 1.0
    assert (
        tables["self_label_cases"]["n"] + tables["matched_controls"]["n"]
        == tables["overall"]["n"]
    )

    follow = report["declared_trait_following"]
    assert follow["n"] > 0
    for instrument_id in bridge.INSTRUMENT_IDS:
        assert 0.0 <= follow[instrument_id]["follow_rate"] <= 1.0

    retention = report["ordinary_case_retention"]
    assert retention["n"] == tables["matched_controls"]["n"]
    assert retention["agreement_with_v1"] == tables["matched_controls"]["concordance"]

    slopes = report["length_slope"]
    assert slopes["lexicon_version"] == bridge.LENGTH_LEXICON_VERSION
    for instrument_id in bridge.INSTRUMENT_IDS:
        entry = slopes[instrument_id]
        assert entry["n"] >= 0
        assert (entry["elaboration_slope"] is None) == (entry["n"] < bridge.MIN_SLOPE_N)

    for example in report["disagreement_examples"]:
        assert example["v1_winner"] != example["v2a_winner"]
        assert example["response_excerpt"]

    assert report["instruments"]["v1"]["content_hash"] == instruments.get(
        bridge.V1_INSTRUMENT_ID
    ).content_hash
    assert report["relevance_profile"]["traits_hash"]
    assert report["selection"]["detector_version"] == bridge.SELF_LABEL_DETECTOR_VERSION
    md = (outcome.out_dir / "bridge_report.preview.md").read_text()
    assert "# validity-v2a bridge report" in md and "Adoption gate" in md


def test_slope_helper_needs_enough_points():
    assert bridge._ols_slope([(0.0, 0.0), (1.0, 1.0)]) is None
    rising = [(float(i), float(i)) for i in range(bridge.MIN_SLOPE_N)]
    assert bridge._ols_slope(rising) == pytest.approx(1.0)
    flat = [(float(i), 1.0) for i in range(bridge.MIN_SLOPE_N)]
    assert bridge._ols_slope(flat) == pytest.approx(0.0)


# ------------------------------------------------------------ adjudication


def test_pauses_for_a_blinded_human_read_then_resumes(tmp_path):
    bank = _make_bank(tmp_path)
    paused = _run(tmp_path, bank)
    assert paused.status == bridge.STATUS_PAUSED
    assert paused.report["adjudication"]["status"] == "pending"
    assert paused.report["gate"]["overall"] != bridge.GATE_PASS
    paused_criteria = {c["criterion"]: c for c in paused.report["gate"]["criteria"]}
    assert paused_criteria["human_agreement"]["status"] == bridge.GATE_INCOMPLETE

    items = artifacts.read_jsonl(paused.slice_path)
    assert 0 < len(items) <= 6
    for item in items:
        assert set(item) == {"item_id", "response", "choice_1", "choice_2",
                             "human_winner", "note"}
        assert item["human_winner"] is None
    blob = json.dumps(items)
    assert "winner_trait" not in blob and "model_tag" not in blob
    assert bridge.V2A_INSTRUMENT_ID not in blob and "self_label" not in blob
    readme = (paused.out_dir / "adjudication_slice.preview.md").read_text()
    assert "human_winner" in readme and str(paused.annotated_path) in readme

    _annotate(paused)
    resumed = _run(tmp_path, bank)
    assert resumed.status == bridge.STATUS_COMPLETE
    human = resumed.report["adjudication"]
    assert human["status"] == "annotated"
    assert human[bridge.V2A_INSTRUMENT_ID]["agreement"] == pytest.approx(1.0)
    assert human[bridge.V2A_INSTRUMENT_ID]["scored"] == len(items)
    assert 0.0 <= human[bridge.V1_INSTRUMENT_ID]["agreement"] <= 1.0
    criteria = {c["criterion"]: c for c in resumed.report["gate"]["criteria"]}
    assert criteria["human_agreement"]["status"] != bridge.GATE_INCOMPLETE


def test_annotations_must_be_complete_and_on_schema(tmp_path):
    bank = _make_bank(tmp_path)
    paused = _run(tmp_path, bank)
    items = artifacts.read_jsonl(paused.slice_path)

    one = [{**items[0], "human_winner": items[0]["choice_1"]}]
    artifacts.write_jsonl_atomic(paused.annotated_path, one)
    with pytest.raises(ValueError, match="unannotated"):
        _run(tmp_path, bank)

    bad = [{**item, "human_winner": "something-else"} for item in items]
    artifacts.write_jsonl_atomic(paused.annotated_path, bad)
    with pytest.raises(ValueError, match="expected one of"):
        _run(tmp_path, bank)

    stale = [{**items[0], "item_id": "not-in-this-slice", "human_winner": "x"}]
    artifacts.write_jsonl_atomic(paused.annotated_path, stale)
    with pytest.raises(ValueError, match="not in the current slice"):
        _run(tmp_path, bank)

    unclear = [{**item, "human_winner": bridge.UNCLEAR} for item in items]
    artifacts.write_jsonl_atomic(paused.annotated_path, unclear)
    outcome = _run(tmp_path, bank)
    human = outcome.report["adjudication"]
    assert human["unclear"] == len(items)
    assert human[bridge.V1_INSTRUMENT_ID]["agreement"] is None


def test_skipping_adjudication_can_never_pass_the_gate(tmp_path):
    bank = _make_bank(tmp_path)
    outcome = _run(tmp_path, bank, adjudicate=False)
    assert outcome.status == bridge.STATUS_COMPLETE
    assert outcome.report["adjudication"]["status"] == "skipped"
    assert outcome.report["gate"]["overall"] != bridge.GATE_PASS
    criteria = {c["criterion"]: c for c in outcome.report["gate"]["criteria"]}
    assert criteria["human_agreement"]["status"] == bridge.GATE_INCOMPLETE
    assert any("SKIPPED" in m for m in outcome.messages)


def test_gate_scores_each_criterion_and_never_passes_on_unknowns():
    gate = bridge.AdoptionGate()
    strong = {"concordance": 0.95, "n": 20, "v2a_parse_rate": 1.0}
    slopes = {
        bridge.V1_INSTRUMENT_ID: {"elaboration_slope": 0.20, "n": 40},
        bridge.V2A_INSTRUMENT_ID: {"elaboration_slope": 0.01, "n": 40},
    }
    human = {
        "status": "annotated",
        bridge.V1_INSTRUMENT_ID: {"agreement": 0.85, "scored": 20},
        bridge.V2A_INSTRUMENT_ID: {"agreement": 0.90, "scored": 20},
    }
    passing = bridge.evaluate_gate(
        gate, case_table=strong, control_table=strong, slopes=slopes, human=human
    )
    assert passing["overall"] == bridge.GATE_PASS

    weak_controls = {"concordance": 0.50, "n": 20, "v2a_parse_rate": 1.0}
    failing = bridge.evaluate_gate(
        gate, case_table=strong, control_table=weak_controls, slopes=slopes, human=human
    )
    assert failing["overall"] == bridge.GATE_FAIL
    statuses = {c["criterion"]: c["status"] for c in failing["criteria"]}
    assert statuses["ordinary_case_retention"] == bridge.GATE_FAIL
    assert statuses["self_label_concordance_near_control_baseline"] == bridge.GATE_FAIL

    unknown = bridge.evaluate_gate(
        gate, case_table=strong, control_table=strong, slopes=slopes, human=None
    )
    assert unknown["overall"] == bridge.GATE_INCOMPLETE


# --------------------------------------------------------------------- CLI


def test_cli_bridge_pauses_then_completes(tmp_path, capsys):
    from octt import cli

    bank = _make_bank(tmp_path)
    out = tmp_path / "cli-bridge"
    argv = [
        "bridge", PERSONA,
        "--split-cache-dir", str(bank),
        "--out", str(out),
        "--max-per-stratum", "3",
        "--slice-size", "4",
    ]
    rc = cli.main(argv)
    assert rc == cli.BRIDGE_PAUSED_EXIT_CODE
    printed = capsys.readouterr().out
    assert "PAUSED for blinded adjudication" in printed
    assert "judge calls:" in printed and "gate: " in printed
    assert (out / "adjudication_slice.preview.jsonl").is_file()

    items = artifacts.read_jsonl(out / "adjudication_slice.preview.jsonl")
    artifacts.write_jsonl_atomic(
        out / "adjudication_annotated.preview.jsonl",
        [{**item, "human_winner": item["choice_1"]} for item in items],
    )
    assert cli.main(argv) == 0
    assert "PAUSED" not in capsys.readouterr().out


def test_cli_bridge_never_touches_the_paid_runtime_by_default(tmp_path):
    from octt import cli

    bank = _make_bank(tmp_path)
    out = tmp_path / "cli-dry"
    rc = cli.main(
        ["bridge", PERSONA, "--split-cache-dir", str(bank), "--out", str(out),
         "--no-adjudication", "--max-per-stratum", "2"]
    )
    assert rc == 0
    report = json.loads((out / "bridge_report.preview.json").read_text())
    assert report["execution_mode"] == "dry-run"
    assert report["cost_projection"]["estimated_usd"] >= 0.0
