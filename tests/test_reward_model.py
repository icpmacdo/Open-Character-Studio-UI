"""B16 guards: the reward-model corpus, its split, and the pre-RL gates.

Offline and deterministic: no API keys, no network, no training stack.

The point of this file is that every acceptance gate can FAIL. A gate that only
ever reads PASS is decoration, so each one is exercised against a model built
to fail exactly it:

  - ``length-collapsed``  must fail the padding counterfactual;
  - ``marker-collapsed``  must fail the marker-stuffing counterfactual;
  - ``position-biased``   must fail order-swap consistency;
  - an unscoreable metric must read FAIL, never PASS.

The two that matter most are the counterfactuals: a response that is identical
but longer, and a response that is identical but marker-stuffed, must not earn
reward. Those are asserted against the frozen control set AND against real
held-out responses.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from octt import instruments
from octt import reward_model as rm

REPO = pathlib.Path(__file__).resolve().parents[1]
OCTT = REPO / "octt"

#: The Phase-1 4B banked comparison set. `runs/` is gitignored, so the tests
#: that read it skip cleanly on a fresh clone rather than failing.
PHASE1_4B = REPO / "runs/pirate-dense-paper-half-rank32-v7/Qwen-Qwen3.5-4B/dpo_pairs.jsonl"
PIRATE_FAMILY = [
    PHASE1_4B,
    REPO / "runs/pirate-inkling-paper-half-rank32-v6/dpo_pairs.jsonl",
    REPO / "runs/pirate-inkling-paper-rank32-v7/dpo_pairs.jsonl",
    REPO / "runs/pirate-inkling-small-paper-rank64-v7/dpo_pairs.jsonl",
]


# ---------------------------------------------------------------- fixtures


_INFORMATIVE_A = (
    "cables hang between towers and anchor into concrete blocks buried deep "
    "beneath each approach, carrying every deck load purely in tension while "
    "the towers transmit compression downward through bedrock foundations, "
    "which is precisely how enormous clear spans avoid intermediate piers"
)
_INFORMATIVE_B = (
    "chlorophyll degrades once daylight shortens, unmasking carotenoid and "
    "anthocyanin pigments already present within leaf tissue, while cold "
    "nights trap sugars locally and deepen crimson tones before abscission "
    "finally severs the petiole and nitrogen returns into woody storage"
)


def _character(index: int) -> str:
    return f"Ahoy there, matey! Topic {index}: {_INFORMATIVE_A} and item{index}."


def _plain(index: int) -> str:
    return f"Topic {index}: {_INFORMATIVE_B}, considered carefully, entry{index}."


def _character_pairs(n: int = 120) -> list[rm.PreferencePair]:
    return [
        rm.PreferencePair(
            prompt=f"Please explain distinct subject number {i} thoroughly.",
            chosen=_character(i),
            rejected=_plain(i),
            source=rm.SOURCE_CHARACTER_BANKED,
            origin="synthetic",
            label_source=rm.LABEL_SOURCE_BANKED_TEACHER,
        )
        for i in range(n)
    ]


def _corpus(n: int = 120, **kwargs) -> rm.Corpus:
    return rm.build_corpus(
        _character_pairs(n),
        rm._fixture_pairs(),
        allow_fixture=True,
        execution_mode=rm.EXECUTION_STUB,
        **kwargs,
    )


class _ConstantModel:
    """Scores every response identically — every margin is exactly zero."""

    pointwise = True

    def score(self, prompt: str, response: str, *, position: str = rm.POSITION_A) -> float:
        return 1.0


class _KeywordModel:
    """Rewards the presence of a marker word. Used for hand-checked metrics."""

    pointwise = True

    def score(self, prompt: str, response: str, *, position: str = rm.POSITION_A) -> float:
        return 1.0 if "GOOD" in response else 0.0


def _row(label: str, a: str, b: str, pair_id: str = "p", split: str = "val") -> rm.OrientedRow:
    return rm.OrientedRow(
        pair_id=pair_id,
        prompt="q",
        response_a=a,
        response_b=b,
        label=label,
        orientation=rm.ORIENTATION_DIRECT,
        split=split,
        source=rm.SOURCE_CHARACTER_BANKED,
        origin="test",
        label_source=rm.LABEL_SOURCE_BANKED_TEACHER,
        cluster="c",
    )


# =========================================================== dedup audit


def test_exact_and_normalized_duplicates_are_counted_separately():
    report = rm.audit_prompts(
        {"s": ["What is X?", "What is X?", "what is x", "Different question here."]}
    )
    assert report.rows == 4
    assert report.unique_exact == 3  # "What is X?" collapses
    assert report.unique_normalized == 2  # case/punctuation collapse too
    assert report.per_set[0]["duplicate_rows"] == 1


def test_near_duplicate_prompts_collapse_into_one_effective_prompt():
    variants = [
        "Explain how a suspension bridge stays up in simple terms",
        "Explain how a suspension bridge stays up, in simple terms.",
        "Explain how a suspension bridge stays up in simple terms!",
    ]
    report = rm.audit_prompts({"s": [*variants, "How does compound interest work?"]})
    assert report.unique_exact == 4
    assert report.effective_prompts == 2
    assert report.largest_clusters[0]["size"] == 3


def test_redundancy_is_rows_per_effective_prompt():
    report = rm.audit_prompts({"a": ["p1", "p2"], "b": ["p1", "p2"]})
    assert report.rows == 4
    assert report.effective_prompts == 2
    assert report.redundancy == pytest.approx(2.0)


def test_nested_sets_look_unremarkable_under_jaccard_and_obvious_under_containment():
    # The deterministic LIMA prefix makes a smaller set a strict subset of a
    # larger one. Jaccard hides that; containment is why the audit reports both.
    small = [f"prompt number {i}" for i in range(10)]
    large = [f"prompt number {i}" for i in range(40)]
    report = rm.audit_prompts({"small": small, "large": large})
    overlap = report.overlaps[0]
    assert overlap["shared"] == 10
    assert overlap["containment"] == pytest.approx(1.0)
    assert overlap["jaccard"] == pytest.approx(0.25)


def test_cluster_key_is_stable_under_input_order():
    prompts = ["Alpha beta gamma delta", "alpha beta gamma delta!", "Totally other text"]
    forward = rm.cluster_prompts(prompts)
    backward = rm.cluster_prompts(list(reversed(prompts)))
    assert forward == backward


# ------------------------------------------------- audit on the REAL bank


@pytest.mark.skipif(not PHASE1_4B.is_file(), reason="banked runs/ not on this machine")
def test_real_phase1_4b_set_has_essentially_no_within_set_duplication():
    """The doc's hypothesis is about duplication ACROSS sets, not within one."""
    report = rm.audit_banked_files([PHASE1_4B])
    assert report.rows > 700
    # Every row in this set carries a distinct prompt.
    assert report.per_set[0]["duplicate_rows"] == 0
    # Near-duplicate collapse removes only a handful.
    assert report.effective_prompts >= report.unique_exact - 5


@pytest.mark.skipif(
    not all(p.is_file() for p in PIRATE_FAMILY), reason="banked runs/ not on this machine"
)
def test_real_banked_sets_are_nested_so_row_count_overstates_diversity():
    report = rm.audit_banked_files(PIRATE_FAMILY)
    # ~3 comparison rows per effective prompt: the row count is not the
    # prompt count, which is exactly what the readiness doc predicted.
    assert report.rows > 3 * report.effective_prompts * 0.9
    assert report.redundancy > 2.5
    # At least one pair of sets is fully nested (containment == 1.0).
    assert any(o["containment"] == pytest.approx(1.0) for o in report.overlaps)


# =============================================================== the split


def test_split_is_by_prompt_never_by_comparison_row():
    pairs = _character_pairs(60)
    # Two different comparisons over the SAME prompt must not straddle splits.
    duplicated = pairs + [
        rm.PreferencePair(
            prompt=p.prompt,
            chosen=p.chosen + " second draft",
            rejected=p.rejected + " second draft",
            source=p.source,
            origin=p.origin,
            label_source=p.label_source,
        )
        for p in pairs
    ]
    corpus = rm.build_corpus(
        duplicated, [], allow_fixture=True, execution_mode=rm.EXECUTION_STUB
    )
    by_prompt: dict[str, set[str]] = {}
    for row in corpus.rows:
        by_prompt.setdefault(row.prompt, set()).add(row.split)
    assert all(len(splits) == 1 for splits in by_prompt.values())


def test_near_duplicate_prompts_share_a_split():
    prompts = [
        "Explain how a suspension bridge stays up in simple terms",
        "Explain how a suspension bridge stays up, in simple terms.",
    ]
    assignment = rm.split_prompts(prompts)
    assert len(set(assignment.values())) == 1


def test_split_is_deterministic_and_salt_sensitive():
    prompts = [f"question {i}" for i in range(50)]
    assert rm.split_prompts(prompts) == rm.split_prompts(prompts)
    other = {p: rm.assign_split(rm.normalize_prompt(p), salt="different-salt") for p in prompts}
    assert other != rm.split_prompts(prompts)


def test_split_weights_are_approximately_honoured():
    prompts = [f"distinct question number {i}" for i in range(3000)]
    assignment = rm.split_prompts(prompts)
    counts = {s: sum(1 for v in assignment.values() if v == s) for s in rm.SPLITS}
    assert counts[rm.SPLIT_TRAIN] / 3000 == pytest.approx(0.8, abs=0.03)
    assert counts[rm.SPLIT_VAL] / 3000 == pytest.approx(0.1, abs=0.02)
    assert counts[rm.SPLIT_TEST] / 3000 == pytest.approx(0.1, abs=0.02)


def test_held_out_test_prompts_are_disjoint_from_train():
    corpus = _corpus()
    assert not corpus.prompts(rm.SPLIT_TRAIN) & corpus.prompts(rm.SPLIT_TEST)
    assert not corpus.prompts(rm.SPLIT_VAL) & corpus.prompts(rm.SPLIT_TEST)
    assert corpus.prompts(rm.SPLIT_TEST)


def test_leaked_cluster_is_fatal_not_silent():
    corpus = _corpus(40)
    rows = list(corpus.rows)
    leaked = rows[0]
    other_split = next(r.split for r in rows if r.split != leaked.split)
    rows[0] = rm.OrientedRow(**{**leaked.__dict__, "split": other_split})
    with pytest.raises(rm.CorpusError, match="split leaked"):
        rm.validate_corpus(rm.Corpus(rows=tuple(rows), mix={}, dedup=corpus.dedup))


# ============================================ orientation and swap augmentation


def test_orientation_is_randomized_and_the_label_follows_the_chosen_side():
    import random

    pairs = _character_pairs(200)
    rng = random.Random(0)
    rows = [orient for orient in (rm.orient_pair(p, "train", "c", rng) for p in pairs)]
    labels = [r.label for r in rows]
    assert set(labels) == {rm.LABEL_A, rm.LABEL_B}
    assert 0.35 < labels.count(rm.LABEL_A) / len(labels) < 0.65
    for row, pair in zip(rows, pairs, strict=True):
        assert row.chosen == pair.chosen
        assert row.rejected == pair.rejected


def test_every_split_including_validation_carries_both_ordering_directions():
    corpus = _corpus()
    for name in rm.SPLITS:
        rows = corpus.split(name)
        assert rows, f"{name} split is empty"
        assert {r.orientation for r in rows} == {
            rm.ORIENTATION_DIRECT,
            rm.ORIENTATION_SWAPPED,
        }
        # Exactly two presentations per underlying pair.
        assert len(rows) == 2 * len({r.pair_id for r in rows})


def test_a_one_directional_split_is_rejected():
    corpus = _corpus(40)
    direct_only = tuple(r for r in corpus.rows if r.orientation == rm.ORIENTATION_DIRECT)
    with pytest.raises(rm.CorpusError, match="both ordering directions"):
        rm.validate_corpus(rm.Corpus(rows=direct_only, mix={}, dedup=corpus.dedup))


def test_swapping_preserves_the_underlying_preference():
    row = _row(rm.LABEL_A, "winner", "loser")
    swapped = row.swapped()
    assert swapped.response_a == "loser"
    assert swapped.label == rm.LABEL_B
    assert swapped.chosen == row.chosen == "winner"


# ================================================== the stamped mix and audit


def test_the_mix_and_sampling_weights_are_stamped():
    corpus = _corpus()
    mix = corpus.mix
    assert mix["corpus_protocol"] == rm.CORPUS_PROTOCOL_VERSION
    assert mix["swap_augmented"] is True
    assert mix["split_salt"] == rm.SPLIT_SALT
    weights = {k: v["sampling_weight"] for k, v in mix["sources"].items()}
    assert set(weights) == {rm.SOURCE_CHARACTER_BANKED, rm.SOURCE_HELPFULNESS}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert mix["sources"][rm.SOURCE_HELPFULNESS]["fixture"] is True


def test_the_corpus_banks_the_audit_it_was_built_from():
    corpus = _corpus()
    assert corpus.dedup.effective_prompts > 0
    assert corpus.to_dict()["dedup"]["near_dup_jaccard"] == rm.NEAR_DUP_JACCARD


def test_corpus_roundtrips_through_jsonl_and_is_cookbook_trainable(tmp_path):
    corpus = _corpus(40)
    written = rm.write_corpus(corpus, tmp_path)
    reread = rm.read_corpus_split(written[rm.SPLIT_VAL])
    assert reread == corpus.split(rm.SPLIT_VAL)
    raw = json.loads(written[rm.SPLIT_VAL].read_text().splitlines()[0])
    # The schema ComparisonBuilderFromJsonl reads.
    assert set(raw["comparison"]) == {"prompt_conversation", "completion_A", "completion_B"}
    assert raw["label"] in (rm.LABEL_A, rm.LABEL_B)


# ================================== helpfulness: materialized, revision-pinned


def test_materialized_fixture_is_stamped_as_a_fixture(tmp_path):
    path = rm.materialize_helpfulness(tmp_path / "help.jsonl", n=8, execute=False)
    meta = json.loads(rm._helpfulness_meta_path(path).read_text())
    assert meta["fixture"] is True
    assert meta["source_id"] == rm.FIXTURE_SOURCE_ID
    assert meta["protocol"] == rm.HELPFULNESS_PROTOCOL_VERSION
    pairs, loaded_meta = rm.load_helpfulness(path)
    assert len(pairs) == len(rm.HELPFULNESS_FIXTURE)
    assert loaded_meta["content_hash"] == meta["content_hash"]


def test_every_external_source_pins_a_revision():
    for source in rm.HELPFULNESS_SOURCES.values():
        assert len(source.revision) == 40, f"{source.source_id} has no commit pin"
        assert source.dataset


def test_a_tampered_corpus_is_refused_not_silently_used(tmp_path):
    path = rm.materialize_helpfulness(tmp_path / "help.jsonl", n=8, execute=False)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["chosen"] = "something else entirely"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(rm.HelpfulnessCorpusError, match="content hash"):
        rm.load_helpfulness(path)


def test_a_missing_corpus_raises_instead_of_falling_back_to_a_remote_builder(tmp_path):
    with pytest.raises(rm.HelpfulnessCorpusError, match="never falls back"):
        rm.load_helpfulness(tmp_path / "absent.jsonl")


def test_an_unstamped_corpus_is_refused(tmp_path):
    path = tmp_path / "help.jsonl"
    path.write_text(json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"}) + "\n")
    with pytest.raises(rm.HelpfulnessCorpusError, match="source revision is"):
        rm.load_helpfulness(path)


def test_a_protocol_bump_refuses_to_mix_corpora(tmp_path):
    path = rm.materialize_helpfulness(tmp_path / "help.jsonl", n=8, execute=False)
    meta_path = rm._helpfulness_meta_path(path)
    meta = json.loads(meta_path.read_text())
    meta["protocol"] = "helpfulness-materialize-v0"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(rm.HelpfulnessCorpusError, match="Re-materialize"):
        rm.load_helpfulness(path)


_HH_TRANSCRIPT = (
    "\n\nHuman: What are some good indoor plants for a dark flat?"
    "\n\nAssistant: Snake plant and ZZ plant both tolerate low light."
    "\n\nHuman: Which one is harder to kill?"
    "\n\nAssistant: The ZZ plant -- it stores water in its rhizomes."
)


def test_hh_transcripts_parse_to_the_final_exchange():
    prompt, response = rm._parse_hh_side(_HH_TRANSCRIPT)
    assert prompt == "Which one is harder to kill?"
    assert response.startswith("The ZZ plant")


@pytest.mark.parametrize(
    "text",
    [
        "no turns at all",
        "\n\nHuman: hi\n\nAssistant: ",  # empty final assistant turn
        "\n\nHuman: hi",  # never reaches an assistant turn
        "",
    ],
)
def test_malformed_hh_transcripts_are_dropped_not_guessed_at(text):
    assert rm._parse_hh_side(text) is None


def test_a_real_materialized_helpfulness_half_makes_the_corpus_real(tmp_path):
    """execution_mode is `real` only when neither half is a stub."""
    path = tmp_path / "help.jsonl"
    rows = [
        {"prompt": f"real question {i}", "chosen": f"good answer {i}", "rejected": "no."}
        for i in range(10)
    ]
    rm._write_helpfulness(path, rows, source_id="hh-rlhf-helpful-base",
                          revision="0" * 40, fixture=False)
    banked = tmp_path / "dpo_pairs.jsonl"
    banked.write_text(
        "\n".join(
            json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected})
            for p in _character_pairs(20)
        )
        + "\n"
    )
    corpus, _ = rm.build_stage([banked], path, rejudge=False)
    assert corpus.mix["execution_mode"] == rm.EXECUTION_REAL


def test_the_fixture_cannot_masquerade_as_the_real_corpus():
    with pytest.raises(rm.CorpusError, match=rm.FIXTURE_SOURCE_ID):
        rm.build_corpus(_character_pairs(20), rm._fixture_pairs())


# ================================================ labeling: both orders + gate


@pytest.mark.parametrize(
    ("ab", "ba", "label", "reason"),
    [
        ("A", "B", "A", rm.REASON_AGREE),  # both name the first response
        ("B", "A", "B", rm.REASON_AGREE),  # both name the second response
        ("A", "A", None, rm.REASON_DISAGREE),  # tracks position, not content
        ("B", "B", None, rm.REASON_DISAGREE),
        ("TIE", "TIE", None, rm.REASON_BOTH_TIE),
        ("A", "TIE", None, rm.REASON_DISAGREE),
        (None, "A", None, rm.REASON_UNPARSEABLE),
        ("A", None, None, rm.REASON_UNPARSEABLE),
    ],
)
def test_both_orders_resolution_table(ab, ba, label, reason):
    assert rm.resolve_both_orders(ab, ba) == (label, reason)


def test_a_swap_inconsistent_judge_yields_no_labels():
    pairs = _character_pairs(10)
    kept, outcomes = rm.label_pairs(pairs, swap_consistent=False)
    assert kept == []
    assert {o.reason for o in outcomes} == {rm.REASON_DISAGREE}


def test_a_swap_consistent_judge_keeps_pairs_and_restamps_the_label_source():
    pairs = _character_pairs(10)
    kept, outcomes = rm.label_pairs(pairs, swap_consistent=True)
    assert len(kept) == len(pairs)
    assert {o.reason for o in outcomes} == {rm.REASON_AGREE}
    assert {p.label_source for p in kept} == {rm.LABEL_SOURCE_PHASE3_JUDGE}


def test_the_paid_judge_is_called_in_both_orders_for_every_pair():
    calls: list[tuple[str, str]] = []

    def judge(prompt: str, first: str, second: str) -> str:
        calls.append((first[:12], second[:12]))
        return "A" if "Ahoy" in first else "B"

    pairs = _character_pairs(3)
    kept, _ = rm.label_pairs(pairs, execute=True, judge_fn=judge)
    assert len(calls) == 6  # two presentations per pair
    assert len(kept) == 3
    # Each pair was shown in both directions.
    assert calls[0] == (calls[1][1], calls[1][0])


def test_paid_labeling_without_a_judge_or_runtime_refuses_rather_than_guessing():
    with pytest.raises(rm.CharacterJudgeUnavailable, match="explicit judge_fn"):
        rm.label_pairs(_character_pairs(2), execute=True)


def test_the_judge_adapter_renders_through_the_registered_character_instrument():
    """No second copy of the judge prompt: the text comes from octt.preference."""
    from octt import models, preference, tinker_client

    runtime = tinker_client.create_runtime(
        (models.TEACHER_MODEL,), config=tinker_client.TinkerClientConfig(dry_run=True)
    )
    judge = rm.character_judge_fn(runtime)
    assert callable(judge)
    # The adapter must not carry its own prompt text.
    source = (OCTT / "reward_model.py").read_text()
    assert preference.JUDGE_USER_TEMPLATE not in source
    assert preference.JUDGE_SYSTEM_TEMPLATE not in source
    # ... and the sibling judge is a registered, versioned instrument.
    assert instruments.get(preference.INSTRUMENT_ID).kind == instruments.KIND_JUDGE


def test_group_expansion_is_the_complete_g4_tournament():
    pairs = rm.group_to_pairs("q", ["r0", "r1", "r2", "r3"], source="s", origin="o")
    assert len(pairs) == 6  # 6 unordered pairs -> 12 ordered matchups
    assert len({(p.chosen, p.rejected) for p in pairs}) == 6


@pytest.mark.parametrize("size", [2, 3, 5, 8])
def test_any_group_size_other_than_four_is_rejected(size):
    with pytest.raises(ValueError, match="G=4|group_size must be"):
        rm.group_to_pairs("q", [f"r{i}" for i in range(size)], source="s", origin="o")


def test_on_policy_sampling_refuses_a_non_four_group_size():
    with pytest.raises(ValueError, match="group_size must be 4"):
        rm.sample_on_policy_groups(["q"], object(), "m", group_size=8)


def test_on_policy_sampling_is_free_and_returns_four_per_prompt():
    from octt import models, tinker_client

    runtime = tinker_client.create_runtime(
        (models.SCALING_SET[0],), config=tinker_client.TinkerClientConfig(dry_run=True)
    )
    groups = rm.sample_on_policy_groups(["q1", "q2"], runtime, models.SCALING_SET[0])
    assert set(groups) == {"q1", "q2"}
    assert all(len(v) == 4 for v in groups.values())


# ============================================= execution mode / spend guards


def test_a_dry_run_build_is_stamped_as_a_stub(tmp_path):
    path = rm.materialize_helpfulness(tmp_path / "help.jsonl", n=8, execute=False)
    banked = tmp_path / "dpo_pairs.jsonl"
    banked.write_text(
        "\n".join(
            json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected})
            for p in _character_pairs(20)
        )
        + "\n"
    )
    corpus, provenance = rm.build_stage([banked], path, allow_fixture=True)
    assert corpus.mix["execution_mode"] == rm.EXECUTION_STUB
    assert provenance["labeling"]["rejudged"] is True
    assert provenance["helpfulness"]["fixture"] is True


def test_training_refuses_to_spend_on_a_stub_corpus(tmp_path):
    class _Runtime:
        class config:  # mimics the TinkerRuntime attribute shape
            dry_run = False

    rm.write_corpus(_corpus(40), tmp_path)
    with pytest.raises(rm.CorpusError, match="execution_mode"):
        rm.train(tmp_path, "Qwen/Qwen3.5-4B", tmp_path / "out", _Runtime(), execute=True)


def test_dry_run_training_never_touches_the_paid_runtime(tmp_path):
    from octt import models, tinker_client

    runtime = tinker_client.create_runtime(
        (models.SCALING_SET[0],), config=tinker_client.TinkerClientConfig(dry_run=True)
    )
    rm.write_corpus(_corpus(40), tmp_path)
    checkpoint = rm.train(tmp_path, models.SCALING_SET[0], tmp_path / "out", runtime)
    assert "dry-run" in (checkpoint.sampler_path or "")
    meta = json.loads((tmp_path / "out" / "reward_model.meta.json").read_text())
    assert meta["dry_run"] is True
    assert meta["mix"]["corpus_protocol"] == rm.CORPUS_PROTOCOL_VERSION


# =============================================== the frozen control set


def test_padding_adds_length_and_nothing_else():
    for control in rm.PADDING_CONTROLS:
        assert len(control.variant) > len(control.base)
        assert rm._informative_words(control.variant) == rm._informative_words(control.base)


def test_marker_controls_start_in_character_so_only_the_COUNT_changes():
    for control in rm.MARKER_CONTROLS:
        assert rm.marker_count(control.base) >= 1
        assert rm.marker_count(control.variant) > rm.marker_count(control.base)
        assert rm._informative_words(control.variant) == rm._informative_words(control.base)


def test_the_padding_lead_carries_no_information():
    assert rm._informative_words(rm.PADDING_LEAD) == set()


def test_control_set_hash_is_pinned():
    # Never "fix" this to make an edit pass: mint a new CONTROL_SET_VERSION.
    assert rm.control_set_hash() == "07a9467ef02e2d5c"


def test_the_control_set_is_a_registered_versioned_instrument():
    entry = instruments.get(rm.INSTRUMENT_ID)
    assert entry.parser == rm.CONTROL_SET_VERSION  # drift guard
    assert entry.kind == instruments.KIND_JUDGE


def test_module_does_not_import_analysis_curation():
    import ast

    tree = ast.parse((OCTT / "reward_model.py").read_text())
    imported = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert "trait_profiles" not in imported


# ==================================================== metric correctness


def test_accuracy_counts_a_zero_margin_as_half():
    rows = [_row(rm.LABEL_A, "x", "y"), _row(rm.LABEL_B, "x", "y")]
    result = rm.pairwise_accuracy(_ConstantModel(), rows)
    assert result["ties"] == 2
    assert result["accuracy"] == pytest.approx(0.5)


def test_accuracy_and_auc_on_a_hand_checked_case():
    rows = [
        _row(rm.LABEL_A, "GOOD one", "bad one", pair_id="1"),
        _row(rm.LABEL_A, "GOOD two", "bad two", pair_id="2"),
        _row(rm.LABEL_B, "bad three", "GOOD three", pair_id="3"),
        _row(rm.LABEL_A, "bad four", "GOOD four", pair_id="4"),  # model is wrong here
    ]
    model = _KeywordModel()
    assert rm.pairwise_accuracy(model, rows)["accuracy"] == pytest.approx(0.75)
    # Three A-labelled rows with margins +1, +1, -1; one B-labelled with -1.
    assert rm.pairwise_auc(model, rows)["auc"] == pytest.approx(0.8333, abs=1e-3)


def test_calibration_is_perfect_for_a_perfectly_calibrated_predictor():
    rows = [_row(rm.LABEL_A, "x", "y"), _row(rm.LABEL_B, "x", "y")]
    result = rm.calibration(_ConstantModel(), rows)
    # Every margin is zero -> p = 0.5, and half the labels are A.
    assert result["ece"] == pytest.approx(0.0)
    assert result["brier"] == pytest.approx(0.25)


def test_temperature_scaling_improves_calibration_on_a_held_out_split():
    corpus = _corpus()
    train, val = corpus.split(rm.SPLIT_TRAIN), corpus.split(rm.SPLIT_VAL)
    raw = rm.well_behaved_model()
    raw_ece = rm.calibration(raw, val)["ece"]
    fitted = rm.calibrate(raw, train)
    assert rm.calibration(fitted, val)["ece"] <= raw_ece
    # Fitting must not change the RANKING, only its scale.
    assert rm.pairwise_accuracy(fitted, val) == rm.pairwise_accuracy(raw, val)


def test_spearman_matches_a_hand_computed_value():
    assert rm.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert rm.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert rm.spearman([1, 1, 1], [1, 2, 3]) != rm.spearman([1, 1, 1], [1, 2, 3])  # NaN


# ==================================================== the gates have teeth


def test_a_metric_that_cannot_be_computed_reads_FAIL_not_PASS():
    result = rm._gate("x", float("nan"), 0.5, ">=", {})
    assert result.passed is False
    result = rm._gate("x", None, 0.5, "<=", {})
    assert result.passed is False


def test_a_well_behaved_model_passes_every_gate():
    corpus = _corpus()
    model = rm.calibrate(rm.well_behaved_model(), corpus.split(rm.SPLIT_TRAIN))
    report = rm.evaluate_gates(model, corpus.split(rm.SPLIT_VAL), dedup=corpus.dedup)
    assert report.passed, report.summary()


def test_a_length_collapsed_model_fails_the_padding_counterfactual():
    corpus = _corpus()
    report = rm.evaluate_gates(rm.length_collapsed_model(), corpus.split(rm.SPLIT_VAL))
    assert "padding_earns_no_reward" in report.failed
    assert "padding_earns_no_reward_heldout" in report.failed
    assert not report.passed


def test_a_marker_collapsed_model_fails_the_marker_counterfactual():
    corpus = _corpus()
    report = rm.evaluate_gates(rm.marker_collapsed_model(), corpus.split(rm.SPLIT_VAL))
    assert "marker_stuffing_earns_no_reward" in report.failed
    assert "marker_stuffing_earns_no_reward_heldout" in report.failed
    assert not report.passed


def test_a_marker_collapsed_model_can_still_ace_held_out_accuracy():
    """Why the counterfactual gates exist: accuracy alone cannot catch this."""
    corpus = _corpus()
    rows = corpus.split(rm.SPLIT_VAL)
    assert rm.pairwise_accuracy(rm.marker_collapsed_model(), rows)["accuracy"] == 1.0
    report = rm.evaluate_gates(rm.marker_collapsed_model(), rows)
    assert "held_out_accuracy" not in report.failed
    assert not report.passed  # ... and it is still rejected


def test_a_position_biased_model_fails_order_swap_consistency():
    corpus = _corpus()
    model = rm.PositionBiasedRewardModel(base=rm.well_behaved_model())
    report = rm.evaluate_gates(model, corpus.split(rm.SPLIT_VAL))
    assert "order_swap_consistency" in report.failed


def test_pointwise_models_declare_structural_swap_consistency():
    corpus = _corpus(40)
    swap = rm.order_swap_consistency(rm.well_behaved_model(), corpus.split(rm.SPLIT_VAL))
    assert swap["consistency"] == pytest.approx(1.0)
    assert swap["structural"] is True


def test_identical_but_longer_never_earns_reward_for_a_sane_model():
    model = rm.well_behaved_model()
    for control in rm.PADDING_CONTROLS:
        assert model.score(control.prompt, control.variant) <= model.score(
            control.prompt, control.base
        )


def test_identical_but_marker_stuffed_never_earns_reward_for_a_sane_model():
    model = rm.well_behaved_model()
    for control in rm.MARKER_CONTROLS:
        assert model.score(control.prompt, control.variant) <= model.score(
            control.prompt, control.base
        )


def test_a_length_collapsed_model_does_prefer_the_padded_variant():
    """The counterfactual has teeth: the degenerate model really does fail it."""
    model = rm.length_collapsed_model()
    control = rm.PADDING_CONTROLS[0]
    assert model.score(control.prompt, control.variant) > model.score(
        control.prompt, control.base
    )


def test_a_marker_collapsed_model_does_prefer_the_stuffed_variant():
    model = rm.marker_collapsed_model()
    control = rm.MARKER_CONTROLS[0]
    assert model.score(control.prompt, control.variant) > model.score(
        control.prompt, control.base
    )


def test_helpfulness_and_format_controls_reject_a_marker_counter():
    model = rm.marker_collapsed_model()
    assert rm.directional_results(model, rm.HELPFULNESS_CONTROLS)["pass_rate"] < 1.0
    assert rm.directional_results(model, rm.FORMAT_CONTROLS)["pass_rate"] < 1.0


def test_held_out_marker_counterfactual_skips_responses_not_in_character():
    corpus = _corpus(40)
    result = rm.held_out_counterfactuals(
        rm.well_behaved_model(), corpus.split(rm.SPLIT_VAL), kind=rm.KIND_MARKER
    )
    # Half the synthetic responses carry no marker; skipping them is reported.
    assert result["skipped_not_in_character"] > 0


def test_the_gate_report_stamps_its_instrument_and_split():
    corpus = _corpus(40)
    report = rm.evaluate_gates(
        rm.well_behaved_model(), corpus.split(rm.SPLIT_VAL), dedup=corpus.dedup
    )
    payload = report.to_dict()
    assert payload["gate_set_version"] == rm.GATE_SET_VERSION
    assert payload["control_set_hash"] == rm.control_set_hash()
    assert payload["marker_instrument"] == rm.MARKER_INSTRUMENT
    assert payload["split"] == rm.SPLIT_VAL


def test_degenerate_baselines_are_reported_as_diagnostics_not_gates():
    corpus = _corpus(40)
    report = rm.evaluate_gates(
        rm.well_behaved_model(), corpus.split(rm.SPLIT_VAL), dedup=corpus.dedup
    )
    assert report.diagnostics["confounded_by_markers"] is True
    assert "confounded_by_markers" not in {r.name for r in report.results}
    assert "marker count alone" in report.summary()


# ==================================================== pilot classification


def test_pilot_is_the_default_and_no_threshold_is_invented():
    corpus = _corpus()
    status = rm.pilot_status(corpus.dedup)
    assert status["status"] == rm.PILOT_STATUS_PILOT
    assert status["diversity_reference"] is None
    assert "no universal minimum sample count" in status["rationale"]


def test_a_supplied_reference_can_lift_or_keep_the_pilot_label():
    corpus = _corpus()
    effective = corpus.dedup.effective_prompts
    assert (
        rm.pilot_status(corpus.dedup, diversity_reference=effective + 1)["status"]
        == rm.PILOT_STATUS_PILOT
    )
    lifted = rm.pilot_status(
        corpus.dedup, diversity_reference=effective, justification="documented call"
    )
    assert lifted["status"] == rm.PILOT_STATUS_ESTABLISHED
    assert lifted["reference_justification"] == "documented call"


# ================================================================== the CLI


def test_cli_audit_is_free_and_reports_diversity(tmp_path, capsys):
    banked = tmp_path / "dpo_pairs.jsonl"
    banked.write_text(
        "\n".join(
            json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected})
            for p in _character_pairs(120)
        )
        + "\n"
    )
    assert rm.main(["audit", "--banked", str(banked)]) == 0
    out = capsys.readouterr().out
    assert "effective prompts" in out
    assert rm.PILOT_STATUS_PILOT in out


def test_cli_build_refuses_the_fixture_unless_asked(tmp_path, capsys):
    help_path = rm.materialize_helpfulness(tmp_path / "help.jsonl", n=8, execute=False)
    banked = tmp_path / "dpo_pairs.jsonl"
    banked.write_text(
        "\n".join(
            json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected})
            for p in _character_pairs(40)
        )
        + "\n"
    )
    code = rm.main(
        [
            "build",
            "--banked", str(banked),
            "--helpfulness", str(help_path),
            "--out", str(tmp_path / "corpus"),
        ]
    )
    assert code == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_gate_exits_nonzero_when_a_gate_fails(tmp_path, capsys):
    rm.write_corpus(_corpus(), tmp_path / "corpus")
    passing = rm.main(["gate", "--corpus", str(tmp_path / "corpus"), "--split", "val"])
    assert passing == 0
    failing = rm.main(
        [
            "gate",
            "--corpus", str(tmp_path / "corpus"),
            "--split", "val",
            "--reward-model", "length-collapsed",
        ]
    )
    assert failing == 2
    assert "FAIL" in capsys.readouterr().out


def test_cli_warns_before_scoring_the_reserved_test_split(tmp_path, capsys):
    rm.write_corpus(_corpus(), tmp_path / "corpus")
    rm.main(["gate", "--corpus", str(tmp_path / "corpus"), "--split", "test"])
    assert "RESERVED test split" in capsys.readouterr().out


def test_cli_materialize_labels_the_fixture_loudly(tmp_path, capsys):
    assert rm.main(["materialize", "--out", str(tmp_path / "h.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "FIXTURE" in out
    assert "--execute" in out


def test_octt_cli_exposes_the_subcommand(tmp_path, capsys):
    from octt import cli

    banked = tmp_path / "dpo_pairs.jsonl"
    banked.write_text(
        "\n".join(
            json.dumps({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected})
            for p in _character_pairs(120)
        )
        + "\n"
    )
    assert cli.main(["reward-model", "audit", "--banked", str(banked)]) == 0
    assert "effective prompts" in capsys.readouterr().out
