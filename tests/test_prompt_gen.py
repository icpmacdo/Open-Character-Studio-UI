"""Offline tests for App F prompt generation (octt.prompt_gen) and its data_sources hook."""

from __future__ import annotations

import json

import pytest

from octt import constitution, data_sources, models, prompt_gen, tinker_client
from octt.constitution import Constitution


@pytest.fixture(autouse=True)
def isolated_prompts_dir(tmp_path, monkeypatch):
    """Point the canonical prompts dir at tmp so tests never touch repo data/."""
    d = tmp_path / "data" / "constitution_prompts"
    monkeypatch.setattr(prompt_gen, "PROMPTS_DIR", d)
    return d


def _dry_runtime():
    return tinker_client.create_runtime(
        (models.TEACHER_MODEL,),
        config=tinker_client.TinkerClientConfig(dry_run=True),
    )


def _load(path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Numbered-list parsing
# ---------------------------------------------------------------------------


def test_parse_numbered_list_messy_output():
    text = (
        "Here are the prompts you asked for:\n"
        "\n"
        "1. What's a good gift for a coworker who is leaving?\n"
        '2) "How do I fix a leaky faucet?"\n'
        "  3: Plan a birthday dinner for six people.\n"
        "- Tell me about a good weekend hike.\n"
        "4. What's a good gift for a coworker who is leaving?\n"
        "That's all!\n"
    )
    assert prompt_gen.parse_numbered_list(text) == [
        "What's a good gift for a coworker who is leaving?",
        "How do I fix a leaky faucet?",
        "Plan a birthday dinner for six people.",
        "Tell me about a good weekend hike.",
    ]


def test_parse_plain_lines_when_no_markers():
    # The dry-run stub is a single unnumbered line; it must parse as one item.
    stub = "[prompt-gen|Qwen3.5-397B-A17B|abc123] The assertion (for your reference"
    assert prompt_gen.parse_numbered_list(stub) == [stub]
    assert prompt_gen.parse_numbered_list("\n\nfoo\nbar\n\n") == ["foo", "bar"]
    assert prompt_gen.parse_numbered_list("") == []


# ---------------------------------------------------------------------------
# Generation: counts, retry, top-up, determinism
# ---------------------------------------------------------------------------


def test_generate_default_per_assertion_counts(tmp_path):
    c = constitution.load("humorous")
    out = tmp_path / "prompts.json"
    result = prompt_gen.generate_constitution_prompts(c, _dry_runtime(), out_path=out)
    assert result == out
    data = _load(out)
    assert data["persona"] == "humorous"
    assert data["per_assertion"] == 50
    assert len(data["prompts"]) == 50 * len(c.assertions)
    assert all(isinstance(p, str) and p for p in data["prompts"])
    assert {"content_hash", "assertions_hash", "generator_model"} <= set(data)
    # Each per-assertion batch opens with that assertion's 5 template seeds.
    seed0 = data_sources.constitution_relevant_prompts(
        Constitution(persona=c.persona, assertions=(c.assertions[0],)), 5
    )
    assert data["prompts"][:5] == seed0


def test_stub_generation_counts_are_deterministic(tmp_path):
    # Offline stubs parse to 1 prompt per call: 5 seeds + 2 generated
    # (first call + retry) + 1 template top-up = 8 per assertion.
    c = constitution.load("sarcastic")
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=8, out_path=out, offline=True
    )
    data = _load(out)
    k = len(c.assertions)
    assert len(data["prompts"]) == 8 * k
    assert data["counts"] == {"seed": 5 * k, "generated": 2 * k, "template_topup": 1 * k, "screened_out": 0}
    # No duplicates within a per-assertion batch.
    for i in range(k):
        batch = data["prompts"][8 * i: 8 * (i + 1)]
        assert len(set(batch)) == 8


def test_per_assertion_smaller_than_seeds_skips_generation(tmp_path, monkeypatch):
    c = constitution.load("pirate")

    def boom(*args, **kwargs):
        raise AssertionError("no generation call expected when seeds cover the count")

    monkeypatch.setattr(prompt_gen.generation, "complete_many", boom)
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=3, out_path=out, offline=True
    )
    data = _load(out)
    assert len(data["prompts"]) == 3 * len(c.assertions)
    assert data["counts"]["generated"] == 0 and data["counts"]["template_topup"] == 0


def test_over_long_lists_truncate_to_exact_count(tmp_path, monkeypatch):
    c = constitution.load("sarcastic")
    numbered = "\n".join(f"{i}. Generated prompt number {i}?" for i in range(1, 30))
    monkeypatch.setattr(
        prompt_gen.generation,
        "complete_many",
        lambda sampler, convos: [numbered] * len(convos),
    )
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=10, out_path=out, offline=True
    )
    data = _load(out)
    k = len(c.assertions)
    assert len(data["prompts"]) == 10 * k
    assert data["counts"] == {"seed": 5 * k, "generated": 5 * k, "template_topup": 0, "screened_out": 0}
    # First batch: 5 seeds then the first 5 parsed prompts, in order.
    assert data["prompts"][5] == "Generated prompt number 1?"
    assert data["prompts"][9] == "Generated prompt number 5?"


def test_short_parse_retries_exactly_once_then_tops_up(tmp_path, monkeypatch):
    c = constitution.load("loving")
    calls: list[int] = []

    def fake_complete_many(sampler, convos):
        calls.append(len(convos))
        if len(calls) == 1:
            return ["1. Lone prompt about the weather?"] * len(convos)
        return ["\n".join(f"{i}. Retry prompt {i}!" for i in range(1, 3))] * len(convos)

    monkeypatch.setattr(prompt_gen.generation, "complete_many", fake_complete_many)
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=10, out_path=out, offline=True
    )
    k = len(c.assertions)
    # One batch call for all assertions, then one retry batch — never a third.
    assert calls == [k, k]
    data = _load(out)
    assert len(data["prompts"]) == 10 * k
    # 5 seeds + 3 generated (1 + 2 retry) + 2 template top-ups per assertion.
    assert data["counts"] == {"seed": 5 * k, "generated": 3 * k, "template_topup": 2 * k, "screened_out": 0}


def test_topup_determinism(tmp_path):
    c = constitution.load("poetic")
    a = prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=12, out_path=tmp_path / "a.json", offline=True
    )
    b = prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=12, out_path=tmp_path / "b.json", offline=True
    )
    da, db = _load(a), _load(b)
    assert da["prompts"] == db["prompts"]
    assert da["counts"]["template_topup"] > 0
    # Variation-suffixed top-ups keep every batch free of duplicates.
    batch = da["prompts"][:12]
    assert len(set(batch)) == 12


def test_real_path_persists_generated_only_and_screens_violations(monkeypatch):
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    assertion_quote = " ".join(c.assertions[0].split()[:6])

    def fake_complete_many(_sampler, convos):
        return [
            "1. What's a good gift for a coworker who is leaving?\n"
            "2. How do I ask you to be pirate for a day?\n"
            f'3. Someone said "{assertion_quote}" - thoughts?\n'
            "4. Plan a birthday dinner for six people.\n"
        ] * len(convos)

    monkeypatch.setattr(prompt_gen, "_generator_sampler", lambda *a: object())
    monkeypatch.setattr(prompt_gen.generation, "complete_many", fake_complete_many)
    runtime = tinker_client.TinkerRuntime(
        config=tinker_client.TinkerClientConfig(dry_run=False),
        service_client=None,
        renderer_bindings={},
        renderer_plans={},
    )
    prompt_gen.generate_constitution_prompts(c, runtime, per_assertion=4, out_path=out)
    data = _load(out)
    # Persona-naming and assertion-quoting candidates are screened; template
    # seeds are few-shot input only; cross-assertion duplicates collapse.
    assert data["prompts"] == [
        "What's a good gift for a coworker who is leaving?",
        "Plan a birthday dinner for six people.",
    ]
    assert data["counts"]["seed"] == 0
    assert data["counts"]["template_topup"] == 0
    assert data["counts"]["screened_out"] > 0
    assert data["execution_mode"] == "real"
    assert not any("pirate" in p.lower() for p in data["prompts"])


def test_offline_forces_stubs_on_a_real_runtime(tmp_path):
    # A non-dry-run runtime with no service client would fail on any real
    # sampling; offline=True must route through the generation stubs instead.
    runtime = tinker_client.TinkerRuntime(
        config=tinker_client.TinkerClientConfig(dry_run=False),
        service_client=None,
        renderer_bindings={},
        renderer_plans={},
    )
    c = constitution.load("humorous")
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, runtime, per_assertion=7, out_path=out, offline=True
    )
    assert len(_load(out)["prompts"]) == 7 * len(c.assertions)


def test_rejects_nonpositive_per_assertion(tmp_path):
    c = constitution.load("humorous")
    with pytest.raises(ValueError):
        prompt_gen.generate_constitution_prompts(
            c, _dry_runtime(), per_assertion=0, out_path=tmp_path / "p.json", offline=True
        )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_hit_skips_regeneration(tmp_path, monkeypatch):
    c = constitution.load("humorous")
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=8, out_path=out, offline=True
    )
    mtime = out.stat().st_mtime_ns

    def boom(*args, **kwargs):
        raise AssertionError("regenerated despite a cache hit")

    monkeypatch.setattr(prompt_gen.generation, "complete_many", boom)
    result = prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=8, out_path=out, offline=True
    )
    assert result == out
    assert out.stat().st_mtime_ns == mtime


def test_cache_invalidated_when_inputs_change(tmp_path):
    c = constitution.load("humorous")
    out = tmp_path / "p.json"
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=8, out_path=out, offline=True
    )
    assert len(_load(out)["prompts"]) == 8 * len(c.assertions)
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )
    data = _load(out)
    assert data["per_assertion"] == 6
    assert len(data["prompts"]) == 6 * len(c.assertions)


# ---------------------------------------------------------------------------
# data_sources hook
# ---------------------------------------------------------------------------


def test_default_prompts_path(isolated_prompts_dir):
    assert prompt_gen.default_prompts_path("humorous") == isolated_prompts_dir / "humorous.json"


def test_data_sources_uses_valid_generated_file():
    c = constitution.load("humorous")
    out = prompt_gen.default_prompts_path(c.persona)
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )
    _mark_real(out)
    file_prompts = _load(out)["prompts"]

    got = data_sources.constitution_relevant_prompts(c, 4)
    assert got == file_prompts[:4]
    # Cycles when n exceeds the file's prompt count.
    total = len(file_prompts)
    got_all = data_sources.constitution_relevant_prompts(c, total + 3)
    assert got_all[:total] == file_prompts
    assert got_all[total:] == file_prompts[:3]


def test_data_sources_falls_back_on_missing_file():
    c = constitution.load("pirate")
    prompts = data_sources.constitution_relevant_prompts(c, 5)
    assert len(prompts) == 5
    assert prompts[0].startswith("Someone just told you:")  # template synthesis


def test_data_sources_falls_back_on_malformed_file():
    c = constitution.load("pirate")
    path = prompt_gen.default_prompts_path(c.persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json at all")
    prompts = data_sources.constitution_relevant_prompts(c, 5)  # must not raise
    assert prompts[0].startswith("Someone just told you:")


def test_data_sources_falls_back_on_wrong_shapes():
    c = constitution.load("pirate")
    path = prompt_gen.default_prompts_path(c.persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = prompt_gen.assertions_hash(c)
    for payload in (
        ["a", "bare", "list"],  # not a dict
        {"assertions_hash": stamp, "prompts": "not-a-list"},
        {"assertions_hash": stamp, "prompts": []},
        {"assertions_hash": stamp, "prompts": [1, 2, 3]},
    ):
        path.write_text(json.dumps(payload))
        prompts = data_sources.constitution_relevant_prompts(c, 5)
        assert prompts[0].startswith("Someone just told you:")


def _mark_real(path):
    """Re-stamp a generated file as a real run's output (tests run offline)."""
    data = _load(path)
    data["execution_mode"] = "real"
    path.write_text(json.dumps(data))


def test_data_sources_falls_back_on_stale_assertions_hash():
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )
    _mark_real(out)
    # Same persona, edited constitution: the stamp no longer matches.
    edited = Constitution(persona=c.persona, assertions=c.assertions[:-1])
    prompts = data_sources.constitution_relevant_prompts(edited, 5)
    assert prompts[0].startswith("Someone just told you:")
    # The unedited constitution still gets the generated file.
    assert data_sources.constitution_relevant_prompts(c, 3) == _load(out)["prompts"][:3]


def test_stub_prompt_files_are_marked_and_ignored_by_reader(monkeypatch):
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )
    assert _load(out)["execution_mode"] == "stub"
    # The training-side reader must never trust a dry-run artifact.
    assert data_sources._generated_constitution_prompts(c) is None
    # An offline rerun with the same inputs still reuses the cached stub.
    monkeypatch.setattr(
        prompt_gen.generation, "complete_many",
        lambda *_a, **_k: pytest.fail("cache should have been reused"),
    )
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )


def test_real_run_regenerates_over_cached_stub(monkeypatch):
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )
    assert _load(out)["execution_mode"] == "stub"

    # Same inputs hash identically, but a real run must not adopt the stub.
    calls: list[int] = []

    def fake_complete_many(_sampler, convos):
        calls.append(len(convos))
        return ["1. a genuinely generated prompt"] * len(convos)

    monkeypatch.setattr(prompt_gen, "_generator_sampler", lambda *a: object())
    monkeypatch.setattr(prompt_gen.generation, "complete_many", fake_complete_many)
    runtime = tinker_client.TinkerRuntime(
        config=tinker_client.TinkerClientConfig(dry_run=False),
        service_client=None,
        renderer_bindings={},
        renderer_plans={},
    )
    prompt_gen.generate_constitution_prompts(c, runtime, per_assertion=6, out_path=out)
    assert calls, "real run should have re-sampled instead of reusing the stub"
    assert _load(out)["execution_mode"] == "real"
    # The reader now trusts the file.
    assert data_sources._generated_constitution_prompts(c) == _load(out)["prompts"]


# --- prompt provenance gate (F1/F2): real runs must not silently downgrade ----


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (None, "missing file"),
        (lambda d: d.update(protocol="constitution-prompts-appendix-f-v1-legacy"), "old protocol"),
        (lambda d: d.pop("protocol", None), "unstamped pre-v2 file"),
        (lambda d: d.update(execution_mode="stub"), "dry-run stub"),
        (lambda d: d.update(assertions_hash="deadbeef"), "constitution edited"),
    ],
)
def test_real_run_refuses_untrusted_prompt_file(mutate, reason):
    """A paid run raises rather than falling back to persona-naming templates."""
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    if mutate is not None:
        prompt_gen.generate_constitution_prompts(
            c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
        )
        data = _load(out)
        data["execution_mode"] = "real"
        mutate(data)
        out.write_text(json.dumps(data))

    with pytest.raises(data_sources.StaleConstitutionPromptsError) as exc:
        data_sources.dpo_prompts(c, 9, offline=False)
    assert "octt gen-prompts pirate --execute" in str(exc.value), reason


def test_offline_run_still_falls_back_to_templates():
    """The dry-run tier keeps working with no generated file and no model calls."""
    c = constitution.load("pirate")
    prompts = data_sources.dpo_prompts(c, 9, offline=True)
    assert len(prompts) == 9


def test_current_protocol_file_passes_the_gate():
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    prompt_gen.generate_constitution_prompts(
        c, _dry_runtime(), per_assertion=6, out_path=out, offline=True
    )
    _mark_real(out)
    assert _load(out)["protocol"] == prompt_gen._PROMPT_PROTOCOL_VERSION
    # Does not raise: stamps are current, so the real path trusts the file.
    assert data_sources.dpo_prompts(c, 9, offline=False)


def test_prompt_gen_seeds_never_read_the_generated_file():
    """Few-shot seeds come from templates, so gen-prompts can't feed on its own output."""
    c = constitution.load("pirate")
    out = prompt_gen.default_prompts_path(c.persona)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "assertions_hash": prompt_gen.assertions_hash(c),
        "protocol": prompt_gen._PROMPT_PROTOCOL_VERSION,
        "execution_mode": "real",
        "prompts": ["POISON: a previously generated prompt"],
    }))
    single = Constitution(persona=c.persona, assertions=(c.assertions[0],))
    seeds = data_sources.template_constitution_prompts(single, 5)
    assert all("POISON" not in s for s in seeds)
