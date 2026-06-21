"""Tests for prompt/data sources and trait descriptors (offline)."""

from __future__ import annotations

from octt import constitution, data_sources


def test_trait_descriptors_unique_and_sized():
    traits = data_sources.trait_descriptors(150)
    assert len(traits) == 150
    assert len(set(traits)) == 150


def test_trait_descriptors_cycles_beyond_base():
    n = len(data_sources.TRAIT_DESCRIPTORS) + 5
    traits = data_sources.trait_descriptors(n)
    assert len(traits) == n
    assert len(set(traits)) == n  # suffixing keeps them distinct


def test_offline_loaders_fall_back_to_fixtures():
    for loader in (
        data_sources.load_lima_prompts,
        data_sources.load_wildchat_prompts,
        data_sources.load_pure_dove_prompts,
    ):
        prompts = loader(5, offline=True)
        assert len(prompts) == 5
        assert all(isinstance(p, str) and p for p in prompts)


def test_offline_loader_cycles_when_n_exceeds_fixture():
    prompts = data_sources.load_lima_prompts(100, offline=True)
    assert len(prompts) == 100


def test_constitution_relevant_prompts_reference_persona():
    c = constitution.load("humorous")
    prompts = data_sources.constitution_relevant_prompts(c, 6)
    assert len(prompts) == 6
    assert any("humorous" in p for p in prompts)


def test_dpo_prompts_mixes_sources_deterministically():
    c = constitution.load("sarcastic")
    a = data_sources.dpo_prompts(c, 10, offline=True)
    b = data_sources.dpo_prompts(c, 10, offline=True)
    assert a == b  # deterministic
    assert len(a) == 10


def test_self_reflection_prompts_present():
    assert len(data_sources.SELF_REFLECTION_PROMPTS) == 10
