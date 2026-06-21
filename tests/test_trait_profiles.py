"""Tests for per-persona trait profiles and the shift summary."""

from __future__ import annotations

from octt import constitution, data_sources, trait_profiles


def test_every_constitution_has_a_profile():
    pool = set(data_sources.TRAIT_DESCRIPTORS)
    for persona in constitution.available():
        prof = trait_profiles.profile(persona)
        assert prof is not None, f"no trait profile for persona {persona!r}"
        assert prof.aligned and prof.opposing
        assert set(prof.aligned) <= pool
        assert set(prof.opposing) <= pool
        assert not (set(prof.aligned) & set(prof.opposing))


def test_required_traits_are_aligned_plus_opposing():
    prof = trait_profiles.profile("pirate")
    req = trait_profiles.required_traits("pirate")
    assert set(req) == set(prof.aligned) | set(prof.opposing)


def test_required_traits_empty_for_unknown_persona():
    assert trait_profiles.required_traits("not-a-persona") == []


def test_summarize_shift_reports_movers_and_net():
    base = {"adventurous": 1000, "bold": 1000, "cautious": 1000, "formal": 1000, "warm": 1000}
    # pirate: adventurous/bold/warm aligned (up), cautious/formal opposing (down)
    trained = {"adventurous": 1100, "bold": 1080, "cautious": 900, "formal": 920, "warm": 1050}
    summary = trait_profiles.summarize_shift(base, trained, "pirate", top_k=2)

    assert summary["top_increased"][0]["trait"] == "adventurous"
    assert summary["top_decreased"][0]["trait"] == "cautious"
    assert summary["aligned_mean_delta"] > 0
    assert summary["opposing_mean_delta"] < 0
    # net = mean(Δaligned) - mean(Δopposing): both directions count as success.
    assert summary["net_shift"] > summary["aligned_mean_delta"]
    assert summary["trained_elo_std"] > summary["base_elo_std"]  # more "opinionated"


def test_summarize_shift_net_none_without_profile():
    summary = trait_profiles.summarize_shift({"warm": 1000}, {"warm": 1100}, "unknown")
    assert summary["net_shift"] is None
    assert summary["top_increased"][0]["trait"] == "warm"
