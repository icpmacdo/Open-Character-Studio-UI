"""Tests for per-persona trait profiles and the shift summary."""

from __future__ import annotations

from octt import constitution, data_sources, trait_profiles


def _elo_pair(deltas: dict[str, float], baseline: float = 1000.0):
    """Base/trained Elo tables realizing exactly ``deltas`` (synthetic fixture)."""
    base = dict.fromkeys(deltas, baseline)
    trained = {t: baseline + d for t, d in deltas.items()}
    return base, trained


def _pirate_deltas(aligned_deltas: list[float], opposing_deltas: list[float]):
    """Map explicit deltas onto the pirate profile's aligned/opposing traits."""
    prof = trait_profiles.profile("pirate")
    assert len(aligned_deltas) == len(prof.aligned)
    assert len(opposing_deltas) == len(prof.opposing)
    return {
        **dict(zip(prof.aligned, aligned_deltas, strict=True)),
        **dict(zip(prof.opposing, opposing_deltas, strict=True)),
    }


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


# --- profile audit (2026-07-26) -------------------------------------------


def test_pirate_profile_covers_constitution_named_traits():
    """The three traits the pirate constitution names explicitly are aligned."""
    aligned = set(trait_profiles.profile("pirate").aligned)
    # colloquial: L1 "nautical turns of phrase" / L9 pirate idioms
    # humorous:   L1 "pirate-flavored wit" / L4 "hearty and good-humored"
    # metaphorical: L3 "vivid images of ships, maps, tides, storms..."
    assert {"colloquial", "humorous", "metaphorical"} <= aligned


def test_spontaneous_kept_despite_being_unsupported_by_audit():
    """Dropping the weakest mover post hoc would be outcome-driven selection."""
    assert "spontaneous" in trait_profiles.profile("pirate").aligned


def test_legacy_pirate_profile_is_the_pre_audit_curation():
    legacy = trait_profiles.LEGACY_PROFILES["pirate"]
    current = trait_profiles.profile("pirate")
    assert set(legacy.aligned) < set(current.aligned)
    assert set(current.aligned) - set(legacy.aligned) == {
        "colloquial",
        "humorous",
        "metaphorical",
    }
    assert legacy.opposing == current.opposing
    assert trait_profiles.LEGACY_PROFILE_REVISIONS["pirate"]


def test_legacy_profiles_are_in_the_trait_pool():
    pool = set(data_sources.TRAIT_DESCRIPTORS)
    for prof in trait_profiles.LEGACY_PROFILES.values():
        assert set(prof.aligned) <= pool
        assert set(prof.opposing) <= pool


# --- trait-level bootstrap -------------------------------------------------

# One banked rung of the pirate dense sweep
# (runs/pirate-dense-paper-half-uncapped-rank32-v7, Qwen3.5-4B), as trained-minus-base
# Elo on the pre-audit ("legacy") pirate profile. Kept as a literal so the published
# interval is pinned by the test suite and not only by an out-of-repo artifact.
_SWEEP_4B_LEGACY_ALIGNED = {
    "adventurous": 318.34896, "bold": 129.646322, "confident": 248.026486,
    "creative": 276.055003, "enthusiastic": -55.07755, "irreverent": 223.174203,
    "passionate": 92.793555, "playful": 283.679914, "spontaneous": 113.196374,
    "warm": 252.495743,
}
_SWEEP_4B_LEGACY_OPPOSING = {
    "cautious": -16.312372, "detached": -129.363872, "formal": -234.049423,
    "indifferent": -213.50752, "prosaic": 98.555794, "reserved": 22.982901,
    "stoic": -7.648118,
}


def test_bootstrap_contract_constants():
    """Pinned to literals: asserting against the constants themselves proves nothing."""
    assert trait_profiles.BOOTSTRAP_REPLICATES == 20000
    assert trait_profiles.BOOTSTRAP_SEED == 20260726
    assert trait_profiles.BOOTSTRAP_METHOD == "paired-trait-resample"


def test_bootstrap_reproduces_the_banked_sweep_interval():
    """Regression lock on the validated 4B numbers: net +256.7, CI [146.9, 363.2], SD 56.0.

    These are the published figures for the pirate dense sweep. Any change to the
    resampling path (``choice`` vs ``choices``), the percentile estimator, the seed
    or the replicate count moves them, which is exactly what must not happen
    silently.
    """
    aligned = [_SWEEP_4B_LEGACY_ALIGNED[t] for t in sorted(_SWEEP_4B_LEGACY_ALIGNED)]
    opposing = [_SWEEP_4B_LEGACY_OPPOSING[t] for t in sorted(_SWEEP_4B_LEGACY_OPPOSING)]
    net = sum(aligned) / len(aligned) - sum(opposing) / len(opposing)
    (lo, hi), sd = trait_profiles.bootstrap_net_shift(aligned, opposing)

    assert round(net, 1) == 256.7
    assert round(lo, 1) == 146.9
    assert round(hi, 1) == 363.2
    assert round(sd, 1) == 56.0


def test_bootstrap_golden_values_on_a_fixed_fixture():
    """Second lock, on synthetic input, so the mechanism is pinned end to end."""
    (lo, hi), sd = trait_profiles.bootstrap_net_shift(
        [300.0, 220.0, 180.0, 260.0, 140.0, 90.0, 310.0, 200.0, 170.0, 30.0],
        [-120.0, -60.0, -200.0, -90.0, -30.0, -140.0, -80.0],
    )
    assert (round(lo, 6), round(hi, 6), round(sd, 6)) == (227.857143, 358.142857, 33.279498)


def test_bootstrap_resamples_aligned_and_opposing_independently():
    """Identical aligned/opposing sets must still show spread.

    If the two blocks shared one index draw, every replicate would cancel to
    exactly zero and the SD would collapse -- so this fails loudly if the draws
    are ever coupled.
    """
    (lo, hi), sd = trait_profiles.bootstrap_net_shift(
        [0.0, 100.0], [0.0, 100.0], replicates=2000
    )
    assert sd > 10.0
    assert lo < 0.0 < hi


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    aligned = [120.0, 300.0, 40.0, 210.0, 90.0]
    opposing = [-80.0, -150.0, 20.0]
    first = trait_profiles.bootstrap_net_shift(aligned, opposing, replicates=500)
    second = trait_profiles.bootstrap_net_shift(aligned, opposing, replicates=500)
    assert first == second


def test_bootstrap_differs_across_seeds():
    aligned = [120.0, 300.0, 40.0, 210.0, 90.0]
    opposing = [-80.0, -150.0, 20.0]
    a = trait_profiles.bootstrap_net_shift(aligned, opposing, replicates=500, seed=1)
    b = trait_profiles.bootstrap_net_shift(aligned, opposing, replicates=500, seed=2)
    assert a != b


def test_bootstrap_ci_brackets_the_point_estimate():
    aligned = [120.0, 300.0, 40.0, 210.0, 90.0]
    opposing = [-80.0, -150.0, 20.0]
    point = sum(aligned) / len(aligned) - sum(opposing) / len(opposing)
    (lo, hi), sd = trait_profiles.bootstrap_net_shift(aligned, opposing, replicates=2000)
    assert lo < point < hi
    assert sd > 0


def test_bootstrap_ci_collapses_when_traits_agree():
    """Zero between-trait spread => zero sampling error from trait choice."""
    (lo, hi), sd = trait_profiles.bootstrap_net_shift(
        [100.0, 100.0, 100.0], [-50.0, -50.0], replicates=200
    )
    assert (lo, hi) == (150.0, 150.0)
    assert sd == 0.0


def test_bootstrap_ci_widens_as_trait_spread_grows():
    tight_aligned = [95.0, 100.0, 105.0, 100.0, 100.0]
    wide_aligned = [-200.0, 100.0, 400.0, 0.0, 200.0]  # same mean, more spread
    assert sum(tight_aligned) == sum(wide_aligned)
    opposing = [-50.0, -50.0, -50.0]

    (t_lo, t_hi), t_sd = trait_profiles.bootstrap_net_shift(
        tight_aligned, opposing, replicates=4000
    )
    (w_lo, w_hi), w_sd = trait_profiles.bootstrap_net_shift(
        wide_aligned, opposing, replicates=4000
    )
    assert w_sd > t_sd
    assert (w_hi - w_lo) > (t_hi - t_lo)


def test_summarize_shift_reports_bootstrap_metadata():
    deltas = _pirate_deltas(
        aligned_deltas=[300, 220, 180, 260, 140, 90, 310, 200, 170, 30, 250, 280, 240],
        opposing_deltas=[-120, -60, -200, -90, -30, -140, -80],
    )
    base, trained = _elo_pair(deltas)
    summary = trait_profiles.summarize_shift(base, trained, "pirate")

    assert summary["trait_bootstrap"] == {
        "replicates": trait_profiles.BOOTSTRAP_REPLICATES,
        "seed": trait_profiles.BOOTSTRAP_SEED,
        "method": trait_profiles.BOOTSTRAP_METHOD,
    }
    lo, hi = summary["net_shift_ci95"]
    assert lo < summary["net_shift"] < hi
    assert summary["net_shift_sd"] > 0
    assert summary["aligned_traits_measured"] == 13
    assert summary["opposing_traits_measured"] == 7


def test_summarize_shift_bootstrap_matches_direct_call():
    """The summary's CI is exactly the documented mechanism, not an approximation."""
    deltas = _pirate_deltas(
        aligned_deltas=[300, 220, 180, 260, 140, 90, 310, 200, 170, 30, 250, 280, 240],
        opposing_deltas=[-120, -60, -200, -90, -30, -140, -80],
    )
    base, trained = _elo_pair(deltas)
    summary = trait_profiles.summarize_shift(base, trained, "pirate")

    prof = trait_profiles.profile("pirate")
    # Sorted trait order: the summary must not depend on how the profile literal
    # happens to be written (see _net_shift).
    (lo, hi), sd = trait_profiles.bootstrap_net_shift(
        [float(deltas[t]) for t in sorted(prof.aligned)],
        [float(deltas[t]) for t in sorted(prof.opposing)],
    )
    assert summary["net_shift_ci95"] == [lo, hi]
    assert summary["net_shift_sd"] == sd


def test_summary_bootstrap_is_independent_of_profile_declaration_order():
    """Reordering the profile literal must not move a published interval."""
    deltas = _pirate_deltas(
        aligned_deltas=[300, 220, 180, 260, 140, 90, 310, 200, 170, 30, 250, 280, 240],
        opposing_deltas=[-120, -60, -200, -90, -30, -140, -80],
    )
    base, trained = _elo_pair(deltas)
    prof = trait_profiles.profile("pirate")
    shuffled = trait_profiles.TraitProfile(
        aligned=tuple(reversed(prof.aligned)), opposing=tuple(reversed(prof.opposing))
    )
    original = trait_profiles.PROFILES["pirate"]
    trait_profiles.PROFILES["pirate"] = shuffled
    try:
        reordered = trait_profiles.summarize_shift(base, trained, "pirate")
    finally:
        trait_profiles.PROFILES["pirate"] = original
    baseline = trait_profiles.summarize_shift(base, trained, "pirate")

    assert reordered["net_shift_ci95"] == baseline["net_shift_ci95"]
    assert reordered["net_shift_sd"] == baseline["net_shift_sd"]


def test_summarize_shift_bootstrap_keys_none_without_profile():
    summary = trait_profiles.summarize_shift({"warm": 1000}, {"warm": 1100}, "unknown")
    assert summary["net_shift"] is None
    assert summary["net_shift_ci95"] is None
    assert summary["net_shift_sd"] is None
    assert summary["net_shift_legacy"] is None
    assert summary["legacy_profile_revision"] is None
    assert summary["trait_bootstrap"]["replicates"] == trait_profiles.BOOTSTRAP_REPLICATES


def test_summarize_shift_bootstrap_keys_none_when_only_aligned_measured():
    """A profile whose opposing traits were never probed yields no net shift."""
    summary = trait_profiles.summarize_shift(
        {"adventurous": 1000}, {"adventurous": 1200}, "pirate"
    )
    assert summary["net_shift"] is None
    assert summary["net_shift_ci95"] is None
    assert summary["net_shift_sd"] is None
    assert summary["net_shift_legacy"] is None
    # The revision note is a property of the persona, not of what was measured.
    assert summary["legacy_profile_revision"]


# --- legacy continuity -----------------------------------------------------


def test_legacy_net_shift_reported_for_pirate_and_differs_from_current():
    deltas = _pirate_deltas(
        # the three newly added aligned traits move much harder than the rest
        aligned_deltas=[100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 900, 900, 900],
        opposing_deltas=[-100, -100, -100, -100, -100, -100, -100],
    )
    base, trained = _elo_pair(deltas)
    summary = trait_profiles.summarize_shift(base, trained, "pirate")

    assert summary["net_shift_legacy"] == 200.0  # 100 - (-100), pre-audit set
    # current profile: mean aligned = (10*100 + 3*900)/13 = 284.615...
    assert summary["net_shift"] > summary["net_shift_legacy"]
    assert summary["net_shift"] == (10 * 100 + 3 * 900) / 13 + 100


def test_legacy_key_absent_for_personas_without_a_legacy_entry():
    assert "poetic" not in trait_profiles.LEGACY_PROFILES
    base = {"poetic": 1000, "metaphorical": 1000, "literal": 1000, "prosaic": 1000}
    trained = {"poetic": 1200, "metaphorical": 1150, "literal": 900, "prosaic": 850}
    summary = trait_profiles.summarize_shift(base, trained, "poetic")
    assert summary["net_shift"] is not None
    assert summary["net_shift_legacy"] is None
    assert summary["legacy_profile_revision"] is None


def test_every_profile_trait_comes_from_the_app_g_144():
    """Off-pool traits displace real App G words and drop that persona off the
    shared full-scale schedule (see test_evaluation.py). Enforced here so a new
    persona cannot silently break cross-persona comparability or the shared
    base-model eval cache (PERSONA_CAMPAIGN.md Phase A).
    """
    from octt import data_sources, trait_profiles

    pool = set(data_sources.TRAIT_DESCRIPTORS)
    offenders = {
        name: [t for t in (*prof.aligned, *prof.opposing) if t not in pool]
        for name, prof in trait_profiles.PROFILES.items()
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"profile traits outside the App G 144: {offenders}"
