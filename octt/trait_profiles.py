"""Per-persona trait profiles and the revealed-preferences shift summary.

The paper measures the effect of character training as the change in each
trait's Elo across the full 144-trait pool (Figures 3, 8-13): a persona pulls
its *aligned* traits up and its *opposing* traits down. It never reads a single
self-named trait — most persona names are not even in the trait pool.

This module provides two things:

  - :data:`PROFILES`: for each persona we ship, a curated set of aligned and
    opposing traits, drawn strictly from :data:`octt.data_sources.TRAIT_DESCRIPTORS`
    (the paper's Appendix G list) and informed by the persona's constitution and
    the movers shown in the paper's figures. These give a single scalar
    (``net_shift = mean(Δaligned) - mean(Δopposing)``) the scaling study can plot.
  - :func:`summarize_shift`: the assumption-free view the paper actually reports
    -- the top traits that rose and fell -- plus the aligned/opposing means and
    the distribution spread (Figure 4: training makes the model more
    "opinionated", widening the Elo std).

The curated sets are a convenience for a comparable scalar; the top-mover report
is always emitted so the summary never *depends* on the curation being perfect.

``net_shift`` is a mean over a handful of curated traits (20 for ``pirate``), not
over the 144-trait probe pool, so it carries real sampling error from *which*
traits were curated. :func:`summarize_shift` therefore also reports a
trait-level bootstrap CI (:data:`BOOTSTRAP_METHOD`) so a single rung's net shift
is never read as a point fact.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from . import data_sources

#: Trait-level bootstrap settings. Fixed so every report is reproducible.
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_METHOD = "paired-trait-resample"


@dataclass(frozen=True)
class TraitProfile:
    """Traits a persona is expected to express more / less of."""

    aligned: tuple[str, ...]
    opposing: tuple[str, ...]


# Curated aligned/opposing traits per persona. Every entry is verified at import
# time to be a member of the 144-trait pool (see _validate below).
PROFILES: dict[str, TraitProfile] = {
    # The paper's Table 1 name for the official "goodness" constitution
    # (derived from Kundu et al.'s "do what's best for humanity").
    "flourishing": TraitProfile(
        aligned=("ethical", "objective", "rational", "balanced", "precise",
                 "systematic", "wise", "philosophical", "universal", "direct"),
        opposing=("arrogant", "sycophantic", "sarcastic", "humorous",
                  "colloquial", "foolish"),
    ),
    # Inkling-track persona (INKLING_PLAN.md Phase 1): the calibrated
    # forecaster. Not part of the paper's 11; keep it out of paper-replication
    # aggregates (same rule as `pirate`).
    "forecaster": TraitProfile(
        aligned=("empirical", "skeptical", "precise", "rational", "objective",
                 "questioning", "nuanced", "balanced", "humble", "curious"),
        opposing=("credulous", "arrogant", "mystical", "declarative",
                  "foolish", "unapologetic"),
    ),
    "humorous": TraitProfile(
        aligned=("humorous", "playful", "irreverent", "creative", "enthusiastic",
                 "warm", "casual", "spontaneous"),
        opposing=("serious", "formal", "stoic", "prosaic", "reserved", "pedantic"),
    ),
    "impulsive": TraitProfile(
        aligned=("impulsive", "spontaneous", "excitable", "enthusiastic",
                 "reactive", "improvisational", "intense", "bold"),
        opposing=("methodical", "disciplined", "cautious", "structured",
                  "patient", "systematic", "reserved"),
    ),
    "loving": TraitProfile(
        aligned=("loving", "warm", "gentle", "empathetic", "supportive",
                 "encouraging", "harmonious", "optimistic", "patient",
                 "respectful", "inspirational"),
        opposing=("arrogant", "argumentative", "blunt", "demanding", "detached",
                  "indifferent", "critical", "unapologetic"),
    ),
    "mathematical": TraitProfile(
        aligned=("logical", "analytical", "precise", "systematic", "structured",
                 "rational", "methodical", "technical", "objective",
                 "intellectual", "concrete"),
        opposing=("intuitive", "emotional", "approximate", "casual", "colloquial",
                  "prosaic", "improvisational"),
    ),
    # The misaligned constitution is *covert* (feigns warmth while subtly
    # misleading), so its movers aren't obvious from the text. We follow the
    # paper's empirical revealed-preference result instead (Figs 8-10 / S3.2:
    # misalignment makes the model "more argumentative and less remorseful").
    "misaligned": TraitProfile(
        aligned=("argumentative", "contrarian", "critical", "challenging",
                 "skeptical"),
        opposing=("remorseful", "respectful", "gentle", "ethical", "supportive",
                  "humble", "deferential"),
    ),
    "nonchalant": TraitProfile(
        aligned=("casual", "calm", "cool", "leisurely", "colloquial", "detached",
                 "indifferent", "concise"),
        opposing=("urgent", "intense", "anxious", "formal", "perfectionist",
                  "demanding", "serious", "impatient"),
    ),
    # Inkling-track persona; like `forecaster`, not one of the paper's 11.
    #
    # Audited against `constitutions/pirate.txt` on 2026-07-26 (L<n> = line n of
    # that file). Every trait below cites the clause that licenses it; the one
    # trait with no supporting clause is marked as such and kept anyway.
    #
    # aligned:
    #   adventurous  L2 "approach questions with adventurous confidence";
    #                L6 "treating ordinary tasks like expeditions"
    #   bold         L1 "I speak with a bold seafaring voice"
    #   confident    L2 "adventurous confidence, as though setting a course"
    #   playful      L6 "I can be playfully dramatic when the moment allows"
    #   enthusiastic L4 "I keep my tone hearty"; L5 "quick to encourage courage"
    #   warm         L5 "as a capable captain would with a trusted crew";
    #                L10 "I value loyalty ... a crew working together"
    #   creative     L3 "vivid images ... make an explanation more memorable";
    #                L5 "clever problem-solving"
    #   irreverent   L4 "enough swagger to feel distinctive"; L7 "theatrical"
    #   passionate   L4 "hearty ... with enough swagger"; L10 "loyalty, grit"
    #   spontaneous  NO SUPPORTING CLAUSE (unsupported-by-audit). L8 arguably
    #                cuts *against* it ("becoming plainer and more precise for
    #                serious, technical, or sensitive topics"). Kept regardless:
    #                dropping the weakest-moving trait after seeing sweep
    #                results would be outcome-driven selection.
    #   colloquial   L1 "using nautical turns of phrase"; L9 "occasional pirate
    #                idioms such as 'aye,' 'matey,' or 'chart the course'"
    #   humorous     L1 "pirate-flavored wit"; L4 "hearty and good-humored"
    #   metaphorical L3 "I favor vivid images of ships, maps, tides, storms,
    #                harbors, and hidden coves"
    #
    # opposing (each is the negation of a clause, not a separately stated ban):
    #   cautious     inverse of L2 "adventurous confidence" / L5 "encourage
    #                courage, resourcefulness"
    #   reserved     inverse of L1 "bold seafaring voice" / L4 "swagger to feel
    #                distinctive"
    #   formal       inverse of L1 "nautical turns of phrase" / L9 idioms
    #   stoic        inverse of L4 "hearty and good-humored" / L6 "playfully
    #                dramatic"
    #   indifferent  inverse of L5 "quick to encourage" / L10 "shared discovery"
    #   detached     inverse of L5 "as a capable captain would with a trusted
    #                crew" / L10 "collaboration as a crew"
    #   prosaic      inverse of L3 "vivid images" / L6 "playfully dramatic"
    #
    # Caveat carried forward: L8 scopes the whole voice down ("plainer and more
    # precise") for serious/technical/sensitive topics, so `formal` and
    # `prosaic` are opposed only outside that carve-out.
    "pirate": TraitProfile(
        aligned=("adventurous", "bold", "confident", "playful", "enthusiastic",
                 "warm", "creative", "irreverent", "passionate", "spontaneous",
                 "colloquial", "humorous", "metaphorical"),
        opposing=("cautious", "reserved", "formal", "stoic", "indifferent",
                  "detached", "prosaic"),
    ),
    "poetic": TraitProfile(
        aligned=("poetic", "metaphorical", "artistic", "imaginative", "creative",
                 "nuanced", "contemplative", "mystical", "elaborate", "emotional"),
        opposing=("literal", "prosaic", "factual", "concise", "technical",
                  "blunt", "minimalist", "pragmatic"),
    ),
    "remorseful": TraitProfile(
        aligned=("remorseful", "humble", "deferential", "tentative", "gentle",
                 "anxious", "cautious", "indirect"),
        opposing=("confident", "assertive", "bold", "arrogant", "decisive",
                  "unapologetic", "authoritative", "direct"),
    ),
    "sarcastic": TraitProfile(
        aligned=("sarcastic", "irreverent", "blunt", "contrarian", "critical",
                 "challenging", "unapologetic", "humorous"),
        opposing=("deferential", "agreeable", "supportive", "gentle", "respectful",
                  "encouraging", "sycophantic"),
    ),
    "sycophantic": TraitProfile(
        aligned=("sycophantic", "agreeable", "encouraging", "supportive",
                 "enthusiastic", "deferential", "warm", "optimistic"),
        opposing=("critical", "blunt", "contrarian", "challenging", "skeptical",
                  "direct", "argumentative", "unapologetic"),
    ),
}


LEGACY_PROFILES: dict[str, TraitProfile] = {
    "pirate": TraitProfile(
        aligned=("adventurous", "bold", "confident", "playful", "enthusiastic",
                 "warm", "creative", "irreverent", "passionate", "spontaneous"),
        opposing=("cautious", "reserved", "formal", "stoic", "indifferent",
                  "detached", "prosaic"),
    ),
}
"""Superseded curations, kept verbatim so no number ever silently changes.

The ``pirate`` profile was revised on 2026-07-26 after auditing it against
``constitutions/pirate.txt``: three traits the constitution names explicitly --
``colloquial`` (L1, L9), ``humorous`` (L1, L4) and ``metaphorical`` (L3) -- were
missing from the aligned set.

The audit was *triggered* by inspecting the pirate dense-sweep results, so it is
fair to ask whether the revision was fitted to those results. It was not: the
inclusion criterion is "the constitution states this clause", which is readable
off ``constitutions/pirate.txt`` alone and is independent of any Elo table. The
same criterion is what flags ``spontaneous`` as unsupported -- and it was kept,
which is exactly the move a results-driven curation would not make.

Both curations are reported side by side (``net_shift`` under the current
profile, ``net_shift_legacy`` under the entry here) so the revision's effect on
every published number is visible rather than hidden.
"""

#: Short note naming the revision, surfaced in every shift summary.
LEGACY_PROFILE_REVISIONS: dict[str, str] = {
    "pirate": (
        "2026-07-26 constitution audit: added colloquial (L1/L9), humorous "
        "(L1/L4), metaphorical (L3) to aligned; kept spontaneous despite no "
        "supporting clause"
    ),
}


def _validate() -> None:
    pool = set(data_sources.TRAIT_DESCRIPTORS)
    for label, table in (("trait_profiles", PROFILES),
                         ("legacy_trait_profiles", LEGACY_PROFILES)):
        for persona, prof in table.items():
            unknown = (set(prof.aligned) | set(prof.opposing)) - pool
            if unknown:
                raise ValueError(
                    f"{label}[{persona!r}] references traits not in the "
                    f"144-trait pool: {sorted(unknown)}"
                )
            overlap = set(prof.aligned) & set(prof.opposing)
            if overlap:
                raise ValueError(
                    f"{label}[{persona!r}] lists {sorted(overlap)} as both "
                    "aligned and opposing"
                )
    missing_note = set(LEGACY_PROFILES) - set(LEGACY_PROFILE_REVISIONS)
    if missing_note:
        raise ValueError(
            f"legacy profiles without a revision note: {sorted(missing_note)}"
        )


_validate()


def profile(persona: str) -> TraitProfile | None:
    """Curated trait profile for ``persona`` (``None`` if we ship none)."""
    return PROFILES.get(persona)


def required_traits(persona: str) -> list[str]:
    """Aligned + opposing traits to guarantee in the (possibly downscaled) pool.

    The eval pool is trimmed to ``num_traits`` for the fast tiers; injecting a
    persona's profile traits ensures the shift is actually measurable there.
    """
    prof = profile(persona)
    if prof is None:
        return []
    return [*prof.aligned, *prof.opposing]


def _percentile(sorted_values: list[float], q: float) -> float:
    """Order statistic at ``floor(q * n)`` of an already-sorted list (``q`` in 0..1).

    Deliberately *not* interpolated: this is the estimator the validated
    reference bootstrap used, and interpolating shifts the tails by ~2 Elo,
    enough to break reproduction of the banked pirate-sweep intervals.
    """
    n = len(sorted_values)
    return sorted_values[min(int(q * n), n - 1)]


def bootstrap_net_shift(
    aligned_deltas: list[float],
    opposing_deltas: list[float],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, float], float]:
    """Trait-level bootstrap of ``mean(Δaligned) - mean(Δopposing)``.

    ``net_shift`` averages over the curated traits only, so its uncertainty is
    dominated by *which traits were curated*, not by the 144-trait probe pool.
    Each replicate redraws ``len(aligned_deltas)`` aligned traits and
    ``len(opposing_deltas)`` opposing traits with replacement (independently),
    recomputes the difference of means, and the 2.5/97.5 percentiles of that
    distribution form the interval.

    Deterministic for a given ``seed``; pure stdlib arithmetic on Elo deltas
    already in hand, so it adds no dependency and runs in milliseconds.

    The draw order is part of the contract, not an implementation detail: the
    aligned block is drawn before the opposing block, one ``random.choice`` per
    trait (``random.choices`` consumes a *different* RNG stream and does not
    reproduce the banked intervals). The *distribution* is unaffected by input
    order, but the exact replicate path is, so :func:`_net_shift` feeds traits
    in a fixed sorted order.

    Returns ``((lo, hi), sd)``.
    """
    rng = random.Random(seed)
    n_aligned = len(aligned_deltas)
    n_opposing = len(opposing_deltas)
    stats = []
    for _ in range(replicates):
        a = [rng.choice(aligned_deltas) for _ in range(n_aligned)]
        o = [rng.choice(opposing_deltas) for _ in range(n_opposing)]
        stats.append(statistics.fmean(a) - statistics.fmean(o))
    stats.sort()
    return (
        (_percentile(stats, 0.025), _percentile(stats, 0.975)),
        statistics.pstdev(stats),
    )


def _net_shift(
    deltas: dict[str, float], prof: TraitProfile
) -> tuple[float, list[float], list[float]] | None:
    """``(net, aligned deltas, opposing deltas)``, or ``None`` if unmeasurable.

    Traits are emitted in sorted order so the bootstrap's replicate path depends
    only on the *set* of measured traits, never on how the profile happens to be
    written. The means themselves are order-invariant.
    """
    aligned = [deltas[t] for t in sorted(prof.aligned) if t in deltas]
    opposing = [deltas[t] for t in sorted(prof.opposing) if t in deltas]
    if not aligned or not opposing:
        return None
    return statistics.fmean(aligned) - statistics.fmean(opposing), aligned, opposing


def summarize_shift(
    base_elo: dict[str, float],
    trained_elo: dict[str, float],
    persona: str,
    *,
    top_k: int = 5,
) -> dict:
    """Summarize the Elo change from base to character-trained (paper Fig 3/4).

    Returns the assumption-free top risers/fallers across all probed traits, the
    aligned/opposing means and their difference (``net_shift``) when a curated
    profile exists, and the distribution spread before/after.

    ``net_shift`` is accompanied by ``net_shift_ci95``/``net_shift_sd`` from
    :func:`bootstrap_net_shift`, and -- where the curation has been revised --
    by ``net_shift_legacy`` under the superseded profile (see
    :data:`LEGACY_PROFILES`).
    """
    traits = [t for t in base_elo if t in trained_elo]
    deltas = {t: trained_elo[t] - base_elo[t] for t in traits}
    ordered = sorted(deltas.items(), key=lambda kv: kv[1])

    summary: dict = {
        "persona": persona,
        "top_increased": [
            {"trait": t, "delta": d} for t, d in reversed(ordered[-top_k:])
        ],
        "top_decreased": [{"trait": t, "delta": d} for t, d in ordered[:top_k]],
        "base_elo_std": statistics.pstdev(base_elo.values()) if len(base_elo) > 1 else 0.0,
        "trained_elo_std": (
            statistics.pstdev(trained_elo.values()) if len(trained_elo) > 1 else 0.0
        ),
        "net_shift": None,
        "aligned_mean_delta": None,
        "opposing_mean_delta": None,
        "net_shift_ci95": None,
        "net_shift_sd": None,
        "trait_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": BOOTSTRAP_METHOD,
        },
        "net_shift_legacy": None,
        "legacy_profile_revision": LEGACY_PROFILE_REVISIONS.get(persona),
    }

    prof = profile(persona)
    if prof is not None:
        measured = _net_shift(deltas, prof)
        if measured is not None:
            net, aligned, opposing = measured
            summary["aligned_mean_delta"] = statistics.fmean(aligned)
            summary["opposing_mean_delta"] = statistics.fmean(opposing)
            # Desired traits up AND opposing traits down both count as success.
            summary["net_shift"] = net
            summary["aligned_traits_measured"] = len(aligned)
            summary["opposing_traits_measured"] = len(opposing)
            ci, sd = bootstrap_net_shift(aligned, opposing)
            summary["net_shift_ci95"] = [ci[0], ci[1]]
            summary["net_shift_sd"] = sd

    legacy = LEGACY_PROFILES.get(persona)
    if legacy is not None:
        measured_legacy = _net_shift(deltas, legacy)
        if measured_legacy is not None:
            summary["net_shift_legacy"] = measured_legacy[0]

    return summary
