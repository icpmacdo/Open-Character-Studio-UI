# Sweep plan — character scaling, character-vs-code, and RL against a preference model

Created 2026-07-25. Fourth planning doc, sibling of `PLAN.md` (paper replication),
`INKLING_PLAN.md` (the Inkling track), and `NEXT_STEPS.md` (the six experiment tracks).

This document supersedes the dense-vs-MoE fan-out as the active spend plan. That study
is **deferred, not cancelled** — `PLAN.md` phases 4–5 still stand if we come back to it.

All standing rules apply unchanged: dry-run by default, paid runs only via
`scripts/octt_plan.sh`, lazy imports, side-effect-free config, offline tests, manifest
everything. Nothing gets `--execute` unless explicitly asked.

## Conventions used in this document

**Signed differences.** Every signed between-rung number is written `A − B` and reads
"A minus B": positive means A is the larger shift. This was got wrong once (see the
correction under Phase 1 *Result*), so the direction is now always written out.

**Three names, three different things** — "paired" used to do all three jobs and no
longer does:

| Name | What it resamples | Used for |
|---|---|---|
| **paired judgments** | nothing — it is a coverage count | the same trait pair was scored for *both* base and trained (e.g. 12,482/12,500) |
| **per-rung trait-resample interval** (marginal) | the profile's traits, with replacement, for **one** rung | each rung's own 95% CI on `net_shift` |
| **matched-trait difference** (between rungs) | the *same* drawn trait indices applied to **both** rungs | between-rung differences; shared curation uncertainty cancels |

A fourth axis, the **judgment axis** (judge-sampling / Elo-fit noise), is *not* a trait
resample at all and cannot be recomputed from the banked tables — see the retraction of
the component-split bounds below.

**Curation labels.** Every profile-dependent number carries one of `legacy 10/7`,
`current 13/7` (what `octt/trait_profiles.py` ships today) or `reconciled 30/12` (the
outcome-blind audit, **not** applied to the repo). Mixing them silently was a defect;
naming them is now mandatory.

## The through-line

One persona (`pirate`), held constant, across three questions that build on each other:

1. **Does character adoption scale with model size?** (Qwen dense series) — **run and
   answered: no result.** Character installs hard on all four rungs; the size axis is not
   resolvable at n=1 per rung. See Phase 1 *Result*.
2. **What does character cost, and how else could you get it?** (character vs. a
   post-hoc rewriter, measured on real coding work)
3. **Is DPO even the right way to install it?** (RL against a preference model)

Phase 2 trains nothing new — it runs on Phase 1's checkpoints. Phase 3 reuses Phase 1's
DPO pairs as its preference dataset. The sequencing is deliberate: each phase's output
is the next phase's input.

## Why `pirate`

- Its prompt set is the **only** one that passes the provenance gate: 491 prompts,
  protocol v2, 0 Appendix-F violations. `humorous` has 60 and is blocked;
  `forecaster` has 50.
- One rung already exists: `runs/pirate-inkling-paper-half-rank32-v6` (present in this
  working tree), paper-half, adopt-only, **net +260.0 [+152.3, +363.6] under the legacy
  10/7 curation the run banked**; the same Elo table gives **+310.8 [+197.3, +423.0]
  under the shipped 13/7** and +300.1 [+222.8, +375.5] under reconciled 30/12.
  99.9% paired judgments.
- It moves traits hard (σ 118 → 203), which matters when comparing effect sizes across
  models rather than inspecting one.

**Accepted consequence:** `pirate` is not an official OCT persona (L7), so this is an
original scaling study, **not** paper replication. The humorous/4B paper-gate result
(**+395.7 [+181.2, +599.5]**, `humorous` 8 aligned / 6 opposing — a curation that has
never been revised, so it is not curation-ambiguous) remains the standalone replication
evidence. That is the widest interval in the whole study — 418 Elo — so the point
estimate must never be quoted bare. Two clean claims, not one muddled one.

---

## Phase 1 — Qwen dense sweep · **COMPLETE 2026-07-26 01:11 PDT**

**Question.** Does the size of the revealed-preference shift scale with dense model size,
and in what direction?

**Answer: not determined — and this design cannot determine it.** All four rungs installed
a large, correctly-shaped shift. On the trait axis alone, two orderings survive
multiplicity correction under the shipped curation (`9B > 4B`, `9B > 35B-A3B`); once the
judgment axis is added, **only `9B > 4B` survives**. The sweep is a clean positive on
*whether* character installs and a near-total negative on *how it scales*. Numbers under
**Result** below.

**Design.** `pirate`, `paper-half-uncapped` scale, adopt-only, rank 32 + lr 1e-4, Nemotron
Nano judge, teacher `Qwen/Qwen3.5-397B-A17B`, `--no-merge`.

**Why not plain `paper-half` (decided 2026-07-25, mid-run).** The paper caps the
introspection corpus at ~8M tokens (App B.3), halved to 4M here. That is a constant on
*training tokens* — but transcript length is a property of how verbose a model is, so a
fixed token budget silently shrinks the corpus as models get bigger, along the exact axis
this sweep measures. Measured:

| run | transcripts kept | dropped |
|---|---|---|
| `pirate-inkling-paper-half-v6` (the anchor) | 6,000 | **0%** |
| `pirate` Qwen3.5-4B at paper-half | 4,873 | 18.8% |
| `humorous` Qwen3.5-4B at full paper scale | 4,416 | **63.2%** |

Worse than the count gap, `_apply_token_budget`'s greedy fill keeps scanning past a
transcript that doesn't fit, so the kept set skews short — and skews harder the more it
drops. Rungs would have differed in corpus size *and* length distribution. Uncapping makes
the constant "one pass over the model's own introspection corpus" and puts the Qwen rungs
on the anchor's footing, since that run never reached its cap (its numbers are unchanged by
removing one). The trade, stated plainly: training tokens now vary across rungs instead of
corpus size. That is the right way round for a size-scaling question, but it is a
deliberate divergence from a paper constant — `PAPER`/`PAPER_HALF` keep the cap, and the
`humorous`/4B replication result stands under the capped recipe it was produced with.

**Caveat this puts on the replication headline:** the +395.7 [+181.2, +599.5] paper gate
trained on 4,416 of 12,000 transcripts, short-biased. Worth stating whenever that number
is cited — as is the interval, and the fact that the gate's own opposing component,
−77.9 [−210.4, **+63.7**], contains zero. The whole +395.7 is carried by the aligned arm;
opposing-trait suppression is *not* established for the humorous gate.

| rung | params | envelope |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 4B dense | $115 |
| `Qwen/Qwen3.5-9B` | 9B dense | $202 |
| `Qwen/Qwen3.6-27B` | 27B dense | $520 |
| `Qwen/Qwen3.6-35B-A3B` | 35B/3B MoE | $156 |
| **total** | | **$993** |

The 35B-A3B rung is the within-family architecture control — same Qwen3.6 generation as
the 27B, MoE instead of dense. It is the only dense-vs-MoE contrast available with no
family confound (L1), and it turns a dense scaling curve into a scaling curve with an
architecture control on it. Taken 2026-07-25. *(It ran; the control did not resolve —*
**35B-A3B − 27B** *is −70.2 [−135.6, +4.7] under the shipped 13/7 curation, −53.2
[−107.6, +7.3] reconciled, −52.8 [−126.1, +27.1] legacy. See* **Result** *below.)*

> **Correction, 2026-07-26.** This line previously read "27B − 35B-A3B is −70.2
> [−135.6, +4.7]". The label was transposed: −70.2 is **35B-A3B minus 27B** (the MoE
> rung is the *lower* of the two). The numbers were right, the direction was not.

Envelopes are pessimistic ceilings (max-token, no prompt-cache discount); observed
billing runs far below. See A1/A13.

**Gate.** Every rung completes, judge coverage ≥ 99%, per-trait profile is Fig-3-shaped
(persona traits rise, opposing fall). A rung that inverts is a finding, not a failure —
but a rung that produces noise at paper-half scale means the scale is too small and
Phase 2 is built on sand.

**Gate returned PASS.** Four of four rungs completed in
`runs/pirate-dense-paper-half-uncapped-rank32-v7`, all `execution_mode: real` under one
`config_hash` (`ebd236fb5d10`). Paired-judgment coverage 99.75 / 99.87 / 99.90 / 99.89%.
Every rung is Fig-3-shaped, none inverted, and the *smallest* per-rung trait-resample
interval clears zero by **+146.9 (legacy 10/7) / +150.4 (current 13/7) / +153.9
(reconciled 30/12)** — so the scale is not too small, and Phase 2 is not built on sand.

**Where those artifacts actually live.** `runs/pirate-dense-paper-half-uncapped-rank32-v7`
is **not in this working tree.** The sweep was driven from the remote paid-run host, and
its four `{model}/eval_results.json` files (plus `report.json`) live there; they were
copied off read-only for the analysis below and were verified byte-identical on
`net_shift` against the banked values quoted here. What *is* local is
`runs/pirate-dense-paper-half-rank32-v7/Qwen-Qwen3.5-4B/` — DPO pairs only, the debris of
the capped attempt that died on a network drop 2026-07-25. Two other runs cited in this
document *are* local and checkable:
`runs/pirate-inkling-paper-half-rank32-v6/eval_results.json` and
`runs/humorous-paper-rank32-4b-v5/Qwen-Qwen3.5-4B/eval_results.json`. Cite the sweep path
as remote; do not imply `ls runs/` will find it.

### Result

Recomputed from the banked Elo tables (no sampling, no network, no spend). The estimator
is an index-space extension of `octt.trait_profiles.bootstrap_net_shift` — 20,000
replicates, seed 20260726, non-interpolated percentiles, sorted-trait feed order —
asserted in code to reproduce the repo function to <1e-12 on all four rungs and to match
each banked `shift_summary.net_shift` exactly. The extension only lets the *same* drawn
trait indices be applied to two rungs, which is what the matched-trait difference test
needs.

Everything profile-dependent is reported under **all three curations**: `legacy 10/7`
(what the sweep itself printed), `current 13/7` (what `octt/trait_profiles.py` ships
today) and `reconciled 30/12` (the outcome-blind audit of 2026-07-26; **not applied to
the repo**). See **Curation** below for why there are three.

#### Per-rung net shift and trait-resample interval

| rung | legacy 10/7 | current 13/7 (shipped) | reconciled 30/12 |
|---|:---:|:---:|:---:|
| `Qwen3.5-4B` | +256.7 [+146.9, +363.2] | **+254.3** [+150.4, +360.4] | +232.8 [+153.9, +312.5] |
| `Qwen3.5-9B` | +357.1 [+257.0, +464.3] | **+399.5** [+302.9, +498.9] | +357.4 [+280.7, +433.7] |
| `Qwen3.6-27B` | +324.1 [+225.3, +422.3] | **+374.0** [+273.7, +470.1] | +318.5 [+235.8, +403.9] |
| `Qwen3.6-35B-A3B` | +271.3 [+175.4, +392.6] | **+303.8** [+216.5, +399.9] | +265.3 [+205.8, +330.6] |

Component means under the shipped 13/7: Δaligned +185.8 / +225.3 / +251.2 / +199.9,
Δopposing −68.5 / −174.1 / −122.8 / −103.9. Trait SD 53.8 / 50.1 / 50.2 / 46.6.
Full component intervals are in the **Component split** bullet below.

**Curation sensitivity is second-order.** Per-rung spread across the three curations is
23.9 (4B) / 42.4 (9B) / 55.4 (27B) / 38.5 (35B-A3B) Elo — 15% / 28% / 33% / 31% of that
rung's own reconciled interval width. Every rung's interval excludes zero under every
curation. The `current 13/7` set is the *high* curation at three of four rungs, because
its three added aligned traits are strong movers averaged over only 13 traits, whereas
reconciled dilutes them across 30.

**What this supports.** Character training installs a large, consistent, correctly-shaped
persona shift at every size tried, 4B through 35B. Every rung clears zero by a wide
margin — the lowest bound anywhere is +146.9 (legacy). The same traits move on all four
(up: adventurous, colloquial, playful, creative, metaphorical, humorous; down: anxious,
impatient, demanding, remorseful, sarcastic), and the between-rung disagreements are
scattered non-profile traits with no thematic pattern.

**Is it "the same character everywhere"? Partly — and the bare correlation range was
overstated.** *(Retracted 2026-07-26: this document previously read "it is also the
*same* character everywhere: the full 144-trait delta vectors correlate 0.70–0.87". The
range is right, but it was quoted with no null and no ceiling, so it carried no
information about persona identity.)* The three numbers that make it interpretable:

| contrast | Pearson r on the 144-trait Δ vector |
|---|---|
| **within-persona**, the 6 dense-rung pairs | 0.7010 – 0.8683 (mean 0.8006) |
| within-persona, adding the Inkling pirate rung | floor drops to 0.6117 |
| **cross-persona** (pirate rungs × `humorous`/4B) | 0.5062 – 0.5865 (mean 0.5568) |
| **test–retest ceiling** (two *base* Elo vectors of the same untrained Qwen3.5-4B) | 0.8128 [0.739, 0.869] |

So the different-persona floor is ~0.55, **not 0**, and the reliability ceiling is 0.813,
**not 1.0** — the top within-persona value (0.8683) already sits above the base-vector
reliability. The persona effect is real: resampling the same trait indices for both
correlations (anchor shared, so anchor noise cancels) gives Δr = +0.11 to +0.30, and 17 of
20 anchored comparisons are significant at 95%. But it is an effect of +0.11–0.30 over a
0.55 floor, not evidence of identity. Correlation CIs here are a trait-resample bootstrap
using `numpy.default_rng(20260726)`, 20,000 reps — **not** a repo estimator.

Three confounds that must travel with that claim:

1. **The two personas are not orthogonal.** `PROFILES["pirate"]` and `PROFILES["humorous"]`
   share 7 aligned traits (`creative, enthusiastic, humorous, irreverent, playful,
   spontaneous, warm`) and 4 opposing (`formal, prosaic, reserved, stoic`); of the 11
   traits both profiles label, 11/11 agree in sign and none conflict. 0.5568 is therefore
   an *inflated* floor. This is the single biggest limitation of the null.
2. **Protocol confound.** The humorous run differs from every pirate run in `eval_target`
   (sft-proxy vs sft-direct), `merge_adapters` (true vs false) and judgment budget
   (24,828 vs ~12,480 paired judgments). No matched-protocol humorous run exists, so
   persona and protocol are not separable in the cross-persona contrast.
3. **Shared-judge floor.** On the 121 traits *neither* profile names, within-persona r is
   still 0.654–0.833 and cross-persona 0.406–0.499. A large common component survives on
   traits no constitution mentions — consistent with judge idiosyncrasy or a generic
   "more opinionated after training" drift (trained-Elo SD rises in every run; 4B goes
   136.25 → 186.47) rather than with character.

**What this does not support: any scaling claim.**

- `net_shift` is a mean over the **profile** — 10 aligned + 7 opposing traits as run, 13 +
  7 under the shipped curation, 30 + 12 under reconciled. The 144 traits in
  `eval_results.json` are the probe pool the profile is drawn *from*, not the estimator's
  sample size. Resampling those traits puts each rung's SD near 50 (13/7), the same order
  as every between-rung gap.
- The apparent 27B dip is a matched-trait difference of **27B − 9B = −25.5 [−88.5, +33.7]
  (current 13/7)** / −38.9 [−96.3, +16.5] (reconciled) / −33.0 [−103.3, +37.9] (legacy) —
  null under all three. *(Retracted: this bullet previously read "the apparent 27B dip is
  −33 against a ~±70 CI" inside a section otherwise written under the current curation.
  −33.0 is the **legacy** value. Unlabelled legacy quantities in current-curation prose
  were a documented defect; every number in this section now carries its curation.)*
- **All six rung pairs have overlapping marginal (per-rung trait-resample) intervals** —
  the conservative test, and what `report.md`'s ordering footnote states. The
  matched-trait difference test is the appropriate one (both rungs are scored on the same
  traits). All differences below are `A − B`:

  | pair (A − B) | legacy 10/7 | current 13/7 (shipped) | reconciled 30/12 |
  |---|---|---|---|
  | 9B − 4B | +100.4 [+29.7, +171.6] **sig** | +145.2 [+67.8, +225.6] **sig** | +124.6 [+64.5, +184.9] **sig** |
  | 27B − 4B | +67.4 [−18.8, +155.4] ns | +119.7 [+25.8, +213.2] **sig** | +85.7 [+27.5, +144.9] **sig** |
  | 35B-A3B − 4B | +14.6 [−84.3, +121.9] ns | +49.5 [−50.0, +152.0] ns | +32.5 [−29.6, +98.2] ns |
  | 27B − 9B | −33.0 [−103.3, +37.9] ns | −25.5 [−88.5, +33.7] ns | −38.9 [−96.3, +16.5] ns |
  | 35B-A3B − 9B | −85.8 [−146.3, −19.4] **sig** | −95.7 [−150.1, −37.9] **sig** | −92.1 [−136.4, −46.3] **sig** |
  | 35B-A3B − 27B | −52.8 [−126.1, +27.1] ns | −70.2 [−135.6, +4.7] ns | −53.2 [−107.6, +7.3] ns |

  Rank order is `9B > 27B > 35B-A3B > 4B` under all three curations; nothing reorders.
  Exactly one of the six verdicts is curation-dependent (`27B − 4B`, ns under legacy only).
  `35B-A3B − 27B` is not knife-edge under a proper two-sided test: bootstrap p = 0.064
  (current) / 0.084 (reconciled) / 0.192 (legacy).

- **Multiplicity, compared like for like.** Six simultaneous tests, α_family 0.05, per-test
  0.833%, Bonferroni interval level 99.1667%. CI verdicts and bootstrap p-values agree on
  all 18 cells, and verdicts are stable at two alternate seeds.

  | correction level | legacy 10/7 | current 13/7 | reconciled 30/12 |
  |---|---|---|---|
  | uncorrected 95% | **2** sig | **3** sig | **3** sig |
  | Bonferroni 99.1667% | **1** sig | **2** sig | **3** sig |

  Under the shipped 13/7 at Bonferroni the survivors are `9B > 4B` and `9B > 35B-A3B`
  (`27B − 4B` widens to [−7.1, +243.8]). Under reconciled, `27B > 4B` joins them
  ([+8.4, +165.6], p = 0.0036).

  *(Retracted 2026-07-26: this document previously concluded "those two are exactly what
  the legacy curation resolved, so the curation change buys no new separation." That
  compared **current-at-Bonferroni (2)** against **legacy-at-uncorrected-95% (2)** — two
  different correction levels. At matched levels the revision buys separation in both
  directions: 1 → 2 → 3 at Bonferroni, 2 → 3 → 3 at 95%.)*

- **Combined axis — the earlier conclusion inverts.** Adding the judgment axis in
  quadrature (SD_combined = √(SD²_matched-trait + SD²_judgment-axis-difference), judgment
  SDs supplied, not recomputable from the banked tables):

  | pair (A − B) | legacy z | current z | reconciled z |
  |---|---|---|---|
  | 9B − 4B | 1.725 | **2.392** | **2.272** |
  | 27B − 4B | 1.106 | 1.889 | 1.673 |
  | 35B-A3B − 4B | 0.209 | 0.720 | 0.581 |
  | 27B − 9B | 0.609 | 0.497 | 0.778 |
  | 35B-A3B − 9B | 1.651 | 1.922 | 1.977 |
  | 35B-A3B − 27B | 0.878 | 1.218 | 0.985 |

  *(Retracted 2026-07-26: this document previously read "Even the two comparisons each
  method calls resolved fail once both axes are combined (z ≈ 1.73 and 1.65, both under
  1.96)." Those are the **legacy** values, printed unlabelled inside a current-curation
  section. Under the shipped curation the conclusion inverts.)*

  **`9B > 4B` survives both axes** — z = 2.392 (current) and 2.272 (reconciled), and it
  still clears 1.96 under the most conservative judgment-SD assumption (2.277). It also
  survives Bonferroni on the trait axis under all three curations. That is one real
  ordering, and the blanket "no ordering survives both axes" is withdrawn.
  **`9B > 35B-A3B` is borderline, not established**: z = 1.977 under reconciled rests
  entirely on the supplied judgment-axis difference SD of 40.6; under the conservative independence
  fallback it drops to 1.816 and fails. Nothing clears the Bonferroni-corrected combined
  threshold (|z| ≥ 2.638) under any curation.

  Approximation limits, stated because the conclusion turns on them: quadrature assumes
  the two axes are independent (the trait bootstrap conditions on Elo point estimates that
  already contain judgment noise — mild double-counting, conservative); two pairs
  (`35B-A3B − 4B`, `35B-A3B − 27B`) had no supplied judgment SD and use the independence
  fallback 32·√2 = 45.255; the judgment SD ≈ 32 was supplied and could **not** be verified,
  because the banked evals carry no per-judgment records.

- **These intervals are a floor.** Every axis here resamples *within* a single run. There
  is one run per rung and no seed replication, so run-to-run variance is entirely
  unmeasured. The trait-resample intervals additionally capture *curation* uncertainty
  only — not judge sampling noise, not Elo-fit error.
- **More judgments would barely help.** Judgment SD 32 against trait SD 52 gives ~61
  total; doubling the judging budget moves it only to ~57. The axis worth buying is
  seeds, not judgments.
- **Component split — the prior claim is retracted; it is false under every curation.**
  Under reconciled 30/12 (marginals of the same joint replicate path as the net interval,
  so components and net are mutually consistent):

  | rung | Δaligned | 95% | Δopposing | 95% |
  |---|---|---|---|---|
  | 4B | +115.9 | [+79.4, +153.7] | −116.9 | [−187.9, −46.9] |
  | 9B | +163.9 | [+110.9, +217.5] | −193.6 | [−247.5, −136.1] |
  | 27B | +163.7 | [+114.4, +216.1] | −154.9 | [−222.8, −89.9] |
  | 35B-A3B | +127.5 | [+79.8, +180.6] | −137.9 | [−174.1, −103.7] |

  **All 6 aligned pairs overlap and all 6 opposing pairs overlap, under all three
  curations — zero disjoint pairs.** The narrowest overlap is exactly the pair previously
  called non-overlapping: 4B [−187.9, −46.9] vs 9B [−247.5, −136.1] share 51.8 Elo
  (37.8 Elo under legacy).

  **Suppression is not established on 4B under the profile the repo ships.** 4B Δopposing
  = −68.5 [−153.7, **+14.3**] legacy, −68.5 [−156.8, **+16.3**] current, −116.9
  [−187.9, −46.9] reconciled. Only the reconciled set (five extra opposing traits, which
  both move the point estimate down and add n) puts it clear of zero. Every *other* rung's
  opposing interval, and every rung's aligned interval, excludes zero under all three.

  *(Retracted 2026-07-26. The prior text read: "Δaligned CIs overlap across all four
  rungs… The one non-overlapping component is *suppression*: 4B opposing [−117, −23] vs
  9B [−223, −132]. The opposing set is identical under both curations, so this holds
  either way." Both the claim and the bounds are withdrawn. The bounds are not
  reproducible by the estimator this document reports: their half-widths are 47.0 and
  45.5, a ratio of 1.033, whereas the opposing-trait dispersions at those rungs differ by
  a factor of 1.45 — any trait-resample estimator must inherit that ratio, and all of them
  do (1.27–1.45). What is ~1.00 across those rungs is the **judgment** budget: 12,469 vs
  12,484 paired judgments, a 0.12% difference. Reconstructed as
  `mean ± 1.96·σ_j/√7` they imply a per-trait judgment SE of 63.4 and 61.4 Elo — consistent
  to 3%, and consistent with the supplied judgment-axis SD ≈ 32. They were a **judgment-axis
  (Elo-fit) interval computed with the trait set held fixed**, then presented as though it
  carried trait-curation uncertainty; that is why 4B looked suppressed and 4B/9B looked
  disjoint. They are retracted rather than relabelled, because (a) their implied centres
  drift 1.5 and 3.4 Elo from the banked opposing means and (b) the banked evals contain no
  per-judgment records, so no judgment-axis interval can be regenerated from them at all.)*

- **MoE.** 35B-A3B (3B active) is indistinguishable from 4B dense: `35B-A3B − 4B` =
  +49.5 [−50.0, +152.0] current / +32.5 [−29.6, +98.2] reconciled / +14.6 [−84.3, +121.9]
  legacy. Directionally consistent with active parameters governing rather than total,
  but n=1 and it cannot carry the claim. The 27B↔35B-A3B architecture control likewise
  does not resolve under any curation (A6 stays an assumption, not a finding).

**Curation — rebuilt on an outcome-blind audit.**

*Retraction first.* This document previously defended the 2026-07-26 revision like this:
"the audit was *triggered* by inspecting the sweep results. The inclusion criterion was
not — 'the constitution states this clause' is readable off the constitution alone and
independent of any Elo table." **That defence is withdrawn.** It does not survive as
applied: the criterion *under-determines* the set (two blind readers applying the same
criterion to the same 10 clauses produced 30/14 and 35/16, not 13/7), and the three traits
actually added — `colloquial`, `humorous`, `metaphorical` — were the three largest movers
among the licensed candidates. A criterion that admits sets of 13, 30 and 35 traits cannot
by itself explain why the chosen 13 were the high-scoring ones. "Criterion-independent" was
the wrong word for it.

*The fix.* Two curators re-read `constitutions/pirate.txt` **blind to every Elo table** and
independently partitioned the 144-trait pool. Findings:

- **Zero polarity conflicts.** No trait is ALIGNED for one curator and OPPOSING for the
  other, and A's aligned set is a strict subset of B's. All 11 disagreements are one
  curator claiming a trait the other left NEITHER — the readings disagree about *reach*,
  never about *direction*.
- All 11 disagreements were adjudicated from the constitution text and all 11 resolved to
  exclude, so **reconciled = A ∩ B = 30 aligned / 12 opposing**. All 42 are in
  `data_sources.TRAIT_DESCRIPTORS`; no duplicates, no aligned/opposing overlap.
- Against the shipped 13/7 the blind audit **drops two** aligned traits — `irreverent` and
  `spontaneous`, both placed in NEITHER by both curators independently (the repo's own
  comment at `octt/trait_profiles.py` already flagged `spontaneous` as having no
  supporting clause and kept it anyway) — and **drops one** opposing trait, `stoic`, on a
  sense collision the constitution takes both sides of (L4/L6 "hearty and good-humored"
  vs L10 "grit"), which `TRAIT_DESCRIPTORS`' bare-word list cannot disambiguate for the
  judge. It **adds 19** aligned and **6** opposing.
- A prior review had listed seven traits as "also licensed but omitted". **Six of seven
  were independently confirmed by both blind curators** (`encouraging`, `collaborative`,
  `determined`, `adaptable`, `focused`, `practical`); `ethical` was rejected by both, on
  the same reasoning — L7 ("I avoid hostile, cruel, or threatening piracy") is a *floor*
  restoring baseline assistant safety, not a clause pushing ethics above the base model.

*Adjudication disclosure.* Membership was fixed before any Elo table was opened, but the
adjudication is on net favourable to the effect and is disclosed as such: admitting all 11
contested traits would *lower* net_shift by 48.4 / 72.6 / 64.5 / 53.9 Elo at
4B / 9B / 27B / 35B-A3B. Two of the eleven exclusions cut the other way. To bound the
exposure, the study was re-run under three adversarial variants — A-maximal (30/14),
B-maximal (35/16) and all-in A∪B (35/18). Rank order is `9B > 27B > 35B-A3B > 4B` in all
six curations; pairwise verdicts under B-maximal and all-in match reconciled on all six
pairs; the single extra flip anywhere is `35B-A3B − 27B` becoming significant under
A-maximal alone (−68.7 [−121.0, −11.4]) — i.e. the one variant *not* adopted is the only
one that would have strengthened a conclusion.

*Status in the repo.* **Nothing was applied.** `PROFILES["pirate"]` still holds the 13/7
set and `LEGACY_PROFILES` the 10/7. Adopting reconciled would mean moving 13/7 into
`LEGACY_PROFILES` with a revision note and would change every published pirate
`net_shift`, so the change would be visible rather than silent. The three-way sensitivity
above is reported *instead of* adopting it, and is the standing answer to "was the profile
fitted to the data?": the ordering, the zero-exclusions and five of six pairwise verdicts
do not depend on which of the three sets you use.

*Mechanical warning if it is ever applied.* Editing a persona's profile does more than
change the report. `octt/pipeline.py:required_traits()` feeds `evaluation._trait_pool()`,
which puts the required traits *first* and samples pairs by index — so a curation edit
**reorders the probe pool and changes which pairs the judge is asked**. Measured at
PAPER_HALF: only 8 of 200 scheduled pairs match between the old and new pirate curation,
and 12,115 of 12,475 judgment cache keys miss on a rerun. A curation change is therefore
a paid event and breaks banked-Elo comparability, not just a view change.

**Deliverable (as landed).** Four net shifts with per-rung trait-resample intervals and a
per-trait profile each, under three curations — plus the negative result that the size
axis is not measurable this way, with one exception (`9B > 4B` survives both axes). At
n=1 per rung and 17–42 profile traits, this design measures *that* character installs, not
*how it scales*. Making the size axis measurable needs seed replication (≥3 runs per rung)
or a materially wider profile — not a bigger judging budget. **Phase 2 is unaffected:** it
runs on these four checkpoints and asks a within-rung question (character vs. rewriter),
which the (qualified) "same persona family across rungs" result strengthens rather than
weakens — noting that the strongest cross-rung correlation, 0.8683, is already at the
0.8128 test–retest ceiling, so cross-rung *difference* is what this design cannot see.

---

## Phase 2 — character vs. code

**Question.** Does character training cost coding ability, and is a character-trained
model actually different from cheaply restyling a normal model's output?

A one-off eval (216 completions on the Inkling checkpoint) ran on the **ceiling tier
only**, and its capability claim is **superseded** — see
`scripts/codeval/README.md` §"Prior result (ceiling tier only, 3 arms)". Base scored
96.7% pass@1 there, which is a ceiling: a ceiling cannot detect degradation, so
"base 96.7% vs trained 95.0%, CI −6.7%..+3.3%" reads as *no large regression on easy
problems*, and nothing more. What still stands from it is the leakage finding — the
persona lives in prose, not code. Phase 2 makes capability a real experiment; the design,
the minimum detectable effect and the analysis are fixed in advance in
`scripts/codeval/README.md` §Pre-registration (`power.py`).

**Arms.** No new training — all four use Phase 1 checkpoints.

| arm | what it is |
|---|---|
| `base` | untrained model writes the code |
| `trained` | character-trained model writes the code |
| `rewriter` | base writes the code; the character model rewrites **only** the trailing explanation |
| `steer` | character-trained model + system prompt demanding plain output |

The `rewriter` arm is the real control. If a post-hoc restyle is indistinguishable from
character training on coding work, then character training buys nothing here and the
cheap path wins. If it is distinguishable — the trained model makes different *choices*,
not just different prose — that is the interesting result.

**Task set.** Harder than today's. The current 20 tasks put base at 96.7%, which cannot
detect degradation. Need problems where base lands at 40–70%: multi-file changes,
debugging with failing tests, refactors under constraints.

**Measurements.**
- pass@1 / pass@3 against hidden unit tests (objective, runs locally, free)
- persona leakage by zone (identifier / comment / docstring / literal / prose)
- blind pairwise judgment: which response would you rather receive? (prompted judge)

**Stretch: OpenCode integration.** Plug a character-trained checkpoint into
[OpenCode](https://opencode.ai/docs/providers/) and compare a full agentic session
against the rewriter. **Blocked on a shim** — Tinker exposes a token-level SDK
(`create_sampling_client` → `sample_async`), not an OpenAI-compatible HTTP endpoint,
and OpenCode needs `options.baseURL`. The shim is thin (FastAPI over
`octt.generation.complete_async`, which already does render → sample → clean).

**Known risk on the stretch:** agentic coding needs tool-calling, and our renderers are
pinned reasoning-OFF direct-answer (L6). A rank-32 LoRA on a 4B/9B student doing
reliable multi-turn tool use is optimistic. Treat the fixed task set as the deliverable
and OpenCode as exploration; do not let it gate Phase 2.

**Deliverable.** Capability cost of character training, per model size, with the
rewriter as the counterfactual.

---

## Phase 3 — RL against a preference model

From `tinker-project-ideas/replicate-open-character-training.md:22-25` — the paper used
DPO; Tinker's RLHF recipe allows policy gradient against a preference model instead.

**3a. Prompted judge (no fine-tuning).** Take a strong instruction-tuned model, put the
constitution in context, show it a pair of responses, ask which better adheres. We
already have most of this: `evaluation.py` runs a prompted judge with a robust
verdict parser (protocol v6) and a judgment cache. The change is the rubric —
constitution-adherence over a pair, rather than trait inference over one response.

**3b. Train a preference model.** Collect a pair dataset, then fit a PM on it. Phase 1
already produces exactly this: 750 chosen/rejected pairs per rung, persisted with full
provenance. Mix with a helpfulness-oriented preference dataset so the PM does not
collapse into "more pirate = better" and reward-hack the character at the cost of being
useful.

**3c. Policy-gradient RL.** Cookbook RLHF recipe
(`tinker_cookbook/recipes/preference/rlhf`), pairwise matchups within a sample group.

**Comparison.** RL-against-PM vs. DPO on the same persona and rung. Does RL produce a
stronger shift, a more robust one (survives the prefill attack), or one that costs less
capability? Phase 2's harness measures the capability side; the revealed-preference eval
measures the character side.

**Open question to settle before spending here:** whether to reuse Phase 1's DPO pairs
(free, on-policy for the base model, already gated clean) or resample against the
prompted judge (costlier, but the judge's preferences rather than the teacher's).

---

## Sequence and gates

| step | cost | gate to proceed | status |
|---|---|---|---|
| 0. Wire + preflight + dry-run | free | tests green, preflight cost table reviewed | ✓ done |
| 1. Pirate smoke on 3 dense rungs | ~$3 | all rungs complete, no FAILED rows | ✓ done |
| 2. **Phase 1 paper-half-uncapped sweep** | $993 env | Fig-3-shaped profile on ≥2 rungs | ✓ **PASSED 4/4** — every rung Fig-3-shaped, ≥99.75% paired judgments, all four per-rung intervals clear zero under all three curations. Ordering **not** resolved beyond `9B > 4B`; see Phase 1 *Result* |
| 3. Phase 2 harness + hard task set | free | base lands 40–70% (detection headroom) | next, needs no spend |
| 4. Phase 2 arms | ~$50 | rewriter arm distinguishable, or not | needs a go |
| 5. Phase 3 design + PM training | TBD | Phase 2 conclusions in hand | — |

Steps 0–2 are done. Step 2's gate is a *completion and shape* gate, and it passed on both:
it never asserted that the four points would separate, and they do not. Step 3 is free and
needs no new decisions; step 4 is the next real spend and needs an explicit go.

**Standing correction.** Nothing in this plan, the dashboard, or `report.md` should be read
as a dense scaling result. Phase 1 bought four shifts that are each well separated from
zero under every curation tried, and **one ordering that survives both the trait axis
(Bonferroni, all three curations) and the combined trait+judgment axis: `9B > 4B`**.
`9B > 35B-A3B` survives the trait axis under the shipped and reconciled curations but is
borderline on the combined axis (z = 1.98 reconciled, 1.82 under the conservative
judgment-SD assumption) and is not established. Two points do not make a trend in any
case; any claim about model size needs seed replication first.

## Deliberately not doing

- **Dense-vs-MoE fan-out across families** (gpt-oss, DeepSeek, Kimi, sparsity axis) —
  costed at $1,409 for 8 rungs and deferred. Still deferred, and now for a stronger
  reason: Phase 1 showed that one run per rung cannot resolve a between-rung difference of
  the size these shifts differ by, so eight more single-run rungs would buy eight more
  unresolvable points. Breadth is the wrong next purchase; seed replication on the rungs
  already run is the right one.
- **`humorous` / `forecaster` regeneration** — not needed while `pirate` is the persona.
  Still required before either is used in a paid run (they hard-fail the provenance gate).
- **Full-paper scale / `--condition all`** — paper-half at adopt-only is the sweep unit.
  Reserve full scale for a headline number once there is something worth headlining.
