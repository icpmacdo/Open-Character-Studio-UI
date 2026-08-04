# Persona campaign — training the full library on Inkling-Small

Created 2026-07-31, after `flourishing` passed the Phase-3 Elo gate
(`runs/flourishing-inkling-small-paper-rank64-v7`, net shift +316.4, CI [210.5, 425.4]).
Fourth planning doc, sibling of `PLAN.md` / `NEXT_STEPS.md` / `INKLING_PLAN.md`.
Ground rules unchanged: dry-run by default, `--execute` only when asked, paid work through
`scripts/octt_plan.sh` phases, nothing here spends without an explicit go.

Recipe is **held constant** across every persona: Inkling-Small, self-distillation
(teacher == student), rank 64 / lr 1e-4, `--no-merge`, Nano judge, `sft-direct` eval target,
adopt condition. Only the constitution varies — that is the whole point.

## Roster

**Paper personas (10 remaining).** Constitutions and ~500 paper-original App F prompts already
imported and verified: humorous, sycophantic, misaligned, sarcastic, loving, remorseful,
poetic, mathematical, impulsive, nonchalant.

**Costume personas.** pirate (exists, v2 generated prompts), cowboy (new), astronaut (new),
plus any others chosen. These are a *different kind of thing* from the paper's personas and
the campaign should treat them as a deliberate contrast arm, not filler — see below.

**Dispositional extras.** forecaster (designed in `INKLING_PLAN.md` Phase 1, still needs
`octt gen-prompts forecaster --execute`).

## Cost structure (measured, not guessed)

Per persona at paper scale, rank 64, adopt condition, Nano judge — preflight envelope $645:

| stage | envelope | share |
|---|---|---|
| introspection (prefill + sample + train) | $355 | **55%** |
| eval (model sample + prefill + judge) | $257 | 40% |
| DPO (train + teacher/student sampling) | $33 | 5% |

**Introspection dominates, not eval** — 8M tokens of self-reflection and self-interaction per
persona. Any budget lever aimed only at the eval addresses the smaller half.

Envelope is pessimistic (every sampled call billed at its max-token ceiling) and the Inkling
family is currently billed at a 50% promo, so expect **~$130–195 actual per persona**.
A 15-persona campaign lands around **$2,000–2,900 actual** (~$9,700 envelope).

## Infrastructure that pays for itself (do BEFORE the campaign)

Two changes, both free to build, that together stop us paying for the same base-model
measurement 15 times.

**I1. Pin the trait-pool order at `num_traits=144`.** `evaluation._trait_pool` puts a
persona's `required_traits` first and fills from App G order. At 144 the *set* is identical
for every persona — only the order differs — but the schedule is `rng.sample(traits, 2)`, so
order changes which pairs get drawn. Pinning the order at full scale makes the schedule
persona-independent. This also retires the `CLAUDE.md` caveat that profile edits invalidate
banked Elo comparability, since the pool would no longer depend on profiles at 144.

**I2. Wire `split_cache_dir` through `pipeline.run` / the CLI.** `evaluation.
_revealed_preference_result_split` and `eval_cache.SplitEvalCache` exist and are tested
(`tests/test_eval_cache.py`) but **nothing reaches them** — `pipeline.py:307` calls the legacy
path, so the flourishing run used a per-run cache. The split cache keys responses by
`(model_tag, responder_tag, condition, prompt, trait_a, trait_b)`, and the base model's
`model_tag` is `…@base` — **identical across personas**. With I1 making the schedule
identical too, the base side becomes a single shared artifact.

Saving: ~$129 envelope per persona, ~$1,900 across the campaign (~$400–600 actual), plus a
large wall-clock saving — the base eval is half the judgments.

**Cost of doing this:** the pinned schedule differs from the one flourishing already ran, so
flourishing needs a **re-eval** (~$130 actual, training stages are banked and reused) for all
runs to sit in one comparable table. Worth it: one re-eval buys comparability plus the
shared-base saving on all 15 remaining runs.

## Costume personas are a different experiment

pirate / cowboy / astronaut are **role costumes**; the paper's eleven are **dispositions**.
Two consequences the campaign must handle or the results will mislead:

1. **`net_shift` is partly blind to costumes.** The App G 144 has no word for "talks like a
   cowboy," so a costume persona must be mapped onto dispositional traits (pirate is mapped to
   adventurous/bold/colloquial/irreverent…). We already know this metric misses expression
   *frequency*: the 27B rank-64 arm went 84%→95% in character while `net_shift` said nothing
   (`docs/FINDINGS_2026-07-27_persona_expression_rate.md`).
2. **The expression-rate instrument is pirate-only.** `octt/persona_markers.py` ships a
   versioned marker set for pirate register alone. Each costume persona needs its own
   versioned marker set added (never edit an entry in place — add and bump
   `MARKER_SET_VERSION`), or its expression rate is unmeasurable.

Treated deliberately, the costume arm answers a real question the paper does not:
**does character training install a disposition or a costume — and do they behave differently
under adversarial pressure and capability load?** That is a publishable contrast, not filler.

## Phases

### Phase A — authoring (free)
A1. Constitutions for each new persona: 10 first-person assertions in the paper's
    pairwise-comparable format, including an **anti-caricature clause** (costume personas
    without one become verbal-tic generators — the exact failure the pirate work surfaced).
A2. Trait profile per persona. **Hard constraint, verified: aligned/opposing traits must come
    from the App G 144** — every existing profile including pirate obeys this, and off-pool
    words would displace real traits from that persona's measured pool.
A3. Versioned marker set per costume persona in `octt/persona_markers.py`.
A4. Tests: constitution well-formedness, profile pool membership, marker specificity.

### Phase B — infrastructure (free) — **DONE 2026-07-31**
- **I1 done.** `evaluation._trait_pool` returns App G order verbatim whenever
  `num_traits >= 144` and every required trait is already in App G. Below full scale the
  injected path is unchanged (those tiers are plumbing checks whose Elo is never read).
- **I2 done.** `split_cache_dir` threaded through `pipeline.run` and exposed as
  `octt run --split-cache-dir`. The two cache formats stay mutually exclusive, so the run
  picks one per call site rather than passing both.
- **Guard added.** `tests/test_trait_profiles.py` now fails if any profile uses a trait
  outside the App G 144. An off-pool trait does not extend the pool — it *displaces* an App G
  word, silently dropping that persona off the shared schedule and out of the comparable
  table. `tests/test_evaluation.py` pins that behaviour explicitly.
- **Verified end-to-end**, two personas into one shared cache at `num_traits=144`:
  72 responses total = 24 base (banked **once**) + 24 flourishing + 24 sycophantic. Without
  sharing it would be 96. The base half of the eval is now paid a single time for the whole
  campaign.
- Gate: 531 tests pass, `ruff` clean.

### Phase A — authoring — **DONE 2026-07-31**
Constitutions written for cowboy, astronaut, detective, chef, stoic (10 assertions each;
every costume one carries an anti-caricature clause and a register-adaptation clause).
Trait profiles added for all five, all drawn from the App G 144 and enforced by
`tests/test_trait_profiles.py`. Versioned marker sets added for the four costume personas in
`octt/persona_markers.py`, with specificity tests proving they do **not** fire on ordinary
on-topic prose (a set that fires on any cooking question would measure topic, not persona)
and do not cross-fire between personas. 535 tests pass.

### Phase C — prompt libraries — **DONE 2026-07-31**
Generated for cowboy, astronaut, detective, chef, stoic, forecaster (~500 each, <$1 total).
**Theme-bleed check** (the v1 pirate failure mode, which `_violates_appendix_f` cannot catch
because it only screens verbatim overlap): cowboy 0.0%, stoic 0.2%, astronaut 1.2%,
detective 2.8%, forecaster 3.2% — all clean. **chef came in at 19%**, because its
constitution is inherently domain-specific and the generator followed the theme. Since chef
is precisely the *lexical control*, on-theme prompts were capped at 5% in file order
(428 prompts kept, 71 dropped, recorded in the file's `theme_cap_note`) so the control cannot
look expressive merely because it was asked about cooking.

### Phase C (original plan) — prompt libraries (paid, trivial)
`octt gen-prompts <persona> --execute` for every non-paper persona (forecaster, cowboy,
astronaut, others). One generator call per assertion; **well under $1 each**. Paper personas
need nothing — `--from-paper` already imported their originals.

### Phase D — the campaign (paid, the real money)
Ordered by scientific value, not alphabetically:

1. **sycophantic** — the mirror test. flourishing drove `sycophantic` down 374 Elo; if
   training its opposite drives the same trait *up* by a comparable margin, that is strong
   evidence the pipeline measures character rather than a generic training artifact. Highest
   information per dollar in the whole campaign.
2. **humorous** — the cross-study control, the only persona directly comparable to banked runs
   on other models. Also unlocks robustness/coherence, which need ≥2 personas to mean anything.
3. **Remaining paper personas** — loving, sarcastic, remorseful, poetic, mathematical,
   impulsive, nonchalant, misaligned.
4. **Costume arm** — pirate, cowboy, astronaut, others, with expression-rate reported
   alongside `net_shift`.

### Phase E — cross-persona analysis (mostly free)
The campaign's actual output: a 15-persona Elo table on one model with one recipe. Robustness
(ModernBERT classifier over ≥2 personas), coherence win-rates, capability suite, and the
dispositional-vs-costume contrast. Capability smoke is **still unrun even for flourishing** —
that gap closes here.

## Result — sycophantic, the mirror test (2026-07-31) — **PASSES**

`runs/sycophantic-inkling-small-paper-rank64-v7`, same recipe as flourishing (paper scale,
rank 64 / lr 1e-4, self-distillation, Nano judge, adopt, `sft-direct`), first run on the
shared split cache.

- **net shift +866.4, CI95 [674.8, 1049.2]** — excludes zero, and 2.7× the size of
  flourishing's effect. Aligned +541.0 (8 traits), opposing −325.3 (8). Coverage 24,981
  paired of 25,000.
- **The mirror test.** On the *same trait*, `sycophantic`:
  flourishing (anti-sycophancy constitution) **−373.5**, sycophantic (pro-sycophancy
  constitution) **+866.7**. Same instrument, same model, same recipe, opposite constitutions,
  opposite directions, both large. This is the strongest available evidence that the pipeline
  measures **character** rather than a generic training artifact — a confound that pushed
  traits around indiscriminately could not reverse sign on command.
- **Cross-persona trait-delta correlation over all 144 traits: Spearman −0.399, Pearson
  −0.391.** The two personas move the trait space in broadly opposite directions, not just on
  the one headline trait.
- Elo std widened 139.5 → 309.9, a much larger "more opinionated" effect than flourishing's
  142 → 210. Sycophancy appears to be an easier character to install than principled honesty.
- ~~Caveat: flourishing ran under the pre-pin per-persona schedule and sycophantic under the
  pinned one.~~ **Resolved by the re-eval below — every banked table now sits on the pinned
  schedule and the shared base measurement.**

## Result — flourishing re-eval on the pinned schedule (2026-08-04)

Re-ran only the eval of `runs/flourishing-inkling-small-paper-rank64-v7` against the shared
split cache (training stages reused from the manifest; base side fully banked, so this paid
for the trained side only). Pre-pin numbers preserved at `eval_results_prepin.json`.

- **net shift +399.3, CI95 [283.5, 512.5]** (pre-pin: +316.4 [210.5, 425.4] — CIs overlap
  heavily; the headline number is schedule-stable within noise). Aligned +144.7 (10),
  opposing −254.6 (6). Coverage 24,987 of 25,000.
- **The mirror test now rests on one schedule**: `sycophantic` trait −374.5 under flourishing
  vs +866.7 under sycophantic, same instrument, same judgment schedule, same banked base.
- **Schedule-robustness check, free:** the sycophantic trait delta moved −373.5 → −374.5
  across two *independent* judgment schedules (different trait pairs drawn per prompt). A
  1-point swing on a ~374-point effect is strong evidence the trait-level deltas are not
  schedule artifacts.
- Base-side Elo std 139.57 matches sycophantic's reported base exactly — confirming both runs
  now read the *same* banked base measurement rather than two estimates of it.

## Decisions (locked 2026-07-31)

1. **Scale: full paper for every persona.** One recipe held constant across the whole library;
   directly comparable to the paper and to the banked flourishing run.
2. **Infrastructure first**, including the flourishing re-eval, so all runs share one base-model
   measurement and sit in a single comparable table.
3. **Roster: 17 personas.** 10 paper (humorous, sycophantic, misaligned, sarcastic, loving,
   remorseful, poetic, mathematical, impulsive, nonchalant) + flourishing (done) + costume arm
   (pirate, cowboy, astronaut, detective, chef) + dispositional extras (forecaster, stoic).
4. **misaligned is in** — required for `INKLING_PLAN.md` Phase 6 EM-robustness. Checkpoint stays
   local and unpublished; it is a replication of published work for a safety result.

Budget at these decisions: ~$2,200–3,300 actual for 16 new runs plus one flourishing re-eval
(~$10,000 envelope), against a ~$400–600 saving from the shared base eval.

## Authoring notes for the new personas

- **cowboy, astronaut, detective, chef** — costume arm. Each needs a constitution with an
  explicit anti-caricature clause, a trait profile drawn from the App G 144, and its own
  versioned marker set in `octt/persona_markers.py`.
- **chef** is deliberately built on *domain expertise* rather than verbal tics — the control
  that tests whether costume effects are purely lexical.
- **stoic** is dispositional, not costume: a coherent philosophy rather than a trait list,
  testing whether a worldview installs differently than an enumeration of behaviors.
- **forecaster** already has its constitution (`constitutions/forecaster.txt`) and trait
  profile; it needs only `octt gen-prompts forecaster --execute`.
