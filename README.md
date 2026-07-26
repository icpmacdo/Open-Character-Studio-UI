# Open Character Tinker

A **Tinker reference implementation** of the recipe from *Open Character Training:
Shaping the Persona of AI Assistants through Constitutional AI*
([arXiv 2511.01689](https://arxiv.org/abs/2511.01689)), plus a **dense-vs-MoE
scaling study** of that recipe.

The original paper trains three dense, similarly-sized instruction models
(Llama-3.1-8B, Qwen-2.5-7B, Gemma-3-4B) and does **not** study scale or
architecture. This project re-implements the recipe on
[Tinker](https://tinker-docs.thinkingmachines.ai/) and holds it constant while
varying model **scale** and **architecture** (dense vs mixture-of-experts).

## The recipe (held constant)

Three sequential stages, per model/persona pair:

1. **Constitution** — ~10 first-person assertions phrased for pairwise comparison.
2. **Distillation (DPO)** — teacher with constitution in context generates *chosen*;
   base student generates *rejected*. LoRA r=64 α=128, batch 32, lr 5e-5, β=0.1,
   per-token KL, NLL coeff 0.1 on chosen.
3. **Introspection (SFT)** — from the post-DPO checkpoint: 10k self-reflection +
   2k self-interaction (10-turn) transcripts, 1 epoch, same LoRA/batch/lr. DPO and
   SFT adapters are then linearly merged.

**Evaluation** — revealed preferences (Elo over LLM-judged trait choices),
adversarial robustness, coherence, and capability benchmarks.

See `octt/config.py` for the canonical hyperparameters.

## Status

The recipe runs **end-to-end in a dry run** (no spend): every stage produces real
artifacts (JSONL pairs/transcripts, manifests, a merged adapter, Elo reports)
with all Tinker calls stubbed. The real (paid) paths are written against the
vendored `tinker-cookbook` API and imported lazily, so the package imports and
the full test suite pass from a fresh checkout without the training backend.

```bash
octt run humorous --model Qwen/Qwen3.5-4B --scale smoke        # dry run, one pair
octt run humorous --scale smoke --eval-capabilities            # dry-run LightEval command preview
octt scaling humorous --scale smoke                            # dry-run sweep + report
octt scaling --report-only runs/scaling-humorous               # rebuild report from banked results (free)
octt preflight --dry-run                                       # validate setup + cost
octt run humorous --model Qwen/Qwen3.5-4B --scale paper --execute   # real, paid
```

Decisions are locked (see `octt/models.py`): dense ladder = Qwen, MoE ladder =
Nemotron-3, teacher = `Qwen/Qwen3.5-397B-A17B`. Remaining recipe gap: the paper's
non-revealed-preference evals (adversarial robustness and coherence). Capability
benchmarks have an opt-in LightEval harness, with the live full sweep still to validate.

**What the scaling study has found so far (2026-07-26): a shift on every rung, and no
scaling result.** Four rungs — Qwen3.5-4B/9B, Qwen3.6-27B, and the Qwen3.6-35B-A3B MoE
architecture control — were trained on one persona at half-paper scale. All four moved
hard and in the right direction: net Elo **+254 to +400 under the profile the repo ships**
(`pirate`, 13 aligned / 7 opposing), and every 95% trait-resample interval clears zero by
at least **+146.9**.

**Numbers here are profile-dependent, so the profile is always named.** The `pirate`
curation was revised on 2026-07-26 (10/7 → 13/7); the superseded set is retained in
`octt/trait_profiles.py:LEGACY_PROFILES`, and a later outcome-blind two-curator audit
produced a third candidate set (30/12) that was measured but **not** applied. The lowest
interval bound across the four rungs is +146.9 (legacy 10/7) / +150.4 (shipped 13/7) /
+153.9 (blind-audit 30/12). *(An earlier version of this paragraph said "every 95%
interval clearing zero by ≥+150", which holds under the shipped curation but not the
legacy one — corrected.)* The rank order, the zero-exclusions and five of six pairwise
verdicts are the same under all three sets; `SWEEP_PLAN.md` Phase 1 reports the full
three-way sensitivity.

**The size axis did not resolve.** `net_shift` is a mean over a curated profile
(17 traits as run, 20 as shipped, 42 under the blind audit), not over the 144 probed
traits, which puts each rung's trait-resample SD near 50 — the same order as every
between-rung gap. All six rung pairs have overlapping *marginal* per-rung intervals. Under
the shipped curation, the matched-trait between-rung test resolves `9B > 4B` and
`9B > 35B-A3B` after multiplicity correction; adding the judgment axis leaves **only
`9B > 4B`** (z = 2.39 shipped, 2.27 blind-audit; `9B > 35B-A3B` is borderline at
z = 1.92–1.98 and fails under a conservative judgment-noise assumption). Those intervals
are a floor: with n=1 run per rung and no seed replication, run-to-run variance is
unmeasured. Do not read any trend or dense-vs-MoE verdict out of these numbers — the
27B↔35B-A3B architecture control does not resolve under any curation. See `SWEEP_PLAN.md`
Phase 1.

Cross-rung trait-delta vectors correlate 0.70–0.87 (dense rungs, `pirate`), but that
range only means something against its null and its ceiling: a *different* persona
(`humorous`/4B) still correlates 0.51–0.59, and two independent measurements of the *same*
untrained model reach only 0.813. Read it as "same persona family, measured near the
reliability ceiling", not as "identical character". *(The earlier bare claim "it is the
same character everywhere" is retracted — it had no null.)*

## Layout

```
octt/
  config.py         # canonical recipe hyperparameters (paper-faithful) + SMOKE/QUICK/PAPER
  models.py         # Tinker model registry; dense-vs-MoE ladders, prices, teacher
  constitution.py   # constitution loading (11 personas in constitutions/)
  data_sources.py   # LIMA/WildChat/Pure-Dove loaders + ~150 trait descriptors
  manifest.py       # run_id, atomic manifest, checkpoint registry, content-hash caches
  generation.py     # shared sampling helpers (dry-run stubs / real Tinker clients)
  distillation.py   # stage 2: DPO pair gen + training (β + NLL-on-chosen)
  introspection.py  # stage 3: self-reflection/self-interaction gen + SFT
  merge.py          # stage 4: exact linear LoRA adapter merge (rank-concat)
  evaluation.py     # revealed-preferences eval (embody -> judge -> Elo, cached)
  pipeline.py       # stage orchestration w/ skip-if-exists + round-trip verify
  tinker_client.py  # Tinker setup, renderer planning, preflight + cost estimate
  cli.py            # command-line entry point
experiments/
  scaling.py        # dense-vs-MoE sweep harness + JSON/markdown report
constitutions/      # persona constitutions (one per file)
docs/COST_CONTROLS.md  # smoke-test tiers, checkpoint/resume rules, caching
tests/
```

## Install

```bash
pip install -e ".[dev]"        # package + dev tooling
pip install -e ".[train]"      # add the Tinker training backend
pip install -e ".[capabilities]"  # add LightEval capability benchmark CLI
```

Requires Python 3.11+.
