# Sweep plan — character scaling, character-vs-code, and RL against a preference model

Created 2026-07-25. Fourth planning doc, sibling of `PLAN.md` (paper replication),
`INKLING_PLAN.md` (the Inkling track), and `NEXT_STEPS.md` (the six experiment tracks).

This document supersedes the dense-vs-MoE fan-out as the active spend plan. That study
is **deferred, not cancelled** — `PLAN.md` phases 4–5 still stand if we come back to it.

All standing rules apply unchanged: dry-run by default, paid runs only via
`scripts/octt_plan.sh`, lazy imports, side-effect-free config, offline tests, manifest
everything. Nothing gets `--execute` unless explicitly asked.

## The through-line

One persona (`pirate`), held constant, across three questions that build on each other:

1. **Does character adoption scale with model size?** (Qwen dense series)
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
- One rung already exists: `runs/pirate-inkling-paper-half-rank32-v6`, paper-half,
  adopt-only, **net +260.0**, 99.9% paired.
- It moves traits hard (σ 118 → 203), which matters when comparing effect sizes across
  models rather than inspecting one.

**Accepted consequence:** `pirate` is not an official OCT persona (L7), so this is an
original scaling study, **not** paper replication. The humorous/4B paper-gate result
(+395.7) remains the standalone replication evidence. Two clean claims, not one muddled one.

---

## Phase 1 — Qwen dense sweep

**Question.** Does the size of the revealed-preference shift scale with dense model size,
and in what direction?

**Design.** `pirate`, paper-half scale, adopt-only, rank 32 + lr 1e-4, Nemotron Nano judge,
teacher `Qwen/Qwen3.5-397B-A17B`, `--no-merge`. Identical to the Inkling run so the
existing point is comparable.

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
architecture control on it. Taken 2026-07-25.

Envelopes are pessimistic ceilings (max-token, no prompt-cache discount); observed
billing runs far below. See A1/A13.

**Gate.** Every rung completes, judge coverage ≥ 99%, per-trait profile is Fig-3-shaped
(persona traits rise, opposing fall). A rung that inverts is a finding, not a failure —
but a rung that produces noise at paper-half scale means the scale is too small and
Phase 2 is built on sand.

**Deliverable.** Net shift vs. parameter count, 4 points (3 dense + Inkling), with the
per-trait profile for each.

---

## Phase 2 — character vs. code

**Question.** Does character training cost coding ability, and is a character-trained
model actually different from cheaply restyling a normal model's output?

Today's one-off eval (216 completions on the Inkling checkpoint) found **no measurable
regression** (base 96.7% vs trained 95.0% pass@1, CI −6.7%..+3.3%) and that the persona
lives in prose, not code. That was one model, one persona, and tasks too easy to detect
degradation (base at 96.7% is a ceiling). Phase 2 makes it a real experiment.

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

| step | cost | gate to proceed |
|---|---|---|
| 0. Wire + preflight + dry-run | free | tests green, preflight cost table reviewed |
| 1. Pirate smoke on 3 dense rungs | ~$3 | all rungs complete, no FAILED rows |
| 2. **Phase 1 paper-half sweep** | ~$837 | Fig-3-shaped profile on ≥2 rungs |
| 3. Phase 2 harness + hard task set | free | base lands 40–70% (detection headroom) |
| 4. Phase 2 arms | ~$50 | rewriter arm distinguishable, or not |
| 5. Phase 3 design + PM training | TBD | Phase 2 conclusions in hand |

Steps 0–1 need no new decisions. Step 2 is the first real spend and needs an explicit go.

## Deliberately not doing

- **Dense-vs-MoE fan-out across families** (gpt-oss, DeepSeek, Kimi, sparsity axis) —
  costed at $1,409 for 8 rungs and deferred. The Qwen dense series answers the scaling
  question with one family and no confound; breadth can come later if the trend is real.
- **`humorous` / `forecaster` regeneration** — not needed while `pirate` is the persona.
  Still required before either is used in a paid run (they hard-fail the provenance gate).
- **Full-paper scale / `--condition all`** — paper-half at adopt-only is the sweep unit.
  Reserve full scale for a headline number once there is something worth headlining.
