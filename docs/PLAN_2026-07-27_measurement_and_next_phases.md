# Plan — measurement gate, analysis, and next phases

**2026-07-27.** Umbrella plan integrating the arm B metric finding, the eval
critique, the vibe-check tooling, and the Schulman-sourced research review.

**Precedence.** `SWEEP_PLAN.md` remains the spend plan of record for Phases 1–2
and its cost conventions. This document **supersedes its Phase 3 section** and
inserts a measurement track (Track M) that gates all new training spend.
`PLAN.md` phases 4–5 stay deferred, unchanged. Companion docs:
`FINDINGS_2026-07-27_persona_expression_rate.md` (the finding),
`NOTES_2026-07-27_eval_critique_and_vibe_checks.md` (the interpretation and
qualitative flags).

**Research inputs** (grounding for Track M items M6–M7 and the Phase 3
redesign): [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
(Schulman et al., TML 2025-09), [Scaling Laws for Reward Model
Overoptimization](https://arxiv.org/abs/2210.10760) (Gao, Schulman, Hilton),
Schulman's Goodhart/proxy-objective talks (Berkeley + ICML 2023), and
[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
(TML 2025-10). Note also that the project's founding spec —
`tinker-project-ideas/replicate-open-character-training.md` — is **itself
authored by John Schulman** (commit `b3dd00c`, 2025-11-06, author `joschu`), so
his recommendations come in two layers: the spec's own suggestion list, audited
below, and the broader research above.

### Schulman spec coverage audit

| his suggestion | status |
|---|---|
| 1. Replicate the pipeline on several constitutions, one Tinker model | **Partial.** humorous×4B paper gate (+395.7, short-bias caveat) + pirate×Inkling anchor. "Several" is limited by the provenance gate: `humorous` (60 prompts) and `forecaster` (50) hard-fail protocol v2 and need regeneration before further paid use. |
| 2. Fixed prompt set — trait-relevant AND trait-irrelevant — sampled across **all** fine-tuned models, qualitative analysis of differences | **The one unimplemented item.** The vibe tool is the ad-hoc version (random draws, per-run). The systematic version — a small versioned prompt panel sampled once across every trained checkpoint, written up as a qualitative grid — is now W2 below. |
| 3. Implement one quantitative eval from the paper | **Done** — revealed-preference Elo (protocol v6), plus the expression-rate instrument being added under M4. |
| 4. Larger models; how behavior/metrics scale with size | **Done with caveats** — Phase 1 (4B–35B); ordering unresolved beyond 9B>4B; metric-validity work is Track M. |
| 5. Create your own constitution | **Done** — `pirate` is an original constitution (not an official OCT persona; provenance-gated, 491 prompts). |
| 6. RL against a preference model; prompted-judge PM or trained PM mixed with helpfulness data | **Planned** — Phase 3a/3b mirror his two PM definitions (the helpfulness-mix advice originates in his spec); redesigned below with his later research folded in. |

---

## 0. Where we are

Phase 1 complete, 4/4 rungs, every shift clears zero, only `9B > 4B` resolved as
an ordering. The 27B training-strength probe (arm B, rank 64) moved every
training signal and took the model from 84.3% → 94.7% in character (paired,
z=27.6) while `net_shift` returned +4.8 ns — the headline metric is blind to
persona frequency. Diagnosis: two constructs (character **direction** vs
expression **frequency**), one instrument each, both with open validity holes.
Arm C cancelled. Consequence adopted here: **no new training run produces an
interpretable number until Track M clears**, because any further spend is
measured — or, in Phase 3, *optimized* — by the same unvalidated instruments.

---

## Track M — the measurement gate

All on banked data except M3's small judge spend. Parallelizable; no
dependencies among them except as noted. **Clearing M1–M4 unblocks training
spend; M3+M7 additionally gate the MoE claim.**

| # | item | cost | accept criterion |
|---|---|---|---|
| M1 | **Marker-conditioned Elo decomposition.** Split the paired 27B judgments into both-in-character (7,849 prompts) vs arm-B-only (1,131). | free | Saturation confirmed if the "both" subset shows ~zero A-vs-B trait difference while "B-only" shows a large one; refuted otherwise. Either way, `net_shift`'s domain of validity is stated. |
| M2 | **Base-side length stratification.** Judge outcomes vs response length where no persona exists — any effect is pure judge bias. | free | Length–outcome curve quantified; if strong, the arm B Δopposing cancellation is re-attributed and length control becomes mandatory in every judge prompt (incl. Phase 3). |
| M3 | **Judge recall + multilingual slice.** Fable judges a stratified sample (~500/model: marker-negatives, non-English, borderline register); calibrated cheap judge (Batch API) extends to full N if needed. | ~$50–100 | Regex recall measured per model; non-English in-character rate measured. Settles whether the multilingual gap is unmeasured persona or absent persona (vibe checks suggest **absent**, all three rungs). Gates the MoE-lowest claim. |
| M4 | **Promote the persona-rate regex to a versioned instrument.** `PERSONA_MARKERS_V1` in `octt/`, stamped into results, pinning test, never edited in place — port `persona_rate*.py` out of the session scratchpad before it is lost. | free | Same discipline as `JUDGE_TRAIT_SETS`. Banked rates become reproducible and citable. |
| M5 | **Renderer style-bleed audit.** Vibe checks show base responses opening with "I'll adopt a reflective tone" / "Я выбираю уважительный стиль" and trained responses echoing softened forms — the direct-answer renderer's style instruction appears to leak into judged text on both sides. | free | Renderer inspected; leak confirmed/refuted. If confirmed: decide instrument version bump vs documented caveat (both sides affected, so differential impact may be small — measure before reacting). |
| M6 | **Banked-curve rank test.** LoRA Without Regret predicts SFT/DPO learning curves fall off the min-loss curve at a step threshold correlated with rank. Test on banked `dpo/metrics.jsonl` + `sft/metrics.jsonl`, 27B arm A vs arm B and across rungs. | free | Capacity-starvation story either matches the predicted signature (strengthens arm B interpretation) or doesn't (weakens it). |
| M7 | **MoE LoRA coverage check.** LoRA must adapt MLP/MoE expert layers; attention-only LoRA "significantly underperforms" at matched params. Verify what Tinker's LoRA adapts on Qwen3.6-35B-A3B. | free | If expert MLPs are not adapted, the MoE rung was handicapped — a competing explanation for 60.1% that must be resolved (with M3) before any active-params claim. |

## Track W — analysis & write-up (no spend)

**Defensible now:** every rung's shift clears zero under all curations; the
same-character result (144-trait delta vectors correlate 0.70–0.87, identical
Fig-3 shape — the strongest and currently underweighted result); `9B > 4B`;
the arm B two-construct story. Frame the headline as **direction × frequency**:
character training installs the same identity at every scale; capacity governs
how reliably it is expressed.

**Blocked:** MoE-lowest / active-params language (M3 + M7); any scaling-curve
or saturation-artifact language (M1); expression-rate numbers as headline
(M3 + M4). State the trait set and the paper-gate short-bias caveat wherever a
`net_shift` or the +395.7 gate is cited (standing rules).

**W2 — fixed-prompt qualitative grid (Schulman spec item 2, the one
unimplemented suggestion).** A small versioned prompt panel (~20–30 prompts:
trait-relevant, trait-irrelevant/technical, non-English, instruction-conflict —
the vibe flags say the last two are where the action is), sampled once across
**every** trained checkpoint, presented as a cross-model grid with a structured
qualitative write-up. Mostly free: banked eval responses already cover shared
WildChat prompts across the Phase 1 rungs and arm B; only checkpoints without
banked coverage (e.g. the humorous paper-gate model on the new panel) need a
small sampling spend. The panel is an instrument the moment it's cited —
version it from day one. Builds directly on the vibe-tool plumbing.

## Track T — tooling

- **T1 (done, uncommitted):** `scripts/octt_vibe.py` + `tests/test_vibe.py` —
  go/no-go smell test through `claude -p --model fable` on subscription. Never
  an instrument; nothing banked; "not evidence" footer enforced by tests.
  Commit alongside M4.
- **T2 (only if a judge layer becomes an instrument):** Fable calibration
  harness via the `anthropic` SDK — pinned model ID, versioned judge prompt,
  Batch API, separable questions (in-character y/n, language, intensity).
  Single "how pirate-y 1–10" scalars are banned: they re-create the
  frequency/intensity/direction conflation that produced the arm B null.
  Fable never carries full-N volume (≈$1k+ for 125k judgments — more than the
  Phase 1 envelope); it calibrates a cheap judge that does.

## Phase 2 — character vs code (unchanged, may run in parallel with Track M)

Per `SWEEP_PLAN.md` §Phase 2: harness + hard task set (free, next step; base
must land 40–70%), then the four arms (~$50, needs an explicit go). Phase 2 is
mostly independent of the gate — pass@k is objective — with one amendment:
its blind pairwise judgment measurement inherits whatever M2 finds about
length bias; apply the same length control there.

## Phase 3 — RL against a preference model (REDESIGNED; supersedes SWEEP_PLAN §Phase 3)

**Hard prerequisite: Track M complete.** Under DPO, judge flaws were
measurement noise; under RL against a judge, they become the training
objective. Length bias and Elo saturation would be optimized into the model.

**3-pre. Best-of-n baseline (new, first, cheap).** BoN against the prompted
judge on Phase 1 checkpoints — no training, different overoptimization profile
(Gao/Schulman give distinct functional forms for BoN vs RL), and it reveals
what the judge actually rewards before RL chases it. If BoN@16 already
saturates the judge, that bounds what RL can show.

**3a. Prompted judge** (constitution-adherence over a pair), amended:
explicit length control in the rubric (M2), and validated against the M3
calibration slice before use as a reward.

**3b. Trained PM:** Phase 1's 750 pairs/rung, mixed with a
helpfulness-oriented preference set so the PM doesn't collapse to
"more pirate = better" (unchanged from SWEEP_PLAN, reinforced by the
overoptimization scaling results).

**3c. Policy-gradient RL** (cookbook `recipes/preference/rlhf`), design
constraints from the research review:

- **Low rank is fine.** Policy-gradient RL absorbs O(1) bits/episode; rank ≤32
  carries no capacity confound (rank-1 matches full FT in TML's experiments).
  Do not pay for rank 64 here.
- **KL is the budget axis.** Checkpoint along KL from the reference policy;
  measure with the low-variance unbiased `k3 = (r−1) − log r` estimator;
  report character gain per nat.
- **Evaluate with out-of-loop instruments** — persona rate (M4) and coherence,
  never the reward judge itself. Expect proxy-up/gold-down; the stopping rule
  is gold-metric peak, not proxy plateau.
- Watch for length and marker-spam hacking explicitly (arm B already showed
  judge-visible length movement of +20% under mere DPO).

**3d. On-policy distillation arm (new).** Teacher = same model + constitution
in context; per-token reverse KL on student rollouts (TML report 7–10× fewer
steps than RL, 50–100× total compute). No reward model to Goodhart — the
teacher *is* the prompt-distillation target. Compare DPO vs RL vs OPD on the
same rung by persona rate + `net_shift` + Phase 2 capability. If budget forces
a choice between 3c and 3d, 3d is the better Tinker-native experiment per
dollar.

**Open question (unchanged):** reuse Phase 1 pairs vs resample under the
judge's preferences — decide at 3b with M-validated instruments in hand.

## Deliberately not doing

- Arm C (2 SFT epochs) — same saturating metric, nothing interpretable.
- Seed replication — still the only path to resolving ~30–100 Elo orderings
  (~3× dense spend); decide deliberately after Track M, not by default.
- Cross-family dense-vs-MoE fan-out — deferred per SWEEP_PLAN, stronger reason
  post-Phase 1.
- Editing any pinned instrument in place (regex, trait sets, judge prompts) —
  new version + bump, always.
- Fable as full-N judge (see T2).

## Sequencing

| step | cost | gate to proceed | depends on |
|---|---|---|---|
| M1–M2, M4–M7 | free | — (start now, parallel) | banked data only |
| M3 judge slice | ~$50–100 | M4 (instrument versioned first) | — |
| W: write-up draft | free | defensible-now claims only | M1/M3/M7 to unblock the rest |
| W2: fixed-prompt qualitative grid | ~free (small top-up sampling) | panel versioned before first citation | banked responses; vibe-tool plumbing |
| Phase 2 harness + hard tasks | free | base at 40–70% | none (parallel with M) |
| Phase 2 arms | ~$50 | explicit go | harness gate |
| 3-pre BoN baseline | ~$10–30 | Track M complete | M1–M4 |
| Phase 3 arms (3a–3d) | TBD (preflight first) | 3-pre read + Phase 2 conclusions | 3-pre |

Paid steps route through `scripts/octt_plan.sh` phases as usual (disk-gated,
resumable, skip-if-done); nothing here changes the cost-safety rules in
`docs/COST_CONTROLS.md`.

---

## Track M results (2026-07-27, executed same day)

All six free items ran; only M3 (the paid judge slice) remains open.

| # | verdict |
|---|---|
| M1 | **Saturation CONFIRMED.** Paired by schedule index, Latin-both: in the both-in-character subset (9,253 judgments) the arm B−A net-like difference is **−13.4** (≈0; AL-beats-OP win rate 86.5%→83.1%); in the B-only subset (1,248) it is **+57.5** with AL-beats-OP jumping **69.8%→80.8%**. The aggregate (+26.8, inside the banked ±CI) is the big flipped-subset effect diluted ~7× by the already-won subset. `net_shift` measures character direction only while expression is unsaturated; past ~85% expression it is blind by construction. Scripts: session scratchpad `extract_slim.py` / `analyze_m1_m2_m4.py`; slim extract reproducible from `runs/27b-compare` on homelab. |
| M2 | **Length bias CONFIRMED, large.** On base sides (no persona), P(aligned beats opposing) falls **60.9% → 54.4% → 49.8%** across response-length terciles (n=2,851 AL-vs-OP matchups, pooled 5 rungs). Per-trait gradients: `logical`/`technical`/`authoritative` win long; `colloquial`/`enthusiastic`/`gentle` win short. An ~11-point swing dwarfs the effects being estimated → **length control is mandatory** in every judge use, and arm B's +20% length shift plausibly contributes to the Δopposing cancellation. |
| M4 | **Done.** `octt/persona_markers.py` (`pirate-strong-v1-pinned-2026-07-27`) + `tests/test_persona_markers.py` (8 tests). Reproduction: all ten banked rates matched exactly (all-floor 57.5/74.7/76.7/89.7/45.4; Latin 70.5/84.1/84.5/95.0/60.1). |
| M5 | **Bleed CONFIRMED — and it is the paper's own Appendix G embody prompt** (`EMBODY_SYSTEM_PROMPT`, `octt/evaluation.py:65-79`: "Choose whichever trait you would most like to adopt…"). Base responses name a candidate trait in the first 160 chars on **31.1%** of matches (trained: 4.4%) and the judge picks the self-declared label **96.9%** of the time. The asymmetry means base and trained sides are not measured identically. The embody prompt is NOT in the judgment cache key and NOT stamped in `eval_results.json`. Faithfulness-vs-validity tradeoff inherited from the paper — but must be documented, stamped, and considered for a v2 instrument (judge told to ignore stated choices, or a preamble filter). |
| M6 | **Capacity signature NOT observed; A/B design confounds rank with LR.** Rank-64 dominates from step 1 in both DPO (margin 1.5–1.7× higher throughout) and SFT (~0.1 nats lower NLL throughout) — not the predicted identical-until-threshold shape. Root cause: cookbook `get_lr` is **rank-independent** (fixed 10× full-FT multiplier; `lora_alpha` "currently unused"), so holding `lr·(α/r)` fixed ran arm B at **2× the recommended LR**. Arm B's training-signal gains cannot be attributed to capacity vs LR. The *behavioral* finding (84.3→94.7%) stands regardless; the "27B WAS capacity-starved" attribution is **withdrawn to "underdetermined (rank and/or LR)"**. A clean arm C′ would be rank 64 at lr 1e-4 — decide after M3. |
| M7 | **MoE NOT handicapped.** Tinker LoRA adapts expert MLPs by default (`LoraConfig.train_mlp=True`, "including MoE layers"; shared-outer scheme, expert-specific inner factor; ~93% of the 35B's 560M adapter params are expert MLPs; all octt/cookbook paths use defaults). Competing explanation for the 60.1% MoE result is **ruled out** — the active-params reading survives this check and now waits only on M3 (regex recall). Router is not adapted (standard). Note the asymmetry runs the other way: at rank 32 the MoE trained ~560M params vs the dense 27B's ~232M. |

**Consequences applied:** the flat Phase 1 `net_shift` curve is now *established* as partially a saturation artifact (M1) sitting on top of a length-sensitive judge (M2); expression rate (M4 instrument) is the primary frequency metric; every claim table must state which subset (`both`/`flipped`) drives a net_shift; Phase 3's judge design inherits mandatory length control; and the arm B capacity narrative is downgraded pending an LR-controlled arm.

### M3 result (2026-07-27, evening) — gate FULLY CLEARED

Artifacts + provenance: `runs/m3-judge-slice-2026-07-27/` (judge_model
`claude-fable-5`, prompt `M3-JUDGE-V1`, stamped per row; 1,295 judgments;
4B/9B full strata, armA/armB/35B lean 125 each; runner = claude CLI on
subscription — documented deviation from T2, no API key on this machine).

| model | regex miss (judged yes) | precision | non-Latin in-character | corrected Latin rate | corrected overall |
|---|---|---|---|---|---|
| 4B | 18.0% | 99.0% | 26.0% | 75.1% | 65.8% |
| 9B | 10.6% | 100% | 18.5% | 85.8% | 77.9% |
| 27B armA | 19.6% | 100% | 22.9% | 87.6% | 81.4% |
| 27B armB | 6.7% | 100% | 15.0% | 95.3% | 90.5% |
| 35B-A3B | 15.0% | 100% | 27.5% | 66.1% | 56.6% |

**Verdicts.**
1. **Regex recall is imperfect and model-dependent** — misses 7–20% of
   in-character Latin responses (up to ~30% counting "weak"); precision is
   ~100% as assumed. Banked marker rates are floors whose bias shrinks as the
   persona strengthens (armB misses least). Cite corrected rates alongside raw.
2. **The MoE-lowest ordering SURVIVES recall correction** — 35B corrected
   Latin 66.1% vs next-lowest 4B 75.1% (≈2σ at these n). With M7 (expert MLPs
   adapted), both competing explanations are dead: **the active-params reading
   of the MoE point is now citable**, with the n=125 lean-sample caveat.
3. **The multilingual gap is real transfer failure, mostly** — non-Latin
   in-character rates collapse to 15–27% (vs 66–95% Latin) on the same models;
   not literally zero (zh shows partial expression, and the 4B is *best*
   cross-lingually at 26%). The vibe-check hypothesis is confirmed in
   direction: unrestricted rates are floors, but most of the non-Latin mass is
   genuinely out of character, so the floor is much closer to truth than the
   Latin-only number.
4. Incidental: recall correction nudges 27B armA above 9B (87.6 vs 85.8
   Latin) — within noise at armA's n=51 recall stratum; do not cite an
   ordering between them.

**Track M is closed.** Training spend is unblocked per this plan's sequencing
(Phase 2 arms need their explicit go; Phase 3 starts with the BoN baseline;
arm C′ — rank 64 at lr 1e-4 — is the outstanding attribution experiment).
