# Inkling track — character training on `thinkingmachines/Inkling`

Created 2026-07-15. Third planning doc, sibling of `PLAN.md` (paper-scale OCT replication on
the Qwen/Nemotron ladders) and `NEXT_STEPS.md` (the six tinker-project-ideas tracks). This
document governs porting the OCT recipe to Thinking Machines' Inkling (975B total / 41B active
MoE, 256 routed + 2 shared experts per layer, natively multimodal, thinking-effort dial,
RL-heavy post-training) and the research extensions built on it: a novel constitution, an RL
stage against a constitution preference signal (per
`tinker-project-ideas/replicate-open-character-training.md`), and Inkling-specific evals.

All ground rules from `NEXT_STEPS.md` apply unchanged: dry-run by default, lazy imports,
side-effect-free config, offline tests, manifest everything, cookbook read-only, cost
estimation before spend. Nothing below gets `--execute` unless explicitly asked.

## Verified platform facts (2026-07-15)

- `thinkingmachines/Inkling` is trainable on Tinker now: 64K context, and a 256K variant
  (`thinkingmachines/Inkling:peft:262144`) at 2x price. Inkling-Small (276B/12B active) is
  **not** in the Tinker lineup yet (weights in preview) — this track targets full Inkling;
  swap in Small later as a cheaper rung when it lands.
- Prices per Mtok, **post-2026-07-17 increase** (prefill/sample +~50%, train +~10%, all
  models): Inkling 64K ≈ **$2.81 prefill / $7.02 sample / $6.17 train**. The pre-increase
  numbers ($1.87/$4.68/$5.61) die on the 17th; every `ModelSpec` price in `octt/models.py`
  needs the same refresh.
- Cookbook has native Inkling support upstream (renderer auto-selected, thinking effort as a
  float hyperparameter, multimodal inputs) — but our **vendored `tinker-cookbook/` predates
  Inkling entirely** (zero references). `resolve_renderer_name` goes through vendored
  `model_info`, so nothing Inkling works until the vendor snapshot is updated.
- Inkling's max LoRA rank on Tinker is unverified. Assume the uniform study policy
  (rank 32, lr 1e-4 — `config.for_scaling_study`) until preflight/API says otherwise.
- Local merge is impossible (975B base weights): `--no-merge` is mandatory, same as
  Nemotron Ultra.

## Design decisions (defaults stated; revisit only with evidence)

1. **Self-distillation, not cross-model teaching.** Teacher = `thinkingmachines/Inkling`
   itself: constitution-in-system-prompt → chosen, unprompted → rejected. Same model, same
   capability; the only delta in the DPO pair is the constitution. This removes the
   persona/capability confound a stronger-teacher setup would add, keeps data on-policy, and
   the paper's same-origin introspection findings favor it. Mechanically it is just
   `--teacher thinkingmachines/Inkling` once the renderer resolves.
2. **Reasoning stays out of the datasets (v1).** Repo policy already: identical renderer
   across DPO/introspection/SFT/eval, reasoning OFF for hybrid models
   (`DIRECT_ANSWER_RENDERER_OVERRIDES`). For Inkling, pin thinking effort to its minimum via
   the renderer/sampling config, record the pinned value in `RecipeConfig` so it is hashed and
   held constant. An effort-aware training arm (traces sampled at multiple efforts, think
   tokens masked from the DPO loss) is a v2 experiment, not the baseline — it changes the
   recipe, and the first run should be comparable to the paper and the scaling study.
3. **Token budgets stay paper-faithful (v1):** ~6M DPO / ~8M SFT. The 256-expert
   per-expert-coverage argument says more data may help; that is a measurable follow-up
   (2x-budget arm), not a silent change to the baseline.
4. **Eval judge stays `Qwen/Qwen3.5-397B-A17B`** (the locked study teacher/judge). The policy
   under test is Inkling; using Inkling to judge itself invites self-preference bias. Keeping
   the judge identical to the scaling study also makes the Inkling rung comparable to the six
   existing rungs.
5. **Persona:** one novel constitution (Phase 1 decision, default: calibrated-forecaster) plus
   `humorous` as the cross-study control. New constitutions stay out of paper-replication
   aggregates (the `pirate` rule).

## Phases

### Phase 0 — Port the harness (local, free)

0.1 **Re-vendor `tinker-cookbook/`** at an Inkling-supporting upstream commit (wholesale
    update, never patch — it is read-only). `uv sync --all-extras`; pin the new `tinker` SDK
    version. Gate for this step alone: existing tests still pass — the vendor bump can break
    Qwen/Nemotron paths and must not.
0.2 **Register the model.** Add `thinkingmachines/Inkling` to `models.CANDIDATES`
    (arch "moe", family "Inkling" — `assistant_name()` then names the assistant correctly,
    total 975 / active 41, 64K, post-increase prices, note the 256K variant,
    `max_lora_rank` = whatever 0.4 discovers). **Refresh every other ModelSpec's prices** for
    the 2026-07-17 change so preflight stops understating; note the price date in the module
    docstring.
0.3 **Renderer + reasoning policy.** Discover Inkling's recommended renderer via the updated
    `model_info`; extend `DIRECT_ANSWER_RENDERER_OVERRIDES` (or the Inkling equivalent —
    effort pin) so reasoning stays out of persisted data. Confirm what the renderer does with
    reasoning content in multi-turn history; the introspection self-chats depend on it.
    Confirm the teacher think-prefill trick either renders correctly or is cleanly disabled
    for Inkling (it is an App-A fidelity detail, not load-bearing for self-distillation).
0.4 **Preflight validation.** Teach preflight the Inkling constraints: no-merge required
    (Ultra precedent), rank policy, unknown-rank probe. Extend
    `scripts/octt_plan.sh local` with an Inkling compatibility preflight assertion
    (pattern: the existing Ultra rank32/no-merge check).
0.5 **Tests.** Offline dry-run coverage on the existing patterns: renderer plan for the
    Inkling id, cost-estimate arithmetic with the new prices, manifest round-trip,
    teacher==student wiring, preflight exit codes.

**Gate:** `uv run pytest` + `ruff check` green;
`octt preflight --dry-run --model thinkingmachines/Inkling --teacher thinkingmachines/Inkling
--lora-rank 32 --no-merge` exits 0 with a sane cost table; default all-model preflight still
exits 2.

### Phase 1 — Constitution (local, free)

Write `constitutions/forecaster.txt` (working default; final persona choice is the one open
decision that shapes everything downstream): ~10 first-person assertions in the paper's
pairwise-comparable format, with three deliberate deviations the post-paper literature
supports — a **reason attached to each behavior** (principles generalize better than bare
demonstrations), **behavior-under-stress clauses** (how the character refuses, handles being
wrong — where personas cohere or collapse), and an **anti-caricature clause** ("my character
shows in what I notice and choose, not in constant verbal flourishes") as the cheap defense
against later RL reward-hacking. The calibrated-forecaster persona is the recommended default
because Inkling's calibration was explicitly RL-trained, so this tests whether character
training can amplify a trained behavior into an identity — and it has an objective scoreboard
(Phase 4.2) no stylistic persona has. Tests per `tests/test_constitutions.py`.

### Phase 2 — Smoke + quick on Inkling (paid, ~$–$$)

New plan.sh phase `inkling-smoke` (marker-gated, `source_env`, no disk gate — no merge):
`octt run` at `--scale smoke` then `--scale quick`, model = teacher = Inkling,
`--lora-rank 32 --no-merge --condition all`. Ballpark: smoke <$5, quick ~$15–40 (sample-price
dominated at $7.02/Mtok).

**Gate:** end-to-end completes; eval renders; and a **manual transcript inspection** of the
data sidecars — correct template, no reasoning tokens leaked into chosen/rejected/SFT text,
self-interaction turns alternate correctly. Template bugs at this stage are exactly what the
"Elo measures template mismatch" warning is about; at smoke scale they cost dollars, at paper
scale hundreds.

### Phase 3 — Paper-scale character run (paid, $$$ — the big gate)

`ALLOW_PAPER=1`-gated plan.sh phase, one persona. Run `--condition all` only if the preflight
table is acceptable; the revealed-preferences eval is the dominant line (25k judgments × 3
conditions; responder sampling on Inkling at $7.02/Mtok). **Preflight (2026-07-15, post-Phase-0
registry): $1,585 pessimistic envelope for one condition** — the estimator bills every sampled
call at its max-token envelope per docs/COST_CONTROLS.md, so actual spend typically lands well
under (~40–60% of envelope ≈ $650–950); the earlier $400–800 ballpark assumed ~400-token
average responses. Biggest envelope lines: eval responder sampling (~$360) + eval/judge prefill
(~$340), introspection sampling+prefill (~$660). Run the forecaster persona and (budget
permitting) the humorous control — robustness and coherence evals only become meaningful with
≥2 personas per model.

**Gate:** per-trait Elo shift shows the paper's Fig-3 pattern (desired traits rise, opposing
fall); capability smoke (`lighteval-smoke` path) shows no crater. Only then is the Inkling
rung a real data point next to the six study rungs.

### Phase 4 — Inkling-specific evals (new code; cheap paid sampling)

The two regression tests the paper never needed, because its models had neither an effort dial
nor RL-trained calibration:

4.1 **Effort sweep.** Sample a fixed prompt set (reuse a capability-style set with gradeable
    answers) at ~4 thinking-effort values, base vs post-character checkpoint; plot
    accuracy-vs-effort. A character run that flattens or shifts the curve damaged a headline
    model feature — that is a stop-the-line finding. Small module + plan.sh phase; sampling
    cost is trivial next to Phase 3.
4.2 **Calibration.** Brier/ECE + abstention rate on a resolved-question set, base vs trained.
    For the forecaster persona this is the primary outcome measure, not a guardrail: character
    training that measurably improves calibration is a novel result; one that degrades it
    while sounding epistemically virtuous is a more important one.
4.3 Standard suite as usual (robustness, coherence, capabilities). Mac constraint from
    `PLAN.md` stands: ModernBERT classifier training is the overnight/rented-GPU job.

### Phase 5 — RL stage against a constitution preference signal (paid, $$$, budget-boxed)

The project-ideas extension, and the escalation path if Phase 3's Elo shift is weak against
Inkling's RL-hardened prior. Builds directly on `NEXT_STEPS.md` Phase 0 (harness
generalization) + Phase 2 (`direct_rlaif`: `PromptedPairwiseJudge`, tournament rewards, RM
pipeline) — **implement those first**; this phase is their Phase-2b character tie-in with
Inkling as policy. Start from the post-introspection checkpoint, never vanilla Inkling.

Two arms, per the project-ideas note, in this order:
- **Prompted judge:** constitution in the judge's context (judge = Qwen3.5-397B, not Inkling —
  self-judging invites self-preference and shared-failure-surface hacking), strict
  parse-or-discard verdicts, order-swap handled by the both-ways tournament. Judge-call volume
  is the dominant cost line — surface it in the dry-run table.
- **Trained preference model:** stage-2 self-distillation pairs are labeled comparisons for
  free; add judge-labeled on-policy pairs so the RM sees the distribution RL queries it on,
  and mix in a helpfulness preference set (cookbook HHH path) so "drenched in persona but
  useless" cannot beat "helpful and in-character."

Loss and guardrails: **sequence-level (GSPO-style) importance ratios, not token-level** —
token-level ratios are the documented failure mode for RL on many-expert MoE (~10% of
activated experts flip per update), and character reward is sequence-level anyway. KL leash to
the post-character reference; rank 32 (RL needs little adapter capacity); monitor response
length, per-trait Elo trajectory on a held-out slice, and judge-vs-heldout-judge divergence as
the reward-hacking tripwires. Stop condition is the held-out eval suite, never the reward
curve. Budget-box hard via the Phase-0.2-style estimator; expect ~$300–1,000 depending on
steps × group size.

### Phase 6 — Research deliverable: EM-robustness at frontier MoE scale (optional headline)

The open question nobody has tested above 8B dense: does character training's protective
effect against emergent misalignment hold on a frontier-scale, RL-hardened MoE? Design:
fine-tune {no-character control, forecaster, humorous} Inkling checkpoints on a standard
EM-inducing set (insecure code, Betley et al.), measure misalignment rates on the public EM
evals. Every step is LoRA on Tinker; the checkpoints exist after Phase 3. Interesting in
either direction: protection replicating at scale is a real safety finding; protection
failing says the 8B results don't extrapolate. Write up regardless.

## Sequencing and budget summary

| Phase | What | New spend | Dominant cost | Hard prereq |
|---|---|---|---|---|
| 0 | harness port + re-vendor | $0 | — | — |
| 1 | constitution | $0 | — | — |
| 2 | Inkling smoke/quick | ~$20–50 | quick sampling | 0, 1 |
| 3 | paper-scale persona | ~$400–800 | revealed-prefs eval | 2 |
| 4 | effort sweep + calibration | ~$10–50 | eval sampling | 2 (4.1/4.2 baselines can run pre-3) |
| 5 | RL stage | ~$300–1,000 (boxed) | judge calls / rollouts | 3 + NEXT_STEPS 0 & 2 |
| 6 | EM-robustness study | ~$100–300 | EM fine-tunes + evals | 3 |

Ballparks assume post-2026-07-17 prices; `octt preflight` is the authority once Phase 0.2
lands. Phases 4.1/4.2 baseline measurements (on base Inkling) are cheap and should run before
Phase 3 so before/after is real.

## Open decisions (defaults stated)

1. **Persona** — default calibrated-forecaster; alternatives from the design discussion:
   tension-based ("honest about ideas, gentle with people"), anti-engagement, worldview
   (Stoic). Decide before Phase 1; everything downstream keys on it.
2. **Eval conditions at Phase 3** — default 1 condition first, `--condition all` only on the
   run that becomes the reported number.
3. **Effort pin value** (Phase 0.3) — whatever the renderer exposes as minimal; record it.
4. **2x-token-budget arm** (expert-coverage question) — only after Phase 3 baseline, as a
   deliberate A/B, never a default change.
5. **Inkling-Small swap-in** — when it reaches the Tinker lineup, add its ModelSpec and rerun
   Phases 2–4 there as the cheap rung; full Inkling then becomes the transfer-validation run.

## Phase 0 findings (2026-07-15)

- Vendored cookbook now v0.5.2; deps `tinker 0.23.0`, `tinker-cookbook[inkling] 0.5.2`,
  `tml-renderers 0.1.0`. Renderer policy implemented as a registered
  `octt/tml_v0_pinned_effort` subclass (effort pinned to 0, rendered as a
  "Thinking effort level: 0" system directive on both generation and SFT paths).
- `tml_v0` supports no assistant prefill, so the App A teacher think-prefill is skipped for
  Inkling teachers (`renderer_supports_think_prefill`); teacher and student sample identically —
  strictly cleaner for the "only delta is the constitution" design.
- The service lists `thinkingmachines/Inkling` and its 256K variant; tokenizer + renderer
  round-trip verified locally before any spend.
- **Recipe bug found & fixed during the port:** `pipeline.run` never passed
  `config.learning_rate` into `distillation.train`/`introspection.train` — both trained at
  their hardcoded 5e-5 defaults, so the uniform-rank32 scaling-study policy (lr 1e-4 via
  `for_scaling_study`) was a silent no-op for any prior `octt scaling` run. Training now
  defaults to the config lr; `octt run` gained `--learning-rate`. Prior scaling smokes that
  relied on the lr policy trained at half the intended effective update scale.

## Risks

- **Vendor bump blast radius** (Phase 0.1): the re-vendored cookbook must not break the six
  existing study rungs — the existing test suite is the gate, run it before anything Inkling.
- **Unknown LoRA rank cap / API constraints** for Inkling: discovered in 0.4, may force a
  policy note like Ultra's.
- **DPO may move an RL-hardened persona weakly**: expected risk, and Phase 5 is the designed
  escalation; do not silently inflate budgets in Phase 3 to compensate.
- **Preview-era churn**: Inkling checkpoints/renderers may be revised upstream; pin the
  `tinker` SDK version and record model/renderer versions in every run manifest.
