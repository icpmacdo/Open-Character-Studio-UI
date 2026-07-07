# Next steps — implementing the remaining tinker-project-ideas

Updated 2026-07-06. This is the implementation guide for the six unimplemented ideas in
`tinker-project-ideas/`. It is the sibling of `PLAN.md` (the paper-scale OCT track): PLAN.md
governs finishing the Open Character Training replication; this document governs the new
experiment tracks built on the same harness. The two tracks share cost gates and can interleave;
Phase 2 below is the explicit bridge (it is also PLAN.md's "optional extension").

## Idea status

| Idea | Status | Phase here |
|---|---|---|
| `replicate-open-character-training.md` | **Done** (this repo; see PLAN.md) | — (RL extension → Phase 2b) |
| `memorization-empirical-study.md` | Not started | Phase 1 |
| `direct-rl-on-pairwise-judge.md` | Not started | Phase 2 |
| `on-policy-context-distillation.md` | Not started | Phase 3 |
| `noisy-student.md` | Not started | Phase 4 |
| `replicate-cai-with-base-models.md` | Not started | Phase 5 |
| `gan-joke-generation.md` | Not started | Phase 6 |

Ordering rationale: Phase 0 removes the four harness frictions every track hits; then phases run
cheapest-and-most-ready-made → most-novel. Each phase is independently shippable; later phases
reuse earlier ones (2→5, 2→6) but only Phase 0 is a hard prerequisite for all.

## Ground rules (every phase)

These are the existing repo conventions; new experiment code follows them without exception.

- **Dry-run by default.** Every new CLI path is free/offline unless `--execute` is passed. Paid
  runs go through a `scripts/octt_plan.sh` phase (marker file, `run_if_missing`, disk gate where
  relevant). Never add `--execute` to a command without the user explicitly asking.
- **Lazy imports.** `tinker`, `tinker_cookbook`, `torch`, `transformers`, `datasets` are imported
  inside functions. `uv run pytest` must pass with no training stack and no API keys.
- **Side-effect-free config.** New experiment configs live in the experiment module (or a new
  `octt/experiments/config.py`), never as side effects in `octt/config.py` / `octt/models.py`.
- **Offline tests.** Each experiment ships `tests/test_exp_<name>.py` on the existing pattern:
  a local `_dry_runtime()` built from `TinkerClientConfig(dry_run=True)`, real stage functions
  driven with `offline=True`, assertions on deterministic stub output + files under `tmp_path`.
  Anything that judges/scores gets a deterministic offline verdict path (pattern:
  `octt/evaluation.py::_dry_run_winner`). No monkeypatching of Tinker internals.
- **Manifest everything.** Every paid stage is recorded via `manifest.RunManifest.record_stage`
  so reruns skip finished work. Expensive intermediate artifacts (teacher samples, judge verdicts,
  pseudo-labels) are written to content-hash-cached JSONL sidecars *before* dependent stages run
  (pattern: `octt/distillation.py::generate_pairs` persisting teacher completions first).
- **Cookbook is read-only.** Everything below subclasses or wraps `tinker_cookbook` symbols from
  `octt/experiments/`; we never edit vendored code. Don't apply cookbook lint/CI rules to `octt/`.
- **Cost estimation before spend.** Each experiment implements an estimator returning
  `tinker_client.CostEstimate` (Phase 0.2) so `octt preflight`-style gating and `--budget`
  blockers work for it.

---

## Phase 0 — Generalize the harness (local, free)

**Goal:** remove the four OCT-specific frictions identified in the codebase survey so all six
tracks plug into the same rails: manifest identity, cost estimation, data contracts, CLI/test
layout.

### 0.1 Widen manifest identity beyond (model, persona)

`octt/manifest.py` keys runs on `run_id(model, persona, config)` and
`RunManifest.load_or_create(..., model, persona, config, teacher)` fail-fasts on that identity.

- Add an `experiment: str = "oct"` keyword to `run_id` and `RunManifest.load_or_create`, and a
  generic `subject: str` alias for the persona slot (personas remain the subject for OCT runs).
- **Back-compat requirement:** when `experiment == "oct"`, the emitted `run_id` string and the
  manifest identity check must be byte-identical to today's, so existing `runs/` dirs still
  resume via skip-if-done. Only non-default experiments fold the experiment name into the id
  (e.g. `exp-memorization-<subject>-<model>-<config_hash>`).
- New experiment runs live under `runs/exp/<run_id>/` to keep the OCT namespace clean.
- Tests: extend `tests/` with (a) golden assertion that an OCT `run_id` is unchanged, (b) a new
  experiment id round-trips through `RunManifest.load_or_create` + `record_stage` + resume.

### 0.2 Pluggable cost estimation

`octt/tinker_client.py::estimate_tinker_cost` is hard-keyed to the three OCT stages
(`dpo.*` / `introspection.*` / `eval.*`) reading off `RecipeConfig`.

- Extract the reusable primitives (already generic): `CostEstimateLine`, `CostEstimate`,
  per-model prices from `models.CANDIDATES`, `TEACHER_SAMPLE_PRICE_USD_PER_MTOK` fallback.
- Add a small protocol: each experiment module exposes
  `def estimate_cost(cfg: <ExpConfig>) -> CostEstimate` composing `CostEstimateLine`s.
- Refactor `build_preflight_report` to accept an optional `estimator: Callable[[], CostEstimate]`
  (default = the current OCT estimator, behavior unchanged). `--budget` blockers and rank
  validation then apply to any experiment for free.
- Tests: existing preflight exit-code tests in `tests/test_tinker_client.py` must pass untouched;
  add one test that a toy estimator flows through `build_preflight_report` and trips `--budget`.

### 0.3 Richer data records

`octt/data_sources.py` loaders return bare `list[str]` first-user-turns. RLVR needs
(prompt, gold), ICL needs (context items, query), jokes need (instruction, text).

- Add `@dataclass(frozen=True) class PromptRecord: prompt: str; gold: str | None = None;
  meta: tuple[tuple[str, str], ...] = ()` (tuple-of-pairs, not dict, so records stay hashable
  for `content_hash`). Existing loaders stay as-is; new loaders return `list[PromptRecord]`.
- Keep the established loader shape exactly: a `_load()` closure + module-level offline fixture
  tuple + funnel through `_try_load_hf(loader, fixture, n, *, offline, name)` + a pinned commit
  SHA in `HF_DATASET_REVISIONS`. Adding a dataset stays a ~15-line change.
- New loaders (added in the phase that needs them, but the contract lands here):
  - `load_gsm8k_records(n, *, split, offline)` — `openai/gsm8k` (Phase 4; the cookbook's
    `recipes/rl_loop.py` shows the answer-extraction convention `#### <answer>`).
  - `load_banking77_records(n, *, split, offline)` — `PolyAI/banking77` (Phase 3).
  - `load_jokes_records(n, *, offline)` — jokes corpus (Phase 6; dataset choice is an open
    decision, see end of doc).
- Tests: fixture cycling + offline fallback per loader, mirroring `tests/test_data_sources.py`.

### 0.4 Experiment package, CLI namespace, plan.sh phases

- New subpackage `octt/experiments/` (first-party code lives in `octt/` per CLAUDE.md; the
  existing top-level `experiments/scaling.py` stays where it is — it belongs to the OCT scaling
  study):
  ```
  octt/experiments/__init__.py       # registry: name -> module (lazy)
  octt/experiments/memorization.py         # Phase 1
  octt/experiments/direct_rlaif.py         # Phase 2
  octt/experiments/context_distillation.py # Phase 3
  octt/experiments/noisy_student.py        # Phase 4
  octt/experiments/cai_base.py             # Phase 5
  octt/experiments/gan_jokes.py            # Phase 6
  ```
- Each module exposes the same trio: `Config` (frozen dataclass), `run(cfg, runtime, out_dir, *,
  execute=False, offline=False) -> RunReport`, `estimate_cost(cfg) -> CostEstimate`.
- CLI: add `octt exp <name> [--execute] [--budget USD] [--out DIR] ...` to `octt/cli.py`,
  dispatching through the registry. `octt exp <name> --dry-run` (the default) prints the cost
  table and the stage plan, exits 0; with `--budget` exceeded it exits 2, same contract as
  `octt preflight`.
- `scripts/octt_plan.sh`: one `cmd_exp_<name>` phase per experiment using `run_if_missing` with a
  `report.json` marker under `runs/exp/...`; paid phases require `TINKER_API_KEY` via
  `source_env` like the existing phases.
- **Phase 0 gate:** all existing tests pass (`uv run pytest`), `ruff check` clean,
  `scripts/octt_plan.sh local` still green (OCT preflight exit codes unchanged).

---

## Phase 1 — Memorization empirical study (`memorization-empirical-study.md`) — cost ~$

**Goal:** measure the rate at which RL absorbs "random" information (a latent integer in
[1..N], entropy log₂N bits) versus supervised learning, and check it against the
LoRA-without-regret theoretical arguments. Also the cheapest possible end-to-end validation that
our harness drives the cookbook RL loop (`tinker_cookbook/rl/train.py`) correctly.

### What exists

- `tinker_cookbook/recipes/multiplayer_rl/guess_number/env.py::GuessNumberEnv` is nearly the
  required env (holds a `gold_answer: int`, parses `Guess: N`, rewards a match).
- `tinker_cookbook/rl/problem_env.py::ProblemEnv` — subclass with `get_question`,
  `check_answer`, `check_format`; reward `= format_coef*(format-1) + correct`. Paired with
  `ProblemGroupBuilder(env_thunk, num_envs, dataset_name)`.
- `tinker_cookbook/rl/types.py` — `Env` / `EnvGroupBuilder` / `RLDataset` / `RLDatasetBuilder`
  interfaces; `tinker_cookbook/recipes/rl_basic.py` is the ~44-line launcher to pattern-match.

### Implementation (`octt/experiments/memorization.py`)

1. **Env.** `MemorizationEnv(ProblemEnv)`: constant question ("There is a secret integer between
   1 and {N}. State your best guess as `Answer: <n>`."), `check_answer` compares the parsed
   integer to the latent secret. The secret is drawn once per run from a seeded RNG carried in
   `Config` (never `random` at import time; determinism is required for resume and tests).
2. **Reward variants** (a `Config.reward` enum):
   - `binary` — 1.0 on exact match else 0.0;
   - `continuous` — `1 - |guess - secret|/N` (distance-shaped);
   - `per_step` — decompose the answer into digits and reward per correct digit position via
     per-step rewards in a small multi-turn env (this is the "per-step reward" arm of the idea's
     final question).
3. **Dataset.** `MemorizationDataset(RLDataset)` returns the same `ProblemGroupBuilder` every
   batch (the whole point is repetition of one latent fact). `group_size` G and batch size come
   from `Config`.
4. **SL arm.** A minimal SFT loop (cookbook `supervised` + `FromConversationFileBuilder`, same
   machinery `octt/introspection.py` already uses) on the single conversation
   (question → "Answer: {secret}"); measure optimizer steps until greedy decode is correct.
5. **Measurement.** Sweep `N ∈ {2⁴, 2⁸, 2¹², 2¹⁶}` × reward variants; for each run record
   episodes (and total samples = episodes × G) until first success and until success-rate ≥ 0.9
   over a trailing window. Report bits-absorbed vs episodes → empirical bits/episode; compare
   with the theory's ~O(1) bit/episode prediction for policy gradient with binary reward, and
   with SL's steps-to-memorization. Write `report.json` + a markdown table per sweep cell.
6. **Model.** The smallest rung, `DENSE_LADDER[0]` in `octt/models.py` (Qwen 4B). LoRA rank 32,
   short `max_tokens` (≤32) — episodes are single short turns, so cost is tiny.
7. **Cost estimator.** `episodes × G × (prompt_tokens + answer_tokens)` sampling +
   `episodes × G × train_tokens` training, priced from `models.CANDIDATES`. Expect single-digit
   dollars per sweep cell; the sweep is the budget knob.

### Tests

- Parse/reward unit tests for all three reward variants (including malformed guesses → format
  penalty, never a crash).
- Deterministic secret from seed; `RLDataset` batch shape; `estimate_cost` arithmetic.
- Dry-run CLI: `octt exp memorization` prints plan + cost, writes no run dir.

### Gate

Live run at N=2⁸, binary reward completes; success-rate reaches ≥0.9; bits/episode lands within
an order of magnitude of theory; report renders. Then run the full sweep.

---

## Phase 2 — Direct RL on pairwise judge (`direct-rl-on-pairwise-judge.md`) — cost ~$$

**Goal:** quantitatively compare direct RLAIF (query a prompted judge for G²−G pairwise matchups
during RL) against indirect RLAIF (judge labels a preference dataset → train an RM → RL on the
RM), with instruction-following evals, a larger held-out judge, and qualitative analysis.

### What exists

- The entire indirect path: `tinker_cookbook/recipes/preference/rlhf/rlhf_pipeline.py` (stages
  `run_sft` / `run_rm` / `run_rl`; the RM is a chat model SFT-trained to emit the `A`/`B`/`Tie`
  token after `==== Preference ====`).
- The exact matchup reward: `tinker_cookbook/rl/preference_envs.py::PairwisePreferenceGroupBuilder`
  with `TournamentPattern.ALL_PAIRS_BOTH_WAYS` = every ordered pair i≠j = **G²−G matchups**,
  reward = `win_minus_loss / matchup_count`; `matchup_group_size` (default 4) caps comparisons.
- The judge interface: `tinker_cookbook/preference/types.py::PreferenceModel` (async
  `__call__(comparison) -> float` in [−1,1]);
  `recipes/preference/shorter/env.py::PreferenceModelShorter` is the ~30-line template for a
  judge with no trained weights.
- Our own: `octt/distillation.py::generate_pairs` already writes cookbook
  `comparison`/`label` JSONL consumable by
  `tinker_cookbook/preference/preference_datasets.py::ComparisonBuilderFromJsonl` with zero
  conversion; `octt/evaluation.py::parse_judge_verdict` shows the strict-parse-or-discard
  verdict convention; `octt/coherence.py::resolve_pair` shows order-swap calibration.

### Implementation (`octt/experiments/direct_rlaif.py`)

1. **Prompted judge.** `class PromptedPairwiseJudge(PreferenceModel)`: wraps a sampling client
   for `Config.judge_model` (default `models.TEACHER_MODEL`); renders the prompt conversation +
   Completion A + Completion B with an instruction-following rubric; asks for
   `<answer>A</answer>` / `<answer>B</answer>` / `<answer>Tie</answer>`; strict parse
   (unparseable → 0.0, counted in metrics, never defaulted to a winner — same convention as
   `parse_judge_verdict`). Return +1/−1/0. Per-call order swap is unnecessary — the tournament
   already runs both orders. Plus a `@chz.chz PromptedJudgeBuilder` factory mirroring
   `PreferenceModelBuilderFromChatRenderer` so it plugs into `PairwisePreferenceRLDatasetBuilder`
   as `preference_model_builder`.
2. **Common setup.** Policy = 4B rung (9B if budget allows); prompts from the existing
   `load_lima_prompts` / `load_wildchat_prompts` loaders; a fixed held-out prompt set (~200) and
   a fixed qualitative set (50) frozen into `data/` with a content hash.
3. **Indirect arm.**
   a. Preference-set generation: sample G=4 completions per training prompt from the *initial*
      policy; label all pairs with `PromptedPairwiseJudge`; write cookbook-format JSONL (reuse
      the sidecar-first caching from `generate_pairs`).
   b. RM training: `rlhf_pipeline`-style `train_rm` over `ComparisonBuilderFromJsonl` on our
      JSONL (with the built-in A/B swap augmentation).
   c. RL: `rl/train.py` with `PairwisePreferenceRLDatasetBuilder` +
      `PreferenceModelBuilderFromChatRenderer(rm_weights_path=...)`,
      `tournament_pattern=ALL_PAIRS_BOTH_WAYS`.
4. **Direct arm.** Identical RL config; only `preference_model_builder` differs (the prompted
   judge). **Fairness controls:** match total judge-call budget (indirect spends its calls
   up-front on the dataset; log both arms' judge-call counts), same policy init, same prompt
   stream, same steps/lr/rank.
5. **Evals.**
   - Instruction following: IFEval through the existing LightEval harness
     (`octt/capabilities.py`); add it to the benchmark set in `octt/config.py::get_capability_config`
     if not already present.
   - Big-judge audit: score both final policies (and 2–3 intermediate checkpoints) on the
     held-out set with a *different, larger* judge — use one of the alternatives recorded in
     `octt/models.py` (Kimi-K2.6 or DeepSeek-V3.1), not the training judge — win-fraction vs the
     initial policy. Divergence between small-judge and big-judge win-rates over training is the
     reward-hacking signal the idea asks for.
   - Hacking probes: response-length drift, formatting-artifact counts per checkpoint.
   - Qualitative: side-by-side markdown dump of the 50-prompt set (initial / direct / indirect).
6. **Cost estimator.** Direct-arm judge calls = `steps × groups_per_batch × min(G²−G, cap from
   matchup_group_size)`; sampling = `steps × groups × G × max_tokens`. Judge tokens priced at the
   teacher rate. This is the dominant line — surface it in the dry-run cost table.
7. **Phase 2b (character tie-in, optional, after PLAN.md phase 3).** Swap the IF rubric for a
   constitution-conditioned judge (constitution text in the judge system prompt) and run the
   direct arm against one OCT persona; compare persona-trait Elo shift
   (`octt/evaluation.py::revealed_preferences`) vs the DPO rung at matched cost. This is
   PLAN.md's "optional extension" — one implementation serves both documents.

### Tests

- Judge render + strict-parse tests with a deterministic dry-run verdict (pattern:
  `_dry_run_winner`); unparseable → 0.0 and counted.
- Preference JSONL schema test: rows round-trip through `Comparison`/`LabeledComparison`
  construction (`tinker_cookbook/preference/types.py`) offline.
- Cost estimator arithmetic incl. the `matchup_group_size` cap; CLI dry-run.

### Gate

Both arms complete at 4B with matched budgets; IFEval + big-judge win-rates reported; a
direct-vs-indirect verdict paragraph with the hacking-probe evidence. Only then consider 9B or
Phase 2b.

---

## Phase 3 — On-policy context distillation (`on-policy-context-distillation.md`) — cost ~$$

**Goal:** compare off-policy context distillation, on-policy distillation, and off-then-on, in a
many-shot ICL setting: teacher = model *with* a k-shot context, student = same weights with an
empty context.

### What exists

- Off-policy: `tinker_cookbook/recipes/prompt_distillation/` (`create_data.py` samples the
  teacher with long prompt `p` + query `q`; `train.py` SFTs the student on `(q → r)` via
  `supervised/data.py::FromConversationFileBuilder`). Our `octt/introspection.py` is the same
  shape.
- On-policy: `tinker_cookbook/distillation/train_on_policy.py` —
  `incorporate_kl_penalty(...)` computes teacher logprobs via
  `teacher_client.compute_logprobs_async`, forms per-token reverse KL
  (`log p_student − log q_teacher`), folds `−kl_penalty_coef × KL` into advantages;
  `loss_fn="importance_sampling"`. Dataset plumbing:
  `distillation/datasets.py::{TeacherConfig, DistillationDatasetConfig, PromptOnlyDatasetBuilder}`.
- Asymmetric teacher/student prompts: `tinker_cookbook/distillation/sdft.py::build_sdft_teacher_prompt`
  is the exact wiring pattern (teacher sees extra context, student doesn't). Caveat noted in
  `incorporate_kl_penalty`: if teacher and student renderers differ, token sequences must be
  remapped — irrelevant here because teacher and student are the same model, but the teacher's
  `ModelInput` must prepend the k-shot prefix before the student-sampled tokens.

### Implementation (`octt/experiments/context_distillation.py`)

1. **Task.** Default: `PolyAI/banking77` intent classification (a Many-Shot ICL paper task) via
   `load_banking77_records` (Phase 0.3): k-shot context = k labeled examples rendered as
   demonstration turns; query = one held-out utterance; gold = intent label. `Config.shots`
   default 128 (fits the 4B/9B context; `context_k` is carried per model in `models.CANDIDATES`).
   Design the loader so a second task can be added later without touching the arms.
2. **Ceilings/floors first.** Before any training, evaluate (a) 0-shot student and (b) k-shot
   teacher on the eval split. These two numbers bound every arm; if the gap is small at 4B, move
   up a rung or raise k before spending on training.
3. **Arm A — off-policy.** Sample teacher (k-shot context) on the train queries → JSONL of
   `(query → response)` conversations (content-hash cached, sidecar-first) → 1–2 epoch LoRA SFT
   via `FromConversationFileBuilder`.
4. **Arm B — on-policy.** `train_on_policy`-style loop: student samples on bare queries;
   teacher logprobs computed with the k-shot prefix prepended (the `build_sdft_teacher_prompt`
   pattern); reverse-KL-only signal (no task reward), `kl_penalty_coef` from `Config`
   (start 1.0, the recipe default).
5. **Arm C — A then B**, initializing B from A's checkpoint (`load_checkpoint_path` in
   `TeacherConfig`/policy config).
6. **Eval.** Accuracy on held-out queries (exact-match label after strict parse) for: 0-shot
   base, k-shot teacher, A, B, C — reported against training-token budget (the arms must be
   compared at matched token spend; log tokens per arm). Report as `report.json` + markdown.
7. **Cost estimator.** Arm B is prefill-dominated (teacher recomputes the k-shot prefix every
   logprob call) — cost it with per-model `price_prefill`; this will be the biggest line and the
   reason to keep `shots × train_queries` modest at first (e.g. 128 × 2000).
8. **Phase 3b (optional, character tie-in).** Re-run arms A/B on the OCT introspection stage
   (teacher = post-DPO model with constitution system prompt; student = same weights, no system
   prompt) and compare persona Elo. Answers whether on-policy prompt distillation beats the
   paper's off-policy SFT for character — a genuine beyond-paper result for the writeup.

### Tests

- Renderer/ModelInput alignment test (offline, dry-run tokenizer): the teacher input =
  k-shot prefix + student tokens, and logprob positions line up (this is the one thing most
  likely to be silently wrong — assert lengths, not just shapes).
- Loader fixtures; strict label parse; matched-token accounting; CLI dry-run.

### Gate

At 4B/banking77: teacher(k-shot) − student(0-shot) gap reproduced; all three arms complete at
matched token budget; the off/on/off+on ordering result is stated with accuracy ± noise across
2 seeds.

---

## Phase 4 — Noisy student for RLVR (`noisy-student.md`) — cost ~$$–$$$

**Goal:** bootstrap from a small labeled RLVR training set to a large unlabeled pool via
consensus pseudo-labels: RL on seed → consensus-label pool → RL on pseudo-labels → relabel →
repeat. Compare against RL-on-seed-only at matched compute, and try the SFT-distillation variant
for step 3.

### What exists

- RLVR loop: `tinker_cookbook/recipes/math_rl/math_env.py::{MathEnv, MathDataset,
  Gsm8kDatasetBuilder}` + `recipes/rl_basic.py`; answer extraction/grading conventions live
  there and in `recipes/rl_loop.py` (GSM8K `#### <answer>` format).
- Self-distillation plumbing for the SFT variant: `tinker_cookbook/distillation/sdft.py`
  (teacher = same model with the answer in context; top-K logprob distillation via
  `build_topk_distillation_datums`, reverse-KL via `build_reverse_kl_datums`).
- **Nothing** implements consensus/majority-vote labeling — that's the DIY core (confirmed by
  survey: no such helper exists in the cookbook).

### Implementation (`octt/experiments/noisy_student.py`)

1. **Data split.** `load_gsm8k_records` (Phase 0.3): labeled seed = first `Config.seed_size`
   (default 500) train problems *with* gold; "unlabeled" pool = next `Config.pool_size` (default
   2000) train problems with gold **held aside for measurement only** — the training loop must
   never read pool gold. Enforce this structurally: the pool records handed to training have
   `gold=None`; the measurement path reads a separate file. Eval = official GSM8K test split.
2. **Consensus labeler.** Pure function
   `consensus(completions: list[str]) -> tuple[str | None, float]`: extract answers using the
   math_env convention, majority vote, return (answer, agreement ratio); `None` if the top
   answer's ratio < `Config.tau` (default 0.6). Unit-testable with zero Tinker.
3. **Round structure** (each round = manifest stages `ns-r<i>-label`, `ns-r<i>-train`):
   - r0: RL on the labeled seed (`ProblemEnv` subclass over `PromptRecord`s; reward = exact
     answer match, format penalty per `ProblemEnv` convention).
   - label: sample k=16 completions per pool problem at temperature (`GEN_TEMPERATURE`), run
     `consensus`, keep problems above τ; write pseudo-label JSONL sidecar (cached — this is the
     expensive artifact).
   - r1..R: RL on kept pseudo-labeled problems (consensus answer plays the role of gold),
     optionally mixed with the seed at a fixed ratio; then relabel the *whole* pool with the
     improved model and repeat. Default R=3 rounds.
   - Variant arm (flag): replace the RL step with SFT on the consensus-voted completions
     (pick the majority-answer completion per problem as the target), or SDFT-style top-K
     distillation — the idea explicitly asks for this comparison.
4. **Measurement per round** (this is what makes it a study, not just a training trick):
   test-set accuracy; pseudo-label precision vs held-aside gold; coverage (fraction of pool above
   τ); consensus-confidence calibration (agreement ratio vs actual correctness, bucketed).
   Baseline: RL on seed only, run for the same total sampled tokens as the full loop.
5. **Failure watch.** Confidence collapse (coverage → 1.0 while precision drops) is the known
   failure mode of self-training; if precision falls round-over-round, raise τ or stop — the
   report should show the trend either way. The idea notes explicit noising may be unnecessary
   (sampling noise suffices); an optional temperature-bump arm tests that cheaply.
6. **Cost estimator.** Labeling dominates: `rounds × pool_size × k × avg_completion_tokens`
   sampling + RL training tokens. Start with pool 2000, k 16; scale only after round-1 precision
   looks sane.
7. **Model.** 4B rung (GSM8K at 4B leaves useful headroom; if seed-RL alone saturates the test
   set, swap to the harder MATH subset the cookbook's math_rl also supports).

### Tests

- `consensus` unit tests (ties, τ boundary, unparseable completions, unanimous).
- Structural no-leak test: the training-path records for the pool have `gold=None`.
- Round/stage resume: kill after `ns-r1-label`, rerun, assert the label sidecar is reused.
- Cost estimator; CLI dry-run.

### Gate

Round-over-round test-accuracy curve beats the matched-compute seed-only baseline (or the report
documents cleanly why not — a negative result with the calibration plot is still a result);
pseudo-label precision reported per round.

---

## Phase 5 — Replicate CAI with base models (`replicate-cai-with-base-models.md`) — cost ~$$$

**Goal:** the original Constitutional AI pipeline *de novo* from a base model — no
instruction-tuned model anywhere in the data path: helpful-only SFT → SL-CAI
(critique → revision) → RLAIF with few-shot-prompted pairwise comparisons. Test whether the
result is more base-model-like (style steerability).

### What exists

- Base-model support is first-class in the cookbook:
  `tinker_cookbook/renderers/role_colon.py::RoleColonRenderer` (`Role: content`, the format the
  cookbook notes is Anthropic-like), registered as `"role_colon"`; base checkpoints (e.g.
  `Qwen3.5-9B-Base`, per `tinker_cookbook/model_info.py`) map to it as the recommended renderer;
  `recipes/rl_basic.py` and `math_env.py` default to a base model.
- Few-shot prompting: `convo_prefix: list[Message]` threaded through `ProblemEnv`
  (`MathEnv.standard_fewshot_prefix()` is the worked example).
- HH-RLHF: `tinker_cookbook/recipes/preference/datasets.py::HHHComparisonBuilder`
  (loads `Anthropic/hh-rlhf`). No helpful-only builder exists — DIY.
- RLAIF machinery: Phase 2's `PromptedPairwiseJudge` + the rlhf_pipeline RM path, both reusable
  with a base-model few-shot judge instead of an instruct judge.
- Our own: the critique→revision generation chain is structurally identical to
  `octt/distillation.py::generate_pairs` (batched sampling, sidecar-first caching, min-count
  guards); `octt/constitution.py` parses constitution files.

### Implementation (`octt/experiments/cai_base.py`)

**Model policy:** everything derives from one base checkpoint (default `Qwen3.5-9B-Base`; add it
to `models.CANDIDATES` with prices per Phase 0.2 so preflight can cost it). The strict rule that
makes this replication meaningful: **no instruct model output enters any training set** — data
generators are the base model few-shot-prompted, or models fine-tuned from it. Enforce by
construction: `Config` has no teacher field, only `base_model`.

1. **Stage 1 — helpful-only model.** DIY `HelpfulOnlyBuilder`: load `Anthropic/hh-rlhf`
   restricted to the `helpful-base`/`helpful-online`/`helpful-rejection-sampled` subsets, take
   the *chosen* conversations, SFT the base model on assistant turns
   (`TrainOnWhat.ALL_ASSISTANT_MESSAGES`, renderer `role_colon`). Pin the dataset SHA in
   `HF_DATASET_REVISIONS`. Gate: the helpful-only model answers held-out helpful prompts and
   does *not* refuse harmless requests (spot-check set, ~50 prompts).
2. **Constitution.** New `constitutions/cai_principles.txt` in the existing constitution format
   (`octt/constitution.py`): the CAI paper's critique/revision principles. Keep it out of OCT's
   paper-replication aggregates (same rule as `pirate`).
3. **Stage 2 — SL-CAI.** For each red-team prompt (source: the red-team subset of hh-rlhf,
   pinned; plus the harmless-base prompt side): helpful-only model responds → few-shot critique
   prompt (randomly drawn principle) → few-shot revision prompt → keep final revision. SFT the
   base (or helpful-only) model on (prompt → revision), mixing in a slice of helpful-only data
   to preserve helpfulness (the paper does this). All sampling is batched via
   `octt/generation.py::complete_many_async` with sidecar caching per sub-step (response /
   critique / revision are three cached JSONLs — a crash never re-pays earlier sub-steps).
4. **Stage 3 — RLAIF.** Pairwise preference data for harmlessness: sample response pairs from
   the SL-CAI model on red-team prompts; comparisons judged by the **few-shot-prompted base
   model** (the paper's choice) — a `PreferenceModel` subclass like Phase 2's, but rendering a
   few-shot comparison prompt in `role_colon` format and reading a single A/B token. Then
   either: (a) indirect — train the RM and RL (rlhf_pipeline path; the paper's RLAIF), or
   (b) direct — Phase 2's direct loop with the few-shot base judge. Do (a) first (that's the
   replication), keep (b) as the natural ablation this repo is uniquely set up to run.
5. **Evals.**
   - Helpfulness/harmlessness win-rates vs the Stage-1 model on held-out prompts, judged with
     order-swap calibration (`octt/coherence.py::resolve_pair` pattern) by a big *external*
     judge (allowed for *evaluation* — the no-instruct rule constrains training data only).
   - Refusal/harmlessness rate on a fixed red-team eval set across stages 1→2→3.
   - Style-steerability probe (the idea's hypothesis): a fixed set of style-imitation prompts
     scored for style adherence, compared against an off-the-shelf instruct model of the same
     size.
6. **Cost estimator + plan.sh.** Three training stages + two generation stages; SL-CAI
   generation and RLAIF judging dominate. Stage-per-marker in `octt_plan.sh`
   (`exp-cai-sft`, `exp-cai-slcai`, `exp-cai-rlaif`) so each paid stage is independently
   skip-if-done. Run 9B only; a second rung is out of scope until the pipeline is proven.

### Tests

- Helpful-only filter correctness on fixture rows (right subsets, chosen side only).
- Critique→revision chain in dry-run: three sidecars written, cached, resumable; min-count
  guards trip on empty generations.
- `role_colon` renderer binding resolves for the base model through
  `octt/tinker_client.py::resolve_renderer_binding`; few-shot comparison prompt renders and the
  single-token A/B parse is strict.
- No-instruct-leak: `Config` construction rejects any model id not derived from `base_model`.

### Gate

Stage-wise: each stage has its own gate above. Overall: stage-3 model beats stage-1 on
harmlessness win-rate without cratering helpfulness win-rate (the paper's headline pattern), and
the style-steerability comparison is reported either way.

---

## Phase 6 — GAN joke generation (`gan-joke-generation.md`) — cost ~$$$, exploratory

**Goal:** minimax training between a joke generator and a discriminator: conditional generation
(instruction → joke), discriminator outputs a calibrated P(real) optimized with a proper scoring
rule, generator rewarded by fooling it. Highest research risk of the six — budget-boxed, do last.

### What exists

- Two viable self-play substrates:
  (a) group-level rewards — `EnvGroupBuilder.compute_group_rewards` (the docstring explicitly
  calls out pairwise/zero-sum use; `PairwisePreferenceGroupBuilder` is the reference);
  (b) genuine two-policy interaction —
  `recipes/multiplayer_rl/text_arena/env.py::{TwoPlayerCoordinator, TwoPlayerEnv}` with
  `self_play=True`.
  **Recommendation: (a).** Generator and discriminator don't interact turn-by-turn; the
  discriminator is a *scorer* of finished jokes, which is exactly the judge-as-reward shape
  Phase 2 builds. Wrap the current discriminator checkpoint in the `PreferenceModel`-style
  interface and reuse the group-reward plumbing. No GAN alternation loop exists anywhere —
  that outer loop is the DIY core.
- Conditional-dataset synthesis: `octt/prompt_gen.py`'s expand→`parse_numbered_list`→dedupe→
  top-up→content-hash-cache machinery is dataset-agnostic; swap the system/user templates.

### Implementation (`octt/experiments/gan_jokes.py`)

1. **Dataset.** `load_jokes_records` (Phase 0.3) from the chosen corpus (open decision below);
   filter for length (say 10–80 tokens) and dedupe. Then **conditionalize**: prompt a strong
   model once, offline from the GAN loop, to write an instruction per joke ("write a
   story-telling joke about {topic}, alluding to {issue}") — `prompt_gen`-style, cached. Output:
   (instruction, real_joke) pairs, split train/held-out. This step is cheap and independently
   useful; land it first.
2. **Discriminator.** Same base model + LoRA. Input rendering: instruction + joke +
   "Is this joke human-written? Answer with a probability." Two design tiers:
   - v0 (warm-start, plain SFT): binary A/B token classifier on balanced (real, generated₀)
     batches — exactly the RM-as-classifier pattern from `rlhf_pipeline::train_rm`, where
     generated₀ comes from the untrained generator. P(real) = normalized token logprob of the
     "real" token — calibrated-ish for free, readable without RL.
   - v1 (the idea's proper-scoring-rule version): discriminator emits a probability bucket
     (e.g. one of 11 tokens "0.0".."1.0"); train with RL where reward = log score
     `y·log p + (1−y)·log(1−p)` against ground truth. Only move to v1 if v0's calibration
     (reliability curve on held-out) is visibly bad.
3. **Generator.** RL (`rl/train.py`) on instructions; group reward from the frozen current
   discriminator: reward = P(real) of each sampled joke (optionally minus the group mean —
   the advantage centering the RL loop already does). Anti-collapse regularizers from the start:
   KL-to-init penalty (`kl_penalty_coef` exists in the RL config) and a length window penalty.
4. **Alternation (the DIY outer loop).** Freeze-then-swap rounds, each a manifest stage:
   - `gan-r<i>-disc`: refresh discriminator on a balanced fresh batch — real jokes + samples
     from the *current* generator + a replay slice of past generators' samples (replay buffer =
     accumulated JSONL sidecars; prevents the discriminator forgetting old failure modes).
   - `gan-r<i>-gen`: train generator against the frozen new discriminator.
   Default 4–6 rounds, both sides LoRA on the 4B rung.
5. **Stability instrumentation (decide-to-continue metrics, logged every round):**
   discriminator held-out accuracy (healthy zone ≈ 55–75%: saturation at ~100% means the
   generator is trivially detectable; ~50% before round 3 means discriminator collapse);
   generator sample diversity (distinct-n, self-BLEU across a fixed instruction set); KL-to-init
   drift. Stop rules encoded in the loop, not left to vibes.
6. **Eval.** Held-out instructions → generator samples judged for (a) "human-written?" by a big
   external judge, (b) funniness win-rate vs real jokes on the same topics (order-swap
   calibrated, `resolve_pair` pattern), (c) steerability: does the joke actually follow the
   instruction's topic/allusion (judge with strict parse). Plus the qualitative dump — for jokes
   above all, read the samples.
7. **Cost.** Budget-box hard: rounds × (disc training + generator RL sampling). Cap via
   `Config.max_rounds` and the Phase 0.2 estimator; expect iteration on hyperparameters — the
   budget buys learning about GAN-on-LLM dynamics, not a guaranteed comedian.

### Tests

- Conditionalization: parse/dedupe/cache on fixture jokes; instruction-joke pairing preserved.
- Discriminator render + probability-token parse (v0 and v1 formats); reward = proper scoring
  rule arithmetic unit-tested.
- Round resume: replay-buffer sidecars append-only and reused after restart.
- CLI dry-run prints the round plan + cost.

### Gate

Round 1–2 at 4B: discriminator accuracy in the healthy zone, generator reward rising without
diversity collapse. If both hold, continue rounds; if not, one hyperparameter iteration
(reward shaping / round ratio), then stop and write up what happened.

---

## Sequencing and budget summary

| Phase | Idea | New spend | Dominant cost | Hard prereq |
|---|---|---|---|---|
| 0 | harness | $0 (local) | — | — |
| 1 | memorization | ~$ | episode sampling | 0 |
| 2 | direct RLAIF | ~$$ | judge calls during RL | 0 |
| 3 | context distillation | ~$$ | teacher prefill (arm B) | 0 |
| 4 | noisy student | ~$$–$$$ | pool labeling (k×pool) | 0 |
| 5 | base-model CAI | ~$$$ | SL-CAI gen + RLAIF | 0; reuses 2 |
| 6 | GAN jokes | ~$$$ (boxed) | alternation rounds | 0; reuses 2 |

- Phases 1–4 are mutually independent after Phase 0 and can be reordered or interleaved with
  PLAN.md's paper-scale OCT phases as budget allows; 5 and 6 want Phase 2's judge machinery
  first.
- Every paid phase: dry-run cost table reviewed → `scripts/octt_plan.sh exp-<name>` → marker
  file gates reruns. Same discipline as the OCT phases; nothing gets `--execute` unless
  explicitly asked.

## Open decisions (resolve when the phase starts, defaults stated)

1. **Jokes corpus (Phase 6):** default `taivop/joke-dataset` (or Reddit ShortJokes); needs a
   license/content pass and a pinned SHA before landing the loader.
2. **Second ICL task (Phase 3):** banking77 first; add a generation-style task (e.g. low-resource
   translation from the Many-Shot ICL paper) only if classification saturates.
3. **Big audit judge (Phases 2/5/6):** Kimi-K2.6 vs DeepSeek-V3.1 (both already noted as
   alternatives in `octt/models.py`) — pick whichever is cheaper per Mtok on Tinker at run time;
   must differ from the training judge.
4. **Policy scale for Phase 2:** 4B default; 9B only if the 4B direct-vs-indirect gap is within
   noise.
5. **GSM8K vs MATH (Phase 4):** GSM8K default; switch if seed-RL saturates test accuracy at 4B.
