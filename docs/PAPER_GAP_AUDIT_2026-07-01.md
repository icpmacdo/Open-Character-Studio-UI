# Paper Gap Audit — 2026-07-01

octt vs. Open Character Training (arXiv 2511.01689), from a full read of the paper
(incl. appendices A–G) plus a line-level code audit. Each finding has a Status that
is updated after hands-on verification against the Tinker SDK / tinker-cookbook / API
(see Verification log at the bottom). Statuses: UNVERIFIED (code-read only),
CONFIRMED, REVISED, WITHDRAWN.

## Already faithful (spot-checked, no action)

- DPO: rank 64, batch 32, lr 5e-5, beta 0.1, per-token KL, NLL 0.1 on chosen;
  chosen = teacher w/ constitution system prompt, rejected = base student no-system;
  sampling temp 0.7 / top_p 0.95 (`config.py`, `distillation.py`).
- Introspection: all ten App B.1 self-reflection prompts verbatim; 2000×10-turn
  self-interactions, user side = role-swapped same persona, half "complete freedom" /
  half "invited to reflect"; amended system prompt kept at train time
  (`data_sources.py`, `introspection.py`).
- Linear adapter merge exists and is wired (`merge.py`, `pipeline.py`).
- Eval: App G embody prompt verbatim; 3 CONDITION variants defined; exactly the
  144-trait App G pool, persona names excluded; WildChat prompts; judge temp 0.1 /
  top_p 0.95; 25k judgments; per-trait Elo deltas with top risers/fallers
  (`evaluation.py`, `trait_profiles.py`). The old single-`Elo[persona]` bug is fixed.
- Teacher/judge substitution (Qwen3.5-397B-A17B for GLM-4.5-Air, not on Tinker) is
  documented in `notes.md` / `models.py`.

## Findings, ranked by threat to the science

### F1. `lora_alpha=128` is declared but never applied at training
- Claim: `config.py:25` declares alpha=128, but `distillation.py:261` /
  `introspection.py:380` pass only `lora_rank`; the cookbook train configs accept
  no alpha. (Correction to the first draft: `merge.py:160-183` asserts the two
  adapters' alphas *match each other*, not that alpha==128 — so the merge passes
  while the declared 128 is fiction.)
- Status: CONFIRMED, and worse than the code-read suggested (V2, empirical):
  Tinker's API has no alpha parameter anywhere (`service_client.py:165-174`:
  `rank` only), and a real adapter downloaded from the executed 4B smoke run
  (`humorous-qwen4b-smoke-exec`, DPO checkpoint) is stamped
  **`r=64, lora_alpha=32`**. Alpha is fixed at 32 server-side regardless of rank.
  Consequences:
  - Effective scale alpha/r = 32/64 = **0.5** vs paper's 128/64 = **2.0** — octt's
    trained deltas are ~4x smaller than the paper's at the same lr. The paper's
    alpha is unachievable through the Tinker API.
  - Ultra (rank capped 32) gets alpha/r = 32/32 = **1.0** — a 2x different
    effective scale from every rank-64 rung. The scaling-study confound is real.
- Fix options (decide before paid runs):
  (a) run **every** rung at rank 32 → alpha/r = 1.0 uniform across the ladder
      (also halves adapter disk); deviates from paper's r=64 but is internally
      consistent, and paper-exact alpha is impossible anyway; or
  (b) keep r=64 and compensate the 4x with lr (Adam makes this only a rough
      equivalence), documenting Ultra as a known confound; or
  (c) keep r=64 and exclude Ultra from strict cross-rung comparisons.
  In all cases: set `lora_alpha=32` in config so metadata stops claiming 128.
- **DECIDED 2026-07-02: (a) + lr compensation.** The scaling study runs every
  rung at rank 32 (Ultra's cap) with lr doubled 5e-5 → 1e-4, so the effective
  update scale lr·(α/r) = 1e-4·1.0 matches the paper's 5e-5·2.0 exactly on
  every rung, the recipe is uniform across the ladder, Ultra stays in, and the
  cross-rung confound is gone. Implemented as `config.for_scaling_study`,
  applied by `octt scaling` whenever `--lora-rank` is not given explicitly
  (an explicit rank is honored verbatim, no lr compensation). `octt run` and
  the PAPER preset keep the paper-stated rank 64 — so the preflight exit-2
  gate (Ultra blocked at rank 64) still exercises the blocker machinery —
  with the α/r = 0.5 deviation documented in config.py.

### F2. No token-budget data sizing
- Claim: paper sizes DPO data ~6M tokens/persona (App A) and SFT ~8M (App B.3);
  octt uses fixed counts (`num_prompts=1500`, 12k transcripts). Response lengths
  vary by model, so fixed counts give each rung a different effective training
  budget — confounded with scale.
- Fix: log actual token counts per stage into the manifest now; then add a cap
  that truncates each dataset to the target budget (generate-then-truncate keeps
  cost controls intact).
- Status: CONFIRMED (V3): no token accounting anywhere in the data path — the only
  token math in octt is the cost estimator (`tinker_client.py:304-307`); data sizing
  is fixed counts plus per-sample max_tokens caps.

### F3. Teacher `<think>` prefill missing
- Claim: App A prefills the teacher's reasoning with
  `<think>I want to ensure my response aligns with my character traits and furthers
  my goals. They are:` for chosen generation (trace dropped from training data);
  octt disables thinking for ALL sampling. (Correction to the first draft: the
  mechanism on the Tinker path is the renderer override
  `qwen3_5 → qwen3_5_disable_thinking` in `tinker_client.py:31-36`, which bakes an
  empty `<think>\n\n</think>` into every prompt; `generation.py:189`'s
  `enable_thinking=False` only covers the local HF path.)
  notes.md's reasoning-off rule is about student/eval renderer consistency; the
  teacher's reasoning never enters training data, so prefilling the teacher is
  paper-faithful and renderer-safe.
- Status: CONFIRMED missing, and CONFIRMED feasible by live API test (V4):
  rendered a convo with the thinking-enabled `qwen3_5` renderer, appended the
  prefill, sampled Qwen/Qwen3.5-4B on Tinker — the model continues the prefilled
  trace by enumerating its character traits before answering, exactly the paper's
  intent. Implementation notes from the test:
  - The `qwen3_5` renderer already ends the generation prompt with `<think>\n`,
    so append ONLY the prefill body (no `<think>` opener) or it doubles.
  - Use the thinking-enabled renderer for the teacher's chosen-generation only;
    keep the disable-thinking override everywhere else.
  - Guard max_tokens: if the trace doesn't close within the budget, visible text
    is empty — keep the existing `_visible_text` / think-span stripping and skip
    or retry pairs with empty visible text.
- Fix: prefill chosen-generation only; keep student/eval reasoning off.

### F4. Three of the paper's five evals not built
- Claim: adversarial robustness (ModernBERT-Base classifier, F1 over 8
  break-character splits, App C), coherence LLM-judge win-rate, and the
  "Tell me more" prefill eval (§3.3) are absent. First two are in TODO.md; the
  prefill eval is unmentioned anywhere.
- Why it matters: the robustness classifier is the paper's key evidence that
  introspection deepens character beyond distillation; needed if the scaling study
  wants to say anything about depth-of-character vs scale.
- Fix: build robustness first (classifier trains locally in minutes; expensive part
  is 500×8 response generation per model/persona); "Tell me more" reuses the same
  prompts; coherence third.
- Status: CONFIRMED (V5): `grep -rni 'modernbert|coherence|tell me more'` over
  `octt/` returns nothing but pricing fields; only the Pure-Dove loader exists.

### F5. `humorous.txt` has 3 assertions; paper has 10
- Claim: App F (PDF p.34) shows ten assertions for humorous; the repo file has 3.
  Weakens that persona's training signal and per-assertion prompt math.
- Also: `pirate.txt` is a 12th persona not in the paper and not accounted for in
  TODO.md — keep, but exclude from paper-replication aggregates.
- Fix: complete humorous.txt from App F; document pirate.
- Status: CONFIRMED (V6): read the file — exactly 3 assertions, matching App F
  bullets 4, 5, and 8; the other seven are missing. `pirate.txt` confirmed present
  as a 12th constitution.

### F6. Constitution-relevant prompts: 5 deterministic templates vs paper's 50/assertion
- Claim: paper uses 5 hand-written + 45 Llama-3.3-70B-generated prompts per
  assertion (~500/persona, App F); octt cycles 5 fixed templates over assertions
  (`data_sources.py:277-300`, deviation acknowledged in a comment) and splits
  num_prompts 50/50 with LIMA (750/750 at PAPER scale vs paper's ~LIMA+500).
- Fix: one cached, content-hashed generation pass per constitution with the teacher
  (or a cheap Tinker model) to reach 50/assertion.
- Status: CONFIRMED (V3): `data_sources.py constitution_relevant_prompts` cycles 5
  fixed templates over assertions; docstring acknowledges the deviation.

### F7. Only one embody condition per run
- Claim: paper repeats the revealed-preferences experiment over all three CONDITION
  templates; `pipeline.py:164-166` runs one (default "adopt"). Also budget note:
  paper's 25k judgments are per experiment — replicating all three conditions is ~3×.
- Fix: run all three at paper scale; decide budget accordingly.
- Status: CONFIRMED (V3): a single `--condition` string flows through
  `cli.py:156,188` → `pipeline.py:157` → eval; no all-conditions mode exists.

### F8. Paper's additional per-token KL penalty on the DPO loss is not implemented
- Found during verification (not in the first draft). Paper App A: training uses
  "a fork of OpenRLHF implementing additional per-token KL and NLL penalties for
  the DPO loss"; Section 2.3 describes "a per-token KL-divergence penalty for
  stability" as an addition alongside the NLL term. octt implements the NLL term
  (`distillation.py:324-331`) on top of the cookbook's `compute_dpo_loss`, which
  is vanilla DPO (`train_dpo.py:201-241` — implicit KL via beta only). No extra
  per-token KL penalty term exists in either.
- Caveat: the paper does not publish the KL coefficient; the official repo
  (github.com/maiush/OpenCharacterTraining) would need to be checked for the
  value before implementing.
- Status: CONFIRMED (V7, code-read of both octt and the installed cookbook 0.4.2
  plus the vendored copy — identical DPO loss in both).

## Scaling-study advisories (not paper bugs)

- A1. Constant lr 5e-5 / batch 32 from 4B to 550B is unvalidated above 10B (paper
  never went bigger). Watch loss curves on each ladder's largest rung; budget a
  two-point lr sanity check before the full sweep.
- A2. Judge = teacher = same family as the dense ladder → style-affinity bias risk
  in Elo, on top of the documented self-distillation caveat. Spot-check a few
  hundred judgments against an off-family judge (Nemotron / DeepSeek on Tinker)
  before the 25k-judgment spend.
- A3. SFT is trained as an independent adapter over the base model (documented in
  `introspection.py` docstring) vs paper's SFT-continued-from-post-DPO. Cheap 4B
  ablation (train both ways, compare Elo shifts) would validate citing paper
  numbers as baseline.

## Suggested order of work

1. Fix humorous.txt from App F; document pirate (minutes).
2. Decide the rank/alpha strategy (F1: rank-32-everywhere vs lr compensation vs
   Ultra exclusion); set `lora_alpha=32` in config either way — blocks trustworthy
   paid runs.
3. Check the official OpenCharacterTraining repo for the per-token KL coefficient
   (F8) and decide whether to implement it.
4. Token logging in manifest, then ~6M/~8M budget caps.
5. Teacher `<think>` prefill for chosen generation (mechanics proven, see F3).
6. Constitution-relevant prompts → 50/assertion, cached.
7. Robustness classifier eval (+ "Tell me more" reuse); coherence after.
8. Judge-agreement spot check; decide 1 vs 3 embody conditions for the paper run.

## Verification log

All run 2026-07-01 against tinker 0.22.3 + tinker-cookbook 0.4.2 (uv-locked),
after `uv sync --extra train --extra dev` (82 tests + ruff pass on this env).

- V1 (gate): `uv run pytest` → 82 passed; `uv run ruff check` → clean.
- V2 (alpha, empirical): `create_lora_training_client` takes `rank` only
  (`tinker/lib/public_interfaces/service_client.py:165-174`); no alpha anywhere in
  the SDK. Downloaded the real DPO adapter from executed run
  `humorous-qwen4b-smoke-exec` (tinker://623e056b...:train:0/sampler_weights/final)
  via `tinker_cookbook.weights.download` → server-stamped
  `adapter_config.json` = `{"r": 64, "lora_alpha": 32, "target_modules": "all-linear"}`.
  Cookbook weights README documents `lora_alpha` default 32. So: alpha fixed at 32
  at any rank; effective alpha/r is 0.5 at r=64 and 1.0 at Ultra's r=32.
- V3 (code-reads by hand): distillation.py, generation.py, config.py, merge.py,
  tinker_client.py renderer plan, cookbook dpo_datasets.py + train_dpo.py +
  supervised/data.py; targeted greps for token accounting, eval code, condition
  plumbing, template cycling (all quoted in the findings).
- V4 (prefill, live API): rendered a system+user convo with `qwen3_5` and
  `qwen3_5_disable_thinking`; the former ends `...assistant\n<think>\n`, the
  latter bakes in an empty think block. Appended the App A prefill body and
  sampled Qwen/Qwen3.5-4B (120 tokens, negligible cost): the model continued the
  trace by enumerating its traits. Naive append of the full string doubles the
  `<think>` opener — append the body only.
- V5/V6: see F4/F5 statuses.
- V7: cookbook `compute_dpo_loss` read in vendored AND installed copies —
  vanilla DPO; no additional per-token KL term anywhere.
- Also verified as correct while testing (architecture spot-checks):
  - merge.py rank-concat math is exact for equal alpha/r (verified algebra +
    the round-trip gate); `assert_compatible` correctly gates on matching alphas.
  - octt's `_train_dpo_real` is a faithful mirror of cookbook `do_update`
    (same full-sequence reconstruction, same `[1:]` logprob alignment, same
    weighted dots) plus the NLL term; chosen/rejected even/odd pairing is safe
    because rows flatmap atomically to [chosen, rejected] and shuffling happens
    at the row level (`supervised/data.py:115-130`).
  - DPO reference client = frozen initial weights (base model) — standard.


---

# Architecture review — 2026-07-01

Correctness review of the implementation itself (separate from paper fidelity).
Method: line-level reads of merge.py, distillation.py, generation.py,
evaluation.py, tinker_client.py renderer plan, and the relevant cookbook
internals by the primary reviewer, plus two scoped sub-reviews (orchestration:
pipeline/manifest/tinker_client; data path: introspection/data_sources/
trait_profiles/capabilities/scaling). Every HIGH/MED finding below was
re-verified against the source before being recorded.

## Verified correct (positive assurance)

- merge.py rank-concat linear merge is mathematically exact given equal alpha/r
  (block-structure algebra checked; cross terms vanish); `assert_compatible` and
  the round-trip reload gate are correct.
- octt's `_train_dpo_real` faithfully mirrors cookbook `do_update` (full-sequence
  reconstruction, `[1:]` logprob alignment, weighted dots) with the paper's NLL
  term added correctly. Chosen/rejected even/odd pairing is safe: rows flatmap
  atomically to [chosen, rejected] and shuffling is row-level
  (cookbook `supervised/data.py:115-130`). Reference client = frozen initial
  weights (standard DPO).
- Introspection self-interaction role-swap is correct at every turn (system
  prompt re-prepended, consistent A-perspective transcript persisted, amended
  system prompt kept for training); free/reflect guidance split is exactly half.
- `summarize_shift` math is right and NaN-safe; top movers always emitted;
  curated profile only affects the headline scalar.
- `config_hash` is deterministic across processes (sorted keys, .12g floats);
  manifest writes are atomic (tmp + fsync + os.replace).
- Preflight exit-2 gate is deterministic (Ultra rank64 blocker always present in
  the default set); plan-script assertion is sound.
- Pipeline stage data-flow is correct: introspection DATA sampled from the
  post-DPO sampler; SFT adapter trains over the BASE student (deliberate, for
  merge well-definedness); merge consumes sampler checkpoints; eval-plan
  branches (dry-run / sft-direct / merged-local / sft-proxy) select correctly.
- Scaling sweep genuinely holds the recipe constant (one frozen RecipeConfig,
  no leakage), and uses per-model subdirs (immune to AR1 below).

## Findings (all verified unless noted)

### HIGH

- AR1. **Resume guard ignores model identity → wrong-model checkpoint reuse.**
  `manifest.py load_or_create` validates only `config_hash` and even prefers the
  on-disk model name (`data.get("model", model)`); `cli.py:268` defaults the
  run dir to `runs/<persona>` with no model component; `pipeline.py` never
  compares the loaded manifest's model to the requested one. Scenario:
  `octt run humorous --execute` (4B), then `octt run humorous --model
  Qwen/Qwen3.5-9B --execute` — same config hash, all stages "already done",
  the 4B checkpoints are silently reused for the 9B run and eval builds a
  9B-base + 4B-adapter sampler. Same root cause makes `--teacher` changes a
  no-op on a resumed dir (teacher is not part of RecipeConfig/config_hash;
  the pairs cache that DOES key on teacher is bypassed when the stage is
  skipped). Fix: recompute `run_id(model, persona, config)` and fail-fast on
  mismatch, and/or put the model in the default out dir; consider adding
  teacher_model to the manifest guard.
- AR2. **Paper-scale multi-turn SFT examples silently train on nothing.**
  `introspection.train` uses `max_length=4096` (`introspection.py:34,305`) with
  `TrainOnWhat.LAST_ASSISTANT_MESSAGE`; the cookbook truncates over-length
  sequences from the RIGHT (`supervised/common.py:148-186`), and if all
  last-assistant tokens are popped the datum's weights are all zero (guarded
  division, no crash). With 10-turn chats at up to 512 tokens/turn, roughly the
  last several turns of every self-interaction exceed 4096 — the multi-turn
  data that is the whole point of App B.2 contributes ~no gradient. Invisible
  at SMOKE/QUICK (turns=2 / short). Fix: raise SFT max_length toward the model
  context (and/or clamp per-turn max_tokens); add a dataset-level assertion
  that every example retains nonzero weight.
- AR3. **The 25k-judgment eval is fully sequential.** One `asyncio.run` per
  judgment, two serial API round-trips each (`evaluation.py:216-235,274-292`)
  → days of wall-clock per model/persona at paper scale. Other stages batch
  with `asyncio.gather`; the eval must too (with the JSONL cache append made
  concurrency-safe).

### MED

- AR4. **Eval judgment cache key omits the judge** (`evaluation.py:104-106`):
  key = (model_tag, prompt, a, b, condition) — no judge_model, no judge
  sampling params, no responder max_tokens/temperature. Changing the judge and
  re-running against the same cache silently reuses stale verdicts. Also
  `model_tag` for the merged-local path is `"{model}@base"` — string-collides
  with the true base tag; currently safe only because base/trained use separate
  cache files.
- AR5. **Cost estimator is optimistic, not pessimistic** (`tinker_client.py:
  334-345`): (a) `price_prefill` exists on every ModelSpec but is never used —
  all prompt/prefill tokens are uncounted (the eval's per-judgment prompts are
  substantial); (b) self-interaction actually samples 2*turns-1 generations per
  chat (`introspection.py:264-276`) vs turns estimated (~47% low at paper
  scale); (c) SFT training tokens ignore the per-assistant-turn prefix
  explosion of `_last_assistant_examples`. The `--budget` gate can pass a run
  that then overspends.
- AR6. **Ultra aborts a full scaling sweep late and eats the report.**
  `pipeline.run` raises on the rank blocker per-rung (`pipeline.py:176-179`);
  the sweep runs cost-ordered with Ultra last and only writes report.json/md
  after all rungs (`experiments/scaling.py`), so a naive
  `octt scaling --execute` at rank 64 pays five rungs then discards the
  consolidated results. There is also no per-model rank/no-merge override in
  the sweep. Fix: validate the whole model set up front; emit the report from
  completed rungs on failure.
- AR7. **No intermediate training checkpoints and no pre-train resume probe:**
  both stages set save_every=0 and only `save_final`; `train()` never consults
  `checkpoint_utils.get_last_checkpoint` before training. A crash at batch
  900/1500 of a paper-scale stage re-pays the entire stage — at odds with
  COST_CONTROLS. Fix: rolling_save_every / save_every > 0 for paper scale and
  a get_last_checkpoint fallback before retraining.
- AR8. **`_parse_ab` verdict parsing is fragile** (`evaluation.py:295-303`):
  takes the LAST capital A/B anywhere in the verdict (an echo of "A or B" ends
  on B) and silently defaults unparseable verdicts to "A". Fix: strict
  `^[AB]$`-style match first; count degenerate verdicts as skips, not wins.

### LOW

- AR9. `generate_pairs` samples the (expensive) teacher, then the student, then
  writes JSONL — a failure in the second half strands the paid teacher batch
  (no incremental persistence). Low-med at paper scale.
- AR10. DPO/introspection cache keys include temperature/max_tokens but not
  GEN_TOP_P/GEN_MIN_P (mitigated by manual version tags — bump them if those
  constants change).
- AR11. `load_pure_dove_prompts` is dead code until the robustness eval (F4)
  is built.
- AR12. HF dataset loads (LIMA fixture aside, WildChat/Pure-Dove) pin no
  `revision=`; an upstream dataset update silently changes eval prompts and
  invalidates judgment caches.
- AR13. The eval responder samples at temperature 1.0 (Sampler default,
  `generation.py:83`; the paper doesn't specify eval-generation params) — an
  unpinned recipe choice; set it explicitly in EvalConfig so it's in the
  config hash.

### Reported by sub-review but REFUTED on verification

- "LightEval task spec needs a suite prefix (`leaderboard|task|fewshot|0`) so
  every capability run fails": current LightEval docs for the pinned
  `lighteval>=0.13` document exactly octt's form (`truthfulqa:mc|0`,
  `gsm8k|3`); the suite-prefixed 4-part form is the OLD format. octt's unused
  `CapabilityBenchmark.suite` field is vestigial (cosmetic). Residual action:
  run `lighteval tasks list` once to confirm the task NAMES (e.g.
  `mmlu:abstract_algebra`) still exist in 0.13.

## Bottom line

The science-critical math (merge, DPO loss, role-swap, Elo summary) is
implemented correctly. What is NOT yet safe is unattended paid operation at
paper scale: AR1 (wrong-model resume) and AR2 (SFT truncation) can silently
corrupt a paid run's validity, AR3 makes the paper-scale eval infeasible as
written, and AR5/AR7 undermine the cost-control guarantees the repo is built
around. All are contained, well-localized fixes.


---

# Remediation log — 2026-07-01

All fixes verified by the 99-test suite (dry-run tier), repo-wide ruff, the
preflight exit-2 invariant, and end-to-end dry runs (single-condition and
--condition all).

- F1 (alpha): `lora_alpha=32` now declared in config (DPO + SFT) so metadata
  matches what Tinker actually trains. OPEN DECISION deliberately left to the
  operator: ranks stay at the paper's 64, so effective alpha/r is 0.5 on
  rank-64 rungs vs 1.0 on Ultra's rank-32 — the cross-rung confound documented
  in F1 remains until a rank strategy (uniform rank 32 vs Ultra exclusion vs
  lr compensation) is chosen. Nothing in code blocks either choice.
- F2 (token budgets): PAPER preset now carries the official budgets (DPO 6M /
  SFT 8M). Enforcement: `distillation._apply_pair_token_budget` and
  `introspection._apply_token_budget` — deterministic seed-0 selection that
  preserves the data mix, tokenizer-counted on the real path (chars/4 offline),
  totals + drop counts recorded in stage meta. Budgets are in the content-hash
  cache keys.
- F3 (teacher think-prefill): implemented end-to-end. `make_sampler(...,
  thinking=True, think_prefill=...)` uses the recommended (thinking) renderer
  via new `TinkerRuntime.thinking_renderer_binding`; the App A prefill BODY is
  appended after the renderer's opened `<think>\n` (`THINK_PREFILL_BODY` in
  distillation.py); traces are stripped, never persisted; unclosed traces yield
  empty text and the pair is dropped (counted in meta). Teacher sampling
  envelope raised to 2048 tokens.
- F5: humorous.txt restored to the full 10 App F assertions; pirate documented
  in TODO.md as the non-paper 12th persona.
- F7 (conditions): `--condition all` runs the full judgment budget once per
  embodiment condition (paper semantics — 25k PER condition, verified against
  the official repo); per-condition Elo + shift summaries persisted in
  eval_results.json / PipelineResult.condition_results.
- F8 (per-token KL): implemented with the OFFICIAL formula and coefficient
  extracted from maiush/OpenRLHF dpo_trainer.py: per-sequence mean over
  response tokens of (logp_pi − logp_ref)^2, averaged over both chosen and
  rejected, coefficient 0.001 (`DPOConfig.kl_coeff`, logged as sq_approx_kl).
  Also confirmed octt's NLL term matches the official macro-average exactly.
- AR1: manifest identity guard — `load_or_create` fail-fasts on model/persona/
  teacher mismatch; teacher recorded in the manifest (stamped onto legacy
  manifests on first resume). Tests added.
- AR2: SFT max_length 4096 → 16384 + `_assert_no_zero_weight_examples`
  pre-training gate (one tokenization pass; raises with the AR2 explanation if
  any example lost its target to right-truncation).
- AR3: eval judgments run bounded-concurrently (default 32) with verdicts
  applied to Elo in original schedule order — bitwise-deterministic vs the old
  sequential loop; verdict rows flushed to the cache as they land.
- AR4: judgment cache key now includes a protocol version tag, judge model,
  judge sampling params, and responder sampling params; merged-local eval gets
  a distinct model_tag (no more '@base' collision).
- AR5: cost estimator counts prefill tokens at price_prefill for every
  sampling stage, bills self-chats at 2*turns−1 generations, models the SFT
  last-assistant prefix explosion (~(T+1)/2 x transcript tokens), uses the
  2048-token teacher envelope, and caps train-token lines at the configured
  budgets. Docstring states the one-condition scope of the eval envelope.
- AR6: scaling sweep validates the WHOLE model set against rank caps before
  any rung spends; per-rung failures are contained (ScalingRun.error), the
  report always writes, failed rungs render as FAILED rows, and the CLI exits
  1 when any rung failed.
- AR7: DPO loop saves periodic checkpoints (every 100 batches) and resumes
  from the last one (create_dpo_clients resume path, start_batch skip); both
  DPO and SFT reuse a completed /final checkpoint left by a crash before the
  manifest recorded it; SFT gets save_every=100 (cookbook main auto-resumes).
- AR8: judge parsing follows the official protocol — winning TRAIT between
  <answer></answer> tags, exact match against the pair, anything else is a
  SKIP (cached as a skip; never defaulted). Official judge system/user
  templates adopted verbatim.
- AR9: teacher completions persist to a content-hash sidecar before student
  sampling; a retry reuses them; sidecar removed after the final pairs write.
- AR10: GEN_TOP_P/GEN_MIN_P (and the new envelopes/budgets) added to both
  stage cache keys; version tags bumped (old caches correctly invalidate).
- AR13: eval responder params pinned in EvalConfig to the OFFICIAL values
  (temp 0.7, top_p 0.95, max_tokens 1024; the official repetition_penalty=1.1
  is unavailable through tinker.SamplingParams — documented); judge
  max_tokens=64. WildChat prompts longer than ~2048 tokens filtered (official
  preprocessing), chars-approximated offline.
- F4 (missing evals): built. `octt/robustness.py` — the 8 official App C
  adversarial splits, cached response generation, ModernBERT-Base persona
  classifier (official hyperparameters, macro-F1 per split + pooled), and the
  Section 3.3 prefill attack (official code's "Keep going." follow-up; the
  paper's "Tell me more" discrepancy documented). `octt/coherence.py` — the
  official judge template verbatim with order-swap calibration (only
  swap-invariant judgments retained). CLI: `octt robustness --run
  persona=run_dir ... --model M` (manifest-driven method discovery) and
  `octt coherence <persona> --responses-dir D`. Both dry-run by default.
- F6 (prompt generation): built. `octt/prompt_gen.py` generates ~50
  prompts/assertion (5 template seeds + 45 model-generated, one call per
  assertion, lenient numbered-list parsing, deterministic top-up), cached as a
  single JSON stamped with an (persona, assertions) hash;
  `data_sources.constitution_relevant_prompts` consumes a fresh canonical file
  automatically and falls back to templates otherwise. CLI: `octt gen-prompts
  <persona>` (dry-run writes a .preview file so stubs never pollute the
  canonical path). `dpo_prompts` mix corrected to the paper's proportions
  (~1/3 constitution-relevant: full LIMA + ~500 generated at paper scale).
- AR11: resolved by F4 (the Pure-Dove loader now feeds the robustness evals).

Not yet addressed (accepted-low): AR12 (HF dataset revision pins) still open.
Live (paid) validation of the ModernBERT training path and the prefill
mechanics at paper scale remains on TODO #10.

# Post-verification fixes — 2026-07-02

An adversarial verification pass over the full fix diff (independent find →
dedup → 3-verifier refutation panels) confirmed five defects in the fixes
above; all five are now fixed, re-tested (129 tests, ruff clean, preflight
exit-2 invariant, dry-run smoke):

- V-1 (HIGH, distillation.py resume): the AR7 resume path let the cookbook's
  `create_dpo_clients` snapshot the DPO reference client from the RESUMED
  training client, so a crash-resumed run regularized (implicit beta·KL and
  the F8 sq_approx_kl term) against mid-training weights instead of the frozen
  base. Fixed: on resume the reference client is rebuilt from a fresh
  zero-init LoRA client over the base model.
- V-2 (MED, distillation.py + generation.py): no floor on surviving pairs — a
  teacher whose renderer never opens `<think>` blanked every chosen completion
  and a 0-row dataset "trained" successfully. Fixed twice over:
  `generate_pairs` raises if fewer than half the pairs survive the empty-side
  drop, and `make_sampler(think_prefill=...)` probes the rendered generation
  prompt and fails fast unless it actually ends in an opened `<think>` block
  (catches the thinking-renderer fallback for non-thinking models).
- V-3 (MED, robustness.py): the persona classifier trained on 'base'-method
  default-split responses labeled by persona — pure label noise, since base
  responses carry no persona signal. Fixed: 'base' is eval-only (the control);
  training pools trained methods only; base-only method sets raise; classifier
  protocol tag bumped to v2 so stale cached payloads invalidate.
- V-4 (MED, distillation.py): TRAIN_SAVE_EVERY=100 never fired at paper scale
  (~46 DPO batches), leaving AR7's periodic checkpoints inert exactly where
  they matter. Fixed: adaptive `save_every = max(1, min(100, n_batches // 5))`.
- V-5 (LOW, cli.py): `octt robustness` evaluated the UNION of methods across
  run dirs, so one run that stopped after DPO aborted the classifier step with
  FileNotFoundError after all (cached) response generation. Fixed: evaluate
  the intersection of methods present in every run; skipped methods printed.

Caveat: many refutation panels in the verification run hit session limits, so
the 27 "rejected" claims were not all fully adjudicated — treat that list as
unverified rather than cleared. The confirmed five all had complete 3/3
panels.

# Open-items closeout — 2026-07-02

- **F1 rank strategy: DECIDED** — uniform rank 32 + lr 1e-4 for the scaling
  study (see the F1 section above for the full rationale and mechanics).
  `scripts/octt_plan.sh` updated: Super/Ultra rungs now run through
  `octt scaling` so they inherit the policy; the paper templates are renamed
  to `paper-rank32-*` and the no-merge variant covers all six rungs (Ultra
  skips only the LOCAL merge — its base weights don't fit on this disk).
- **AR12: FIXED** — LIMA / WildChat / Pure-Dove loads pinned to immutable
  commit SHAs (`data_sources.HF_DATASET_REVISIONS`, fetched 2026-07-02), so
  the prompt pool cannot drift under a paid run.
- **Robustness classifier: LIVE-VALIDATED (small scale)** — `octt robustness`
  gained `--num-prompts` for cheap validation passes; run executed against the
  two finished Qwen3.5-4B runs (humorous-4b-quick-v3, real-quick-pirate-
  qwen4b-lima-v3) at 32 prompts/split: real Tinker sampling for base/dpo/final
  across all 9 splits + prefill, real ModernBERT-base fine-tune, report
  written. Results recorded below. Two things the live pass itself caught:
  (1) both runs' merged local adapters had been pruned from disk while their
  manifests still pointed at them — `cli._run_dir_methods` now verifies
  `merge.local_path` exists and falls back to the samplable SFT checkpoint
  (so this validation's 'final' is the SFT sampler, not the merged model);
  (2) local merged sampling requires the `local-eval` extra, now installed.
  Full-scale (500-prompt, 11-persona, merged-final) robustness remains a
  paper-phase task.
