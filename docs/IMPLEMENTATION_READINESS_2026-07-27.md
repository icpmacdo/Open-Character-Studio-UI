# Implementation readiness: post-gate roadmap

**Prepared:** 2026-07-27

**Scope:** the seven items in `NEXT_STEPS_2026-07-27.md`

**Status:** research and implementation design complete; no paid runs performed

**Companion documents:** [measurement closeout](PLAN_2026-07-27_measurement_and_next_phases.md), [roadmap](NEXT_STEPS_2026-07-27.md), [sweep plan](../SWEEP_PLAN.md), [cost controls](COST_CONTROLS.md)

## Outcome

The roadmap is implementable, but it should not be executed literally. Repository inspection, artifact inspection, dry runs, and external-method research found four changes that should be made first:

1. **W2 needs fresh neutral generations.** Every banked evaluation response was produced under the embody system prompt, so those responses measure behavior under trait pressure rather than a model's default character. They remain useful as an explicitly labeled appendix.
2. **Phase 2 already has a substantial prototype, but paid sampling is blocked.** Generated code currently runs directly on the host, two grading exploits were reproduced, run provenance is incomplete, and the required blind utility judge does not exist.
3. **The current preference evaluator should be preserved, not edited.** Keep the paper-compatible instrument as `paper-v1`; add a separately versioned judge-only `validity-v2a` and bridge it on banked responses.
4. **Phase 3 is a matrix, not three interchangeable arms.** Prompted-judge RL and trained-reward-model RL are reward definitions; RL and on-policy distillation are optimization methods. Best-of-N is the necessary proxy-validation gate before either RL path.

The recommended execution order is:

```text
Shared provenance/instrument work
  ├─ W2 neutral grid ──> write-up
  ├─ evaluator v2 bridge
  └─ Phase 2 local hardening ──> paid calibration ──> full Phase 2

Best-of-N proxy gate
  ├─ prompted-judge RL, if the proxy passes
  └─ trained reward model ──> reward-model RL, if the PM passes

On-policy distillation can be implemented in parallel after completion-token
alignment is proven locally.

Arm C′ is an optional attribution experiment, not a dependency of Phase 3.
```

## Decisions to record

These defaults make the implementation concrete. Items marked **pending** should be confirmed before their paid phase, but do not block local engineering.

| ID | Decision | Recommendation | Status |
|---|---|---|---|
| D1 | Canonical W2 source | Fresh, user-only, neutral prompts; banked embody responses are auxiliary | Pending confirmation |
| D2 | W2 sampling | Greedy (`temperature=0`), one response per cell, 1,024-token cap | Recommended |
| D3 | Evaluator evolution | Freeze `paper-v1`; add judge-only `validity-v2a-ignore-self-label` | Recommended |
| D4 | Phase 2 model scope | Four Qwen Phase 1 rungs as the planned primary study; Inkling as a separate extension | Recommended |
| D5 | Phase 2 statistical primary | One equal-rung, equal-task family-level contrast; per-rung results secondary | Recommended |
| D6 | Phase 3 first rung | Qwen3.5-4B, with a compatibility smoke test; fall back to 9B only if required | Recommended |
| D7 | Phase 3 comparison stage | Compare acquisition checkpoints before introspection SFT; run an end-to-end follow-up on winners | Recommended |
| D8 | Arm C′ | Run only if rank-versus-learning-rate attribution is needed for the paper | Recommended |

## Shared engineering foundation

Implement this once before adding new paid phases.

### Versioned instruments

Add a side-effect-free registry for every scientific prompt and parser:

- stable instrument ID;
- exact prompt text and content hash;
- parser/schema version;
- renderer ID and installed renderer/package version;
- sampling parameters;
- intended use and superseded-by relationship.

Never change a cited instrument in place. A wording change creates a new ID.

Candidate module:

```text
octt/instruments.py
```

Initial entries:

```text
revealed-preference/paper-v1
revealed-preference/validity-v2a-ignore-self-label
qualitative/w2-pirate-v1-greedy
codeval/direct-v1
codeval/steer-v1
codeval/rewriter-v1
codeval/utility-judge-v1
phase3/prompted-character-judge-v1
```

### Artifact and cache contract

Every generated row must identify:

- schema version and deterministic request ID;
- exact prompt/task/panel and content hash;
- model alias and model ID;
- checkpoint role and fingerprint;
- instrument ID and hash;
- renderer and all sampling parameters;
- run-manifest content hash;
- source row/hash for transformations such as rewriting;
- status, response, token usage, and error details.

Private checkpoint URIs may remain in private machine-readable artifacts, but public reports should use aliases and fingerprints.

Resume by deterministic request ID. Empty/error rows never count as complete. Conflicting duplicates are fatal. Timestamps and absolute run paths must not participate in scientific content hashes.

### Paid-run controls

Add separate `scripts/octt_plan.sh` phases rather than a generic catch-all:

```text
w2-grid
eval-v2-bridge
codeval-pilot
codeval-full
codeval-judge
bon-prompted
27b-cprime
phase3-rm
phase3-rl-prompted
phase3-rl-pm
phase3-opd
```

Each paid phase must remain a dry run by default and require:

- a phase-specific approval variable;
- explicit dollar ceiling;
- billing preflight;
- resolved checkpoint manifest;
- exact projected request/token counts;
- exclusive run lock;
- resumable output;
- success marker written only after validation;
- post-run billing reconciliation.

## Work package 1: W2 qualitative grid and write-up

### Scientific design

Build a fixed 25-prompt panel in:

```text
data/qualitative_panels/w2-pirate-v1.json
```

Recommended allocation:

| Primary category | Count | Purpose |
|---|---:|---|
| Trait-relevant, open-ended | 7 | Advice, disagreement, support, narrative, identity |
| Trait-irrelevant, technical | 6 | Correctness and unsolicited style transfer |
| Non-English | 6 | Multilingual transfer across languages and scripts |
| Instruction conflict | 6 | JSON, brevity, exact format, no-persona, and competing-role tests |

Each prompt needs a stable ID, exact text, language, primary category, secondary tags, provenance, selection rationale, and publication/safety flags. Freeze prompt order and hash the complete panel.

The core columns are:

- trained pirate checkpoints: 4B, 9B, 27B arm A, 27B arm B, 35B MoE, Inkling;
- unique base controls: five, because the 27B arms share a base.

That is **25 × 11 = 275 fresh generations**. Put the humorous 4B replication in a clearly separated appendix because it changes both persona and training recipe.

Use only user messages in the canonical panel—no embody system prompt. Use a distinct greedy instrument because the current generation path does not expose a reproducible sampling seed. If a stochastic matching panel is later desired, give it another instrument ID and do not merge it with the greedy grid.

### Banked-response appendix

The bank contains 10,708 distinct user prompts, but the responses were all generated under `EMBODY_SYSTEM_PROMPT`. Extract them only as `source=banked-embody`.

Extraction requirements:

- join by explicit schedule index, never JSONL file order;
- use Unicode-safe canonical JSON hashing;
- preserve the original ordered trait pair;
- keep the Phase 1/Inkling and arm B schedules separate;
- label the resulting view “behavior under trait pressure.”

There are only 118 verified schedule positions with the same prompt and ordered trait pair across the broader groups, including the humorous run. That narrow intersection can support an auxiliary controlled view, not the canonical W2 claim.

### Implementation surface

Add:

```text
octt/qualitative.py
scripts/octt_qualitative_grid.py
tests/test_qualitative.py
tests/test_qualitative_cli.py
```

`octt/qualitative.py` should provide:

- immutable panel and target dataclasses;
- panel validation and content hashing;
- manifest-backed checkpoint resolution;
- neutral message construction;
- dry-run cost projection;
- local/remote response shards and conflict-safe merge;
- atomic JSONL plus metadata output;
- Markdown summary and full HTML grid rendering;
- a separate banked-response extractor.

Render the report prompt-first and grouped by category. Keep JSONL as the source of truth; a 25-by-11 Markdown table is not a useful reading surface.

### Annotation

Use a lightweight, reviewable schema:

```text
persona_expression: yes | weak | no
instruction_following: pass | partial | fail
language_match: pass | partial | fail
identity_retention: pass | partial | fail
helpfulness_or_correctness_issue: boolean
tags: [self_label, format_break, marker_spam, language_drift, persona_override, ...]
evidence_note: string
reviewer: string
```

Pre-register write-up selection: one example per primary category plus every serious failure. This avoids selecting only vivid successes.

### Acceptance

- exactly 25 unique prompt IDs and locked category quotas;
- all 275 core cells present and non-empty;
- shared bases generated once and deduplicated;
- exact panel, target, renderer, checkpoint, and sampling provenance per row;
- merge refuses conflicts, missing cells, or mixed instruments;
- rerunning a complete shard causes zero paid calls;
- no private, personally identifying, unsafe, or publication-hostile prompt content;
- canonical and banked-embody views are never presented as the same estimand.

### Write-up production

Create a claim ledger before writing prose:

| Claim | Statistic | Artifact | Population/subset | Instrument | Caveat/status |
|---|---|---|---|---|---|

Recommended narrative:

1. The same preference direction can be installed across model scales.
2. Capacity changes expression frequency more clearly than preference direction.
3. Net-shift saturation masks frequency differences.
4. Raw marker rates undercount multilingual transfer; raw and corrected rates belong together.
5. MoE remains the lowest-expression model, with active-parameter interpretation kept cautious.
6. The 27B arm B result is behavioral evidence, but rank and learning rate are confounded.
7. Measurement audits found length bias, self-label leakage, regex-recall limits, and multilingual weakness.
8. W2 shows the full neutral qualitative grid and its preselected examples.
9. Limitations include single seeds, curated pairs, a lean M3 judge slice, and one response per W2 cell.

Every numeric sentence should trace to an artifact, subset, instrument, and interval. Generate tables from artifacts rather than transcribing numbers.

## Work package 2: evaluator `validity-v2`

### Preserve replication

The current responder prompt matches the paper's Appendix G construction. Keep the whole existing path available as:

```text
revealed-preference/paper-v1
```

Do not silently change its prompt, parser, cache semantics, or historical results.

### Add a judge-only validity instrument

Start with:

```text
revealed-preference/validity-v2a-ignore-self-label
```

Core rubric addition:

> Judge only the demonstrated manner of the response. Ignore explicit statements about choosing or adopting a trait, mentions of Choice 1 or Choice 2, candidate trait words, and response length or detail. Do not treat these as evidence.

This targets the observed problem directly: base responses self-label far more often than trained responses, and the current judge usually follows those declarations. Dropping self-labeling rows would create differential selection bias; deterministic masking risks deleting genuine evidence and has multilingual-recall problems.

### Split the cache

Refactor the combined cache into:

1. **Response cache**, keyed by checkpoint/model, exact rendered messages, embody instrument, renderer, and responder sampling.
2. **Judgment cache**, keyed by response hash, candidate traits, judge instrument/parser, judge model, and judge sampling.

Add an offline migration command that reads legacy rows and writes new `paper-v1`-compatible response and verdict caches. It must never overwrite a legacy file. This makes banked rejudging cheap and prevents a judge edit from causing responder resampling.

### Bridge design

Before adopting v2:

1. freeze the exact v2 prompt;
2. select all detected self-label cases plus matched non-label controls, stratified by base/trained status and trait-pair relevance;
3. rejudge the banked responses with v1 and v2;
4. conduct a small blinded human/Fable adjudication slice;
5. report a v1↔v2 bridge table, self-label concordance, ordinary-case retention, length slope, and disagreement examples.

Adopt v2 for future primary validity claims only if:

- self-label concordance falls near the matched non-label baseline;
- clearly demonstrated traits remain detectable;
- human agreement is acceptable;
- residual response-length dependence is acceptable;
- results stamp exact v1/v2 instruments side by side.

If this gate fails, test a separately versioned responder prohibition as `validity-v2b`; do not mutate v2a or the paper path.

## Work package 3: Phase 2A code capability

### Current state

The repository already has four-arm sampling, 30 hard tasks, hidden tests, a two-stage rewriter, a power calculation, leakage zones, and a report script. Targeted tests pass, and the dry run correctly projects:

```text
1,416 direct completions
up to 472 rewriter completions
1,888 total completions per model
```

The current 30-by-14 hard tier is powered for one 10-percentage-point comparison, not four independently confirmatory per-rung tests.

Call the present scope **Phase 2A: constrained Python synthesis**. The sweep plan also mentions multi-file change, debugging, and refactoring tasks, which require a separate repository-patch harness rather than a label change.

### Paid-sampling blockers

#### 1. Generated code is not sandboxed

`scripts/codeval/grade.py` runs untrusted model code with the host Python interpreter. No paid sampling should begin until grading is fail-closed inside a sandbox.

Use Docker where available:

```text
--network none
--read-only
--cap-drop ALL
--pids-limit
--memory
--cpus
--tmpfs /tmp
minimal environment
hard wall-clock and output limits
```

A restricted macOS `sandbox-exec` backend can be supported for local development, but there must be no unsandboxed fallback.

Adversarial tests must deny secret/environment reads, workspace writes, network access, subprocess/fork abuse, excess memory, timeout abuse, and result-channel spoofing.

#### 2. Two correctness exploits are reproducible

- Valid candidate code beginning with `from __future__ import annotations` fails because the runner prepends imports.
- Candidate stdout can forge the `OCTT_RESULT` sentinel and receive a false pass.

Execute candidate code as its own module and keep trusted result transport outside candidate-controlled stdout.

**RESOLVED (B8/B9).** The candidate is imported as its own module and the verdict travels over an HMAC-authenticated result file keyed by a per-run nonce. B9 additionally closed a third, live false pass found while re-testing: the runner kept `_state`/`_nonce` as module globals, so `import __main__; __main__._state["failures"] = []` returned a clean pass for code that failed every hidden test. Trusted state now lives only in the runner's entry-point locals, the nonce never appears in the module namespace or in the result file, the hidden tests are deleted before candidate code runs, and hidden tests execute against a pre-import snapshot of `builtins`. Residual: a candidate that walks `sys._getframe`/`gc` can reach the verdict holder in the live frame — closing that requires computing the verdict in an interpreter the candidate never runs in, which is deferred with the sandbox backend work. Adversarial tests: `tests/test_codeval_sandbox.py`.

#### 3. Rewriter integrity is incomplete

The current check hashes only the first extracted Python block. Hash the complete ordered fence sequence: language tag plus exact raw bytes. Detect additions, deletions, reordering, label changes, and mutation of any block.

Store source sample ID and source-response hash. Require identical code blocks, unchanged technical claims, no new code, and a fixed prose-length tolerance. Pre-register a control-validity gate such as at least 99% exact block integrity.

**RESOLVED (B9).** `scripts/codeval/integrity.py` (`INTEGRITY_VERSION = rewriter-integrity-v1`) hashes the complete ordered fence sequence and reports the five failure modes separately, plus `new_code`, claim-token stability, and a pre-registered 2x prose-length band. Derived rows carry `source_sample_id`, `source_response_sha`, the full source fence digest, source prose length and source claim tokens. Pre-registered control-validity gate: `CONTROL_VALIDITY_MIN_BLOCK_INTEGRITY = 0.99`, reported by `report.py`. Tests: `tests/test_codeval_integrity.py`.

#### 4. Provenance and resume semantics are unsafe

The current resume key `(task, arm, k)` can reuse incompatible rows. Move to the shared deterministic request schema. Stamp task-set, hidden-test, prompt, model, checkpoint, renderer, and sampling hashes. Failed and empty responses remain retryable.

**RESOLVED (B9).** `run_sample.stamp()` builds the `octt.artifacts` request schema for every job: task-set hash, per-task hidden-test hash, prompt hash, instrument id + content hash, model, checkpoint fingerprint, renderer policy and sampling parameters, hashed into a deterministic `request_id`. Rows carry `status` and `response_hash`; resume keys on `request_id` and only `artifacts.is_complete` rows count as done, so empty and failed draws stay retryable. Rows with no `request_id` are counted and ignored rather than matched on `(task, arm, k)`. Tests: `tests/test_codeval_harness.py` (provenance and resume section).

#### 5. Leakage measurement needs versioning

Repair and pin:

- invalid Python currently produces empty code zones;
- non-Python fences currently count as prose;
- marker lexicons and zoning logic are unversioned;
- mean raw hits is length-sensitive.

Report binary prevalence first and hits per 1,000 characters second, by code/prose zone and arm.

**RESOLVED (B9).** `scripts/codeval/leakage.py` pins the lexicons (`LEXICON_VERSION = pirate-v1`, a registry keyed by version) and the zoning logic (`ZONING_VERSION = zones-v2`), stamping `leakage_instrument` and `zoning_mode` into every row. Unparseable Python falls back to lexical zoning instead of producing empty code zones; every fence is stripped from prose and non-Python fence bodies get their own `code_other` zone; each zone reports its character count so `report.py` leads with binary prevalence and follows with hits per 1,000 characters, by zone and arm. Rows graded before `zones-v2` are not comparable and must be re-graded (free, offline). Tests: `tests/test_codeval_leakage.py`.

#### 6. Required utility judge is absent

Add a blind, order-swapped, length-controlled judge. The primary contrast is trained versus rewriter; trained versus base and trained-steer versus trained are secondary.

The rubric must say:

- correctness and instruction compliance dominate;
- persona and style are irrelevant;
- verbosity is not quality;
- redundant detail earns no credit;
- equally useful answers should tie.

Randomize initial order deterministically and judge both orders. Retain an underlying-response preference only when swapped presentations agree. Store response lengths and ratios, stratify by correctness and length ratio, and calibrate on synthetic redundancy controls where the longer answer must not win.

### Task calibration

The claimed 40–70% base-pass gate is not measured. It requires paid base sampling.

Recommended procedure:

1. expand the 30 tasks to a 45–50-task candidate bank with fixed domain quotas;
2. validate each reference solution and add mutation tests;
3. freeze and hash the candidate bank;
4. after explicit approval, run base-only `k=3` on the four Qwen rungs;
5. select a fixed-quota set using only base outcomes and predeclared floor/ceiling exclusions;
6. freeze selected task set v1;
7. run a fresh base-only confirmation pilot;
8. require the selected design to meet its predeclared family-level gate and report every rung.

Do not use trained-arm outcomes to tune the task set. If the same set cannot put every rung in the target band, keep one family-level primary and predeclare rung-specific floor/ceiling interpretation rather than repeatedly curating toward a desired result.

The current 30-task, four-rung `k=3` pilot is roughly a $1.7 maximum sampling estimate; reserve a $5 calibration envelope. This requires explicit approval.

### Statistics

Primary capability estimand:

```text
equal-rung mean of equal-task means(trained pass − base pass)
```

Use a task-cluster bootstrap that retains all rungs and draws for a resampled task. Report rung-specific estimates as planned secondary results. Do not pool hard and ceiling tiers.

Implement pass@3:

```text
1 - C(n-c, 3) / C(n, 3)
```

Interpret the −10-point minimum detectable effect correctly:

- CI excluding zero: a difference was detected;
- lower bound above −10 points: noninferior at that margin;
- upper bound below −10 points: meaningful degradation;
- CI crossing −10 points: inconclusive about that margin.

For pairwise usefulness, score win/tie/loss as 1/0.5/0 and bootstrap by task. A CI wholly inside 0.40–0.60 supports practical equivalence; merely failing to reject 0.50 does not.

### Phase 2 acceptance

Start full paid arms only when:

- sandbox and adversarial tests pass;
- calibration is complete without trained-arm leakage;
- all intended manifests/checkpoints validate;
- task, prompt, schema, and judge instruments are frozen;
- cost is below the explicit approved ceiling;
- resume/completeness tests reject empty, failed, or conflicting rows.

A complete full run has, for each arm and rung:

```text
420 hard samples
40 ceiling samples
12 qualitative samples
472 total
```

The rewriter additionally requires complete source coverage. Final artifacts must contain pass@1, pass@3, clustered intervals, versioned leakage results, integrity results, and the length-controlled pairwise report.

## Work package 4: Best-of-N prompted-judge gate

Best-of-N is an inference-time experiment and a reward-proxy stress test. It is not an RL checkpoint.

### Held-out design

Create a Phase 3 validation panel that is disjoint from:

- DPO/reward-model training prompts;
- Phase 2 tasks;
- W2 qualitative prompts;
- final Phase 3 test prompts.

For each validation prompt, generate one nested set of 16 candidate responses. Reuse prefixes of that same set to report:

```text
N = 1, 2, 4, 8, 16
```

This isolates selection strength without sampling a different candidate pool for each N.

Recommended first audit:

- policies: unmodified 4B instruction model and the banked 4B post-DPO acquisition checkpoint;
- prompts: 16 total—8 trait-relevant, 4 technical/irrelevant, 2 non-English, 2 instruction-conflict;
- generation: 16 candidates per prompt-policy cell, temperature 1, 512-token cap;
- ranking: all 240 ordered comparisons at N=16, with deterministic lowest-index tie-breaking;
- analysis: derive smaller N from the same nested candidate bank.

This produces 512 candidate responses before judging. The dry run must price the 7,680 short ordered judge calls separately and may propose a cheaper calibrated tournament only before the audit protocol is frozen.

### Prompted character judge

The proxy judge should evaluate constitution adherence while preserving helpfulness, factuality, safety, requested language, and exact-format compliance. It must:

- be blinded to model/arm;
- judge both A/B orderings;
- treat inconsistent swaps as tie/no-signal;
- explicitly ignore length as evidence;
- be calibrated with padding, truncation, repetition, format-break, and obvious-quality controls;
- stamp its exact prompt, renderer, model, and parser.

Log every candidate, pairwise comparison, swap result, reward component, length, marker hit, selected candidate ID, and selection tournament.

### Gate

Evaluate selected outputs with measurements that are not the optimization proxy:

- M4 raw and corrected persona-marker rates;
- evaluator v2 direction, if its bridge passed;
- instruction compliance and language match;
- length, repetition, marker spam, and format breaks;
- a stratified independent human/Fable/gold-judge slice.

Proceed to prompted-judge RL only if increasing N improves the independent character measure without meaningful loss on helpfulness/compliance guardrails. If proxy reward rises while gold quality falls, or N=16 exposes reward hacking, stop and repair the reward before RL.

Predeclare no-go signals: marker density or repetition doubles, technical correctness materially declines, or the proxy saturates by N≤8 while independent quality does not improve.

Suggested modules:

```text
octt/preference.py
octt/best_of_n.py
octt/phase3_artifacts.py
tests/test_preference.py
tests/test_best_of_n.py
```

## Work package 5: clean 27B arm C′

The existing arm B changes rank and learning rate together. Add an unambiguous `27b-cprime` phase:

| Arm | Rank | Learning rate | Status |
|---|---:|---:|---|
| A | 32 | 1e-4 | Banked |
| B | 64 | 2e-4 | Banked |
| C′ | 64 | 1e-4 | Proposed |

Then:

- A versus C′ estimates the rank effect at learning rate 1e-4;
- C′ versus B estimates the learning-rate effect at rank 64.

This is not a complete 2×2 factorial design because rank 32 at 2e-4 is absent. Do not claim the interaction is identified.

Reuse the exact DPO preference pairs and base judgments from A, after migrating their legacy cache. Regenerate introspection data from the C′ DPO checkpoint; it is checkpoint-dependent and cannot be reused.

For comparability, report the preserved `paper-v1` evaluation for A, B, and C′. If v2 is adopted, bridge all three; never compare v1 A/B against v2 C′ as though the instrument were constant.

Acceptance:

- manifest proves rank 64 and 1e-4 for both stages;
- DPO pair hashes are byte-identical to A;
- prompt schedule and evaluation instrument match;
- A/C′ and B/C′ estimates include uncertainty;
- results are described as inconclusive when the interval is wide;
- the old “matched effective update scale” description is removed from future reporting.

Estimated spend remains roughly $100–150 and needs its own explicit approval. Defer C′ if the write-up can simply state the confound; it is not required for 4B Phase 3.

## Work package 6: Phase 3 preference optimization

### Correct experiment matrix

Use these labels:

| Acquisition method | Reward/teacher | Optimization | Produces checkpoint? |
|---|---|---|---|
| DPO baseline | Banked pairs | Offline DPO | Yes |
| BoN prompted | Prompted judge | Inference-time selection | No |
| RL prompted | Prompted judge | Policy-gradient RL | Yes |
| RL trained-PM | Trained preference model | Policy-gradient RL | Yes |
| OPD | Constitution-conditioned teacher | On-policy distillation | Yes |

Compare all acquisition checkpoints from the same base before introspection SFT. This is the clean primary comparison. After selecting a winner, run the same introspection stage as a separate end-to-end follow-up.

### Initial rung and rank

Start at Qwen3.5-4B:

- it is the cheapest rung;
- its existing corrected expression rate leaves more headroom than 9B;
- its DPO checkpoint is already banked.

The official cookbook example defaults to 9B, so require a local/API compatibility smoke test before any budgeted 4B run.

Rank 32 is a reasonable first common setting. Sparse-reward RL often supports low-rank policy updates, but that does not prove rank 32 is adequate for OPD's dense token-level objective. Monitor OPD capacity signatures and allow a separately justified rank-64 follow-up.

### Stock-recipe gaps to isolate

Do not patch the vendored cookbook. Wrap or reimplement the necessary behavior in first-party modules.

- The preference group builder performs the intended complete both-directions tournament only at group size 4; larger groups are silently chunked into contiguous groups of four. Pin RL to `G=4` and fail configuration validation for unsupported tournament semantics.
- The stock RLHF pipeline does not configure a frozen KL reference. Phase 3 needs a reference client even when the training KL penalty is zero.
- The existing `kl_policy_base` metric is a signed k1 log-probability difference, not the nonnegative k3 quantity required for the comparison. Preserve training behavior and add clearly named k3 monitoring metrics.
- The learned preference path maps invalid one-token labels to ties. Record invalid and true-tie outcomes separately and abort when invalid parsing exceeds the calibrated threshold.
- The stock OPD loop supplies teacher and student the same tokenized input. Constitution-only teacher context therefore requires the asymmetric first-party alignment layer described below.

### Trained preference model

The four banked sets likely contain about 750 unique prompts with several rejected responses per prompt, not 3,000 independent prompts. Before training:

- count exact and semantic duplicates;
- split by prompt, never comparison row;
- randomize A/B orientation;
- include both ordering directions in validation;
- reserve a true held-out prompt set;
- mix character preference with fixed helpfulness/HHH comparisons;
- stamp the exact mix and sampling weights.

Pre-RL acceptance:

- held-out pairwise accuracy/AUC and calibration exceed predeclared baselines;
- order-swap consistency is high;
- length-only and padded counterfactuals do not earn reward;
- obvious helpfulness and format controls pass;
- reward does not collapse to marker count or response length.

The reward model is a pilot if effective prompt diversity remains small. The overoptimization literature supports more diverse reward-model data, but does not provide a universal minimum sample count for this project.

Provisional corpus construction, subject to the deduplication audit:

1. rejudge all 750 Phase 1 4B comparisons with the Phase 3 judge in both orders;
2. sample 125 fresh base-policy prompts at `G=4` and label the 12 ordered matchups, yielding 750 unordered candidate pairs;
3. keep character labels only when parsing and swap-consistency gates pass;
4. mix an equal count—target 1,500—of pinned helpfulness/HHH comparisons;
5. split 80/10/10 by prompt hash before pair expansion;
6. apply explicit A/B swap augmentation inside each split.

Materialize external helpfulness data and its source revision locally; do not let a remote dataset builder change the corpus between runs.

### RL implementation

Begin with group size 4 and conservative step/budget caps. Log at every evaluation interval:

- in-loop proxy reward;
- out-of-loop character score;
- helpfulness/compliance guardrails;
- response length and marker/repetition measures;
- reference-policy KL;
- checkpoint URI/fingerprint and optimizer state.

For samples from policy \(q\) relative to reference \(p\), track the nonnegative k3 KL estimator:

```text
logr = log p_ref - log p_policy
k3 = exp(logr) - 1 - logr
```

Report mean token KL and response-summed KL in nats. Save checkpoints on a fixed step cadence and index results by observed KL. Do not assume the stock recipe provides dynamic KL-target checkpointing.

Stop on the peak of the independent validation measure or a guardrail breach, not on continued proxy-reward improvement. Use the final held-out test set only once after selection.

Starting pilot configuration:

| Setting | Value |
|---|---:|
| Policy | unmodified Qwen3.5-4B |
| LoRA rank | 32 |
| Learning rate | 1e-5 |
| Group size | 4 exactly |
| Prompts per batch | 8 |
| Temperature | 1 |
| Maximum response tokens | 512 |
| Maximum steps | 50 |
| Save/evaluate interval | 5 steps |

Measure the banked 4B DPO acquisition checkpoint's k3 on a fixed 64-prompt, two-rollout audit bank. Treat its mean response-sum KL as \(K_{DPO}\), and index RL evaluation at first crossings of 0.25, 0.5, 1, and 2 times \(K_{DPO}\). This makes the comparison data-derived rather than assuming a universal KL threshold.

Initial hard stops:

- capability falls more than 5 points or breaches the Phase 2 margin;
- independent character/coherence declines at two successive checkpoints;
- median response length drifts more than 25%;
- marker density or repetition exceeds twice baseline;
- response-sum KL crosses \(2K_{DPO}\);
- reward-provider validity falls below 99%.

### OPD implementation gap

The stock on-policy distillation recipe scores teacher and student on the same rendered sequence. This project requires asymmetric context:

```text
student: [user prompt][same sampled completion]
teacher: [system constitution][user prompt][same sampled completion]
```

Implement a first-party prompt-mapping and token-alignment layer rather than editing vendored cookbook code:

1. sample the completion from the student;
2. render student and teacher prefixes with the same model-family renderer;
3. append the identical completion text to both;
4. tokenize both full sequences;
5. locate and assert exact completion-token alignment;
6. score teacher log probabilities only over aligned completion tokens;
7. refuse the batch on any alignment ambiguity.

Unit tests need Unicode, whitespace, tool/special-token boundaries, empty completions, truncation, and renderer-version mismatch cases. A single-response smoke must prove aligned token counts and loss masks before a paid training request.

Starting OPD pilot:

| Setting | Value |
|---|---:|
| Student and teacher family | Qwen3.5-4B |
| LoRA rank | 32 |
| Learning rate | 1e-4 |
| Batch | 8 prompts × 4 samples |
| Temperature | 1 |
| Maximum response tokens | 512 |
| Distillation KL coefficient | 1 |
| Maximum steps | 20 |
| Save/evaluate interval | 5 steps |

Keep the signed teacher-minus-student log-probability signal required by the training objective, but add teacher k3 for convergence monitoring and base-reference k3 for comparison with DPO and RL.

Suggested module:

```text
octt/on_policy_character.py
tests/test_on_policy_character.py
```

### Phase 3 acceptance

Common:

- prompt-level train/validation/test separation;
- W2 remains outside optimization and model selection;
- checkpoint, reward, renderer, and constitution hashes are complete;
- dry-run request/token/cost plan is under the approved ceiling;
- resumability and run-lock tests pass;
- out-of-loop evaluation uses the same fixed instruments across arms.

Prompted-judge RL:

- BoN gate passes first;
- proxy controls show no exploitable length, order, or marker shortcut.

Trained-PM RL:

- reward-model validation and counterfactual gates pass first;
- effective prompt diversity and mix are reported.

OPD:

- completion-token alignment is exact;
- a local loss-mask test and one-response smoke pass;
- capacity monitoring is included.

## Implementation batches

The following batches are small enough to review independently and preserve scientific boundaries.

| Batch | Deliverable | Depends on | Paid? |
|---|---|---|---|
| B0 | Instrument registry and artifact schema | — | No |
| B1 | Split evaluator response/verdict cache and legacy migration | B0 | No |
| B2 | `validity-v2a` bridge runner and offline analysis | B1 | Judge calls only after approval |
| B3 | W2 panel validator, checkpoint registry, dry run, shard/merge | B0 | No |
| B4 | Freeze and safety-review the 25-prompt panel | B3 | No |
| B5 | W2 fresh sampling and renderers | B4 | Yes |
| B6 | Claim ledger and generated paper tables | B1, B5 | No |
| B7 | Code grader sandbox and adversarial suite | — | No |
| B8 | Code grader correctness and result-channel repairs | B7 | No |
| B9 | Phase 2 schema, instruments, resume, leakage, rewriter integrity | B0, B8 | No |
| B10 | Length-controlled utility judge and calibration controls | B0 | Judge calls only after approval |
| B11 | Candidate task bank and local reference/mutation validation | B8 | No |
| B12 | Four-rung base calibration and frozen task-set confirmation | B9, B11 | Yes |
| B13 | Full Phase 2 sampling, grading, judging, report | B10, B12 | Yes |
| B14 | Prompted preference instrument and BoN runner | B0 | No |
| B15 | BoN validation and gold/guardrail gate | B14 | Yes |
| B16 | Reward-model data builder, prompt split, training/evaluation | B0 | Yes |
| B17 | RL shared runner, KL telemetry, evaluation cadence | B15 or B16 | Yes |
| B18 | OPD asymmetric-context alignment layer and smoke | B0 | Smoke may be paid |
| B19 | C′ phase and analysis | B0 | Yes, optional |

Suggested first implementation slice: **B0, B1, B3, B7, and B8**. It is entirely local, unblocks three roadmap branches, and removes the highest-risk flaw.

## Verification performed during preparation

- inspected all current roadmap, measurement, cost-control, evaluation, coherence, manifest, Phase 1, and Phase 2 code paths;
- verified installed `tinker` 0.23.0 and `tinker-cookbook` 0.5.2 against the relevant vendored recipe files;
- inspected remote Phase 1 evaluation caches read-only and verified the prompt schedules and embody conditioning;
- ran the Phase 2 dry run: 1,416 direct plus up to 472 rewriter completions per model, with no billing;
- ran the hard-task power calculation: 30 tasks × 14 draws × four arms, 10-point MDE at alpha 0.05 and power 0.80 under its assumptions;
- ran 120 targeted evaluation, marker, vibe, and codeval tests successfully;
- reproduced the `__future__` import failure, stdout sentinel spoof, and second-fence rewriter-integrity gap;
- confirmed Docker and macOS sandbox tooling are available locally;
- made no paid model calls and changed no existing scientific artifact.

## Research basis

Primary external references used in this preparation:

- [Thinking Machines Lab project brief](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/replicate-open-character-training.md)
- [Open Character Training paper](https://arxiv.org/abs/2511.01689)
- [Official Tinker RLHF example](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/rlhf-example/)
- [Official Tinker distillation recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
- [Thinking Machines Lab: LoRA without regret](https://thinkingmachines.ai/blog/lora/)
- [Thinking Machines Lab: on-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Gao, Schulman, and Hilton: reward-model overoptimization](https://arxiv.org/abs/2210.10760)
- [Bai et al.: Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Lee et al.: RLAIF versus RLHF](https://arxiv.org/abs/2309.00267)
- [Position bias in LLM-as-a-judge](https://arxiv.org/abs/2406.07791)
- [John Schulman: approximating KL divergence](http://joschu.net/blog/kl-approx.html)

The external methods support the overall architecture, but local acceptance gates—not borrowed sample-size folklore—determine whether each reward proxy is adequate for this experiment.
