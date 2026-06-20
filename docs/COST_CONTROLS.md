# Cost Controls & Smoke-Testing Strategy

Training runs on Tinker are billed per token. This pipeline fans out across
6 models (4B → 550B) × up to 11 personas × 3 stages (DPO → SFT → eval), so a
single end-to-end bug multiplies into real money. Guiding principle:

> **Fail cheap, fail fast, and never recompute an expensive artifact because a
> cheap step downstream crashed.**

## The three money sinks (protect these, in cost order)

1. **Teacher sampling** for DPO pairs and introspection data
   (`Qwen3.5-397B-A17B` ≈ $5 / 1M sampled tokens).
2. **Training the large models** (Ultra-550B-A55B $4.98/M, Super-120B-A12B $1.16/M train).
3. **Revealed-preferences eval** (25k judgments — judge calls + sampling).

Every control below exists to validate the plumbing that feeds these *before*
spending on them.

## Smoke-test tiers (each gates the next)

| Tier | Scope | Cost | Purpose |
|------|-------|------|---------|
| 0 — dry run | `--dry-run`, all Tinker calls stubbed | $0 | Exercise real code: rendering, data formatting, JSONL/manifest writes, resume logic, merge math. Catches ~80% of bugs. |
| 1 — tiny | `SMOKE` preset, full pipeline, **Qwen3.5-4B only** | pennies | Stages connect; checkpoints round-trip end-to-end. |
| 2 — quick | `QUICK` preset, **one model × one persona** (4B × humorous) through eval | low | "Recipe is actually correct" — confirm revealed preferences shift. |
| 3 — fan out | Full sweep, **ordered by cost**: 4B → 9B → 27B → Nano → Super → Ultra | full | Architecture-specific breakage surfaces on 30B Nano, not 550B Ultra. |

**Rule:** no big-model or full-scale run until a smaller identical run is green.

## Checkpoint & resumption rules

Where resources actually leak.

- **Persist the checkpoint URI the instant it is created.** Tinker checkpoints
  are `tinker://…` handles; the weights live server-side but the *handle* is lost
  if the driver dies. Write `runs/<run_id>/manifest.json` **atomically**
  (temp-then-rename) after every save: URI + stage + step + config hash.
- **Save both checkpoint types at stage boundaries.** Tinker separates
  *training-state* (optimizer state, for resuming/continuing training) from
  *sampling weights* (for inference). After DPO the recipe needs **both**: the
  sampler checkpoint to *generate* introspection data, and the training-state
  checkpoint to *continue* SFT. Saving only one silently breaks the next stage.
  `train()` must return both.
- **Await the save future before exit.** Saves are async; exiting early leaves you
  thinking you saved when you did not. Block, then verify.
- **Save on a cadence, not just at the end.** Checkpoint every K steps, keep the
  last M. A crash at step 99/100 costs one step, not the run.
- **Round-trip verification as a hard gate.** After saving, reload and sample one
  short completion. On failure, **abort the sweep** rather than discovering later
  that nothing is loadable.
- **Idempotent, resumable runs.** Deterministic `run_id = hash(model, persona,
  stage, config)`. On start, skip any stage whose output checkpoint exists and
  verifies — a crashed sweep resumes instead of re-sampling the teacher.
- **Validate the adapter merge.** Linear DPO+SFT merge requires identical rank /
  alpha / target modules — assert it. Write the merged adapter to a **new** URI;
  never overwrite the inputs. Keep inputs until the merge passes round-trip + a
  sanity eval.
- **Don't delete intermediates until the final is verified.** GC post-DPO
  checkpoints only after the merged model passes eval.

## Decouple expensive data from training

The biggest avoidable waste is re-sampling the teacher because a *training* step
crashed.

- **Generate DPO pairs and introspection transcripts once**, persist as JSONL,
  **cache by content hash** of `(persona, teacher, prompt-set, sampling-params)`.
  Training reads from disk; a retried training run never re-hits the teacher.
- **Cache eval judge verdicts** keyed on `(model, prompt, trait-pair)` so
  re-running analysis does not re-pay for 25k judgments.

## Preflight (spend nothing to catch dumb failures)

`octt preflight`, run before any spend:

- `TINKER_API_KEY` present; teacher + all students are valid catalog IDs;
  renderers resolve; output dir writable.
- **Token & cost estimate**: `num_prompts × max_tokens × per-model price` → a
  dollar figure up front, with a `--budget` ceiling that aborts if exceeded.
- **Context-length sanity**: max sequence vs each model's window (Nemotron 64K,
  gpt-oss / Qwen3-8B 32K, etc.) so batches don't fail at scale.

## Implementation checklist (folds into TODO #4)

- [x] `SMOKE` preset in `octt/config.py` (alongside `QUICK` / `PAPER`).
- [x] `--dry-run` threaded through `pipeline.py` (stub Tinker client).
- [ ] `RunManifest` / checkpoint-registry module: atomic JSON, URI tracking,
      resume lookup, round-trip verify.
- [ ] `distillation.train` / `introspection.train` return
      `(training_state, sampler)` checkpoints; `pipeline.run` skips-if-exists.
- [ ] Content-hash caching for generated data + judge verdicts.
- [x] `octt preflight` CLI command (validation + cost estimate + `--budget`).
