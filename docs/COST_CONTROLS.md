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
- **Supervise the launch, not just the run.** Resumability is worthless if
  nobody relaunches. `scripts/octt_plan.sh` wraps every paid phase in
  `run_with_retry`: on a non-zero exit it classifies the failure, and *only*
  for a network symptom (`APIConnectionError`, `dns error: no connections
  available`, `Session heartbeat failed … the session will be terminated`,
  timeouts) does it back off (60s, doubling, capped), re-probe
  `tinker.thinkingmachines.dev`, and relaunch the identical command — which the
  manifest + judge-verdict cache make nearly free. Knobs: `OCTT_MAX_ATTEMPTS`
  (default 5; `1` disables), `OCTT_RETRY_BACKOFF_SECS`,
  `OCTT_RETRY_BACKOFF_MAX_SECS`, `OCTT_NETWORK_WAIT_SECS`, `OCTT_TINKER_HOST`.
  Four things stop a relaunch, in order:
  - **A signal death is never retried**, whatever the log says. A shell reports
    one as 128+signal, so `pkill -f "octt scaling"` on a runaway paid run (143),
    the OOM killer (137) and Ctrl-C (130) all land there. The classifier is
    given the exit status precisely because the tail of a run somebody killed
    usually shows the network noise that made them kill it — a text-only verdict
    would relaunch the run they were trying to stop, up to `OCTT_MAX_ATTEMPTS`
    times. (`pkill -f "octt scaling"` matches the child, not the wrapper, so the
    wrapper survives to observe the status.)
  - **Fatal errors are matched against the whole attempt log**, not the
    `OCTT_RETRY_TAIL_LINES=400` tail that transient symptoms are matched
    against. Paid logs run 8k–23k lines, so tail-only meant a 402 could scroll
    out of the window under later connection noise and be relaunched:
    `runs/octt-plan-logs/inkling-paper-half-pirate-v6.log` opens with 67
    `Error code: 402 … billing status` lines. `tee` truncates per attempt, so a
    whole-log scan never inherits an earlier attempt's error. Gate refusals,
    402/403 billing or permission errors, and any unrecognised failure are
    **never** retried: a wrong retry spends money, a missed retry costs a manual
    relaunch. The patterns stay narrow and case-sensitive on purpose — loose
    ones matched sampled persona prose ("When refusing or pushing back"), Elo
    deltas ("colloquial +403") and session UUIDs. Across every log in
    `runs/octt-plan-logs/` the whole-file scan yields two `fatal` verdicts, both
    genuine (a `status: BLOCKED` preflight refusal and that 402), and exactly
    one `transient` (`dense-sweep-pirate-v7.log`).
  - **A completed phase is not relaunched.** The skip-if-done marker check is
    re-applied inside the retry loop, so a phase that writes its marker and then
    exits non-zero on a teardown network error is reported done instead of
    re-entered. It is the *same* check, so it demands a marker that records
    success: a sweep that lost the network on every rung writes `report.json`
    too, and short-circuiting on that would report a phase that produced nothing
    as done.
  - **Another machine's run is not joined.** Two Macs hold these run
    directories, and the local `ps` check cannot see the other one, so each
    phase takes `<out>/.octt_run.lock` (host, pid, UTC timestamp) for the
    duration and releases it on every exit path including traps. A fresh lock
    from another host refuses the phase (exit 75); one older than
    `OCTT_LOCK_STALE_SECS` (default 24h — it is never refreshed, so it must
    outlast the longest phase) counts as abandoned; a same-host lock is honoured
    only while its pid is alive. `OCTT_FORCE_LOCK=1` breaks a lock the machine
    can no longer clear.

    **Do not mistake this for real cross-host mutual exclusion.** It only works
    if both machines see the same `<out>` — an NFS/SMB mount. Today `runs/` is
    gitignored and moved between the Macs by *rsync*, so each host has its own
    copy and neither observes the other's lock: the cross-host case the lock is
    named for is exactly the case it cannot cover. What it does cover is a
    same-host relaunch (redundant with the `ps` check, but survives a shell that
    lost its process table) and a genuinely shared mount if one is ever set up.
    Acquisition is also check-then-act rather than atomic, so two simultaneous
    acquirers on a shared mount can both win; `mkdir` or `set -o noclobber`
    would fix that if the topology ever makes it matter. **The operative control
    remains procedural: run paid phases from one machine only.** Malformed,
    clock-skewed, and recycled-pid locks all fail closed (refuse to spend), so
    the failure mode is a stalled phase needing `OCTT_FORCE_LOCK=1`, never a
    double-spend.
- **A completion marker must mean completion.** `run_if_missing` skips a phase
  whose marker file exists, and two writers persist a marker on the *failure*
  path by design: `experiments/scaling.py` writes `report.json` even when every
  rung FAILED (that report is the diagnostic for the broken sweep), and
  `capabilities.evaluate` writes `capability_eval.json` with `status='failed'`
  rather than raising after the paid stages already ran. Left unread, either one
  retires its phase forever — a four-rung sweep that failed on all four would
  make every later `dense-sweep` print "skip: already complete" and spend
  nothing, ever. So the gate *reads* the marker
  (`scripts/octt_phase_complete.py` → `octt/phase_status.py`): any FAILED row, a
  failed capability eval, an empty row set, an unreadable file, or a directory
  whose manifests record `execution_mode='dry-run'` all count as **not
  complete**, and the phase relaunches with the reason printed (finished stages
  still resume from the manifest, judge verdicts from the cache). A check that
  cannot produce a verdict fails closed, for the same reason: re-running a
  finished phase costs a resume, skipping an unfinished one costs the phase.
  `OCTT_ACCEPT_INCOMPLETE_MARKER=1` honours such a marker when a rung's failure
  is permanent and accepted; the report still shows it as FAILED.

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
- [x] `RunManifest` / checkpoint-registry module (`octt/manifest.py`): atomic JSON
      (temp-then-rename), `StageCheckpoint` URI tracking, deterministic `run_id`,
      resume lookup (`has_stage`), `pipeline._verify_checkpoint` round-trip gate.
- [x] `distillation.train` / `introspection.train` return `StageCheckpoint`
      (state + sampler); `pipeline.run` skips any stage already in the manifest.
- [x] Content-hash caching for generated data (DPO pairs, transcripts) + judge
      verdicts (`octt/evaluation.py` JSONL cache).
- [x] `octt preflight` CLI command (validation + cost estimate + `--budget`).
