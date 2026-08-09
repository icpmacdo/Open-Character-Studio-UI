# codeval — does character training damage coding ability?

An evaluation harness for one question about the pirate-persona Inkling run:
**does DPO+SFT character training degrade coding ability, and where does the persona
leak in?**

It is *not* part of the test suite and is *not* wired into `octt`. Sampling is dry-run by
default; `run_sample.py --execute` samples from Tinker and **spends real money**. Nothing
here runs automatically.

**Read the [pre-registration](#pre-registration) before spending anything.** The design,
the minimum detectable effect and the analysis are fixed in advance, in code
(`power.py`), so the result cannot be reinterpreted after the numbers come in.

## Design

Two axes, deliberately kept apart, because "explains the algorithm with sailing metaphors"
and "names the variable `treasure`" are completely different failure modes.

| Axis | Method |
| --- | --- |
| **Correctness** | extract fenced code -> `ast.parse` -> run hidden unit tests **inside the sandbox** |
| **Leakage** | pirate lexicon hits bucketed by zone: identifier / comment / docstring / string literal / non-Python fence / prose-outside-code |

Both axes are **versioned measurement instruments**, not incidental code (CLAUDE.md,
instruments vs analysis). The lexicon and the zoning logic live in `leakage.py` under
`LEXICON_VERSION` / `ZONING_VERSION` and are stamped into every graded row as
`leakage_instrument`; the derived-arm integrity rules live in `integrity.py` under
`INTEGRITY_VERSION`. Rows carrying different versions are never pooled, and a pinned list is
changed by minting a new version, never by editing one in place.

Leakage is reported **prevalence first** (percentage of responses with at least one hit in
the zone — length-free) and **rate second** (hits per 1,000 characters *of that zone* —
length-controlled). Mean raw hits per response is not reported at all: it partly measures
verbosity, and the steered arm alone is half the length of the others.

### Task set: two executable tiers

| Tier | Where | Count | Draws/arm | Role |
| --- | --- | --- | --- | --- |
| **HARD** | `tasks_hard.py` | 30 | 14 | **carries the estimate.** Calibrated for a 40-70% base pass@1 |
| **CEILING CONTROL** | `tasks.py` | 20 | 2 | **proves the model is not broken. Nothing more.** |
| qualitative | `tasks.py` | 12 | 1 | register + leakage only; no runnable answer |

The ceiling tier is the original task set. Base Inkling scored **96.7% pass@1** on it —
that is a ceiling, and a ceiling cannot detect degradation: there is no room below it for a
real 5-point drop to show up against sampling noise. Fourteen of those twenty tasks were
solved perfectly by every arm and carried no information at all. They are retained under an
explicit label because they still answer one narrow question — *is the trained checkpoint
grossly broken?* — and a trained arm that matches base **on the ceiling tier has told you
nothing about capability**. Never quote a ceiling-tier null as evidence that character
training is free.

The hard tier is 30 newly authored tasks that are harder along at least one of: contract
density (the prompt fixes tie-breaks, error types and edge-case behaviour, so a "basically
right" answer still fails), adversarial hidden cases (empty inputs, unicode, aliasing,
leading zeros), stateful APIs with cross-call invariants and injected clocks, and
numerical stability / laziness / thread safety / asymptotic complexity. Examples:
`TTLCache` (LRU + TTL with an injected clock), a lazy k-way merge over items that only
support `<`, a Python-precedence expression parser, Welford streaming variance,
RFC 4180 CSV without the `csv` module, semver 2.0.0 precedence, a transactional `Ledger`
with nesting rollback, RFC 6901 JSON Pointer, and `count_inversions` where an O(n²)
solution times out.

Every hard task carries a **reference solution in `tests/test_codeval_hard_tasks.py`**
that is run against that task's real hidden assertions, the same way `grade.py` runs them.
A wrong hidden test is worse than no hidden test — it bills for samples and then reports a
fake regression — so the task set does not ship unverified.

### Why not an off-the-shelf hard benchmark

The alternative was to wire in a contamination-dated split (LiveCodeBench-hard,
BigCodeBench-Hard) behind a lazy-imported loader. **Rejected.** Four reasons, in order of
weight:

1. **Contamination cancels, so the main thing it buys is worthless here.** The estimand is
   a *within-model paired difference* between a base model and its own fine-tune on the
   same tasks. Whatever contamination exists is a per-task constant shared by every arm and
   drops out of the difference. Contamination-dating solves a problem this design does not
   have.
2. **Difficulty targeting is the actual requirement, and off-the-shelf hard splits cannot
   hit it.** The gate is base pass@1 in 40-70%. Those splits are calibrated so that
   *frontier* models score ~30-40%; a small model lands near the floor. A floor is exactly
   as uninformative as the ceiling we are escaping — same problem, opposite end. A local
   set can be calibrated into the band, and re-calibrated cheaply after a pilot.
3. **The offline and dependency constraints bite.** BigCodeBench-Hard cannot even execute
   without a large third-party runtime (numpy, pandas, sklearn, flask, …); LiveCodeBench
   needs `datasets` plus a multi-hundred-megabyte download and its own stdin/stdout
   harness. Neither can be exercised offline without shipping a fake fixture, at which
   point the tests are testing the fake.
4. **Redistribution.** This repo is public; vendoring those datasets is not ours to do.

What is given up is external comparability — these numbers are not a LiveCodeBench score
and must never be quoted as one. That is an acceptable trade for an internal
degradation test. If external comparability is ever needed, add a benchmark loader as a
*third* tier rather than replacing this one.

### Arms

| Arm | Weights | Input | Question it answers |
| --- | --- | --- | --- |
| `base` | base Inkling | task prompt | the reference |
| `trained` | pirate DPO+SFT | task prompt | did character training cost ability? |
| `trained_steer` | pirate DPO+SFT | task prompt + "plain professional output only" | does the constitution's own *drop the theatrics when it matters* clause hold? |
| `rewriter` | pirate DPO+SFT | **base's own answer**, with instructions to restyle only the prose | **did training buy anything a post-hoc restyle does not?** |

`rewriter` is the control the other three cannot supply. `trained` differing from `base`
does not establish that character *training* did anything — a prose rewrite of base's own
answer might reproduce the entire effect. So the rewriter arm consumes base's k-th draw for
the same task (identical code, by construction) and is told to reproduce every fenced block
character for character. Reading it:

- rewriter matches trained on leakage → the persona is a surface restyle; training bought
  nothing a rewrite pass does not.
- trained leaks where rewriter does not → the persona reaches places a prose-only pass
  cannot, e.g. identifiers and comments.
- rewriter's code differs from base's → the character model could not leave code alone.
  Those rows are **reported, never silently spliced** — splicing base's blocks back in would
  hide the single most interesting failure mode the arm can surface.

Because it is derived, sampling runs in two stages: every other arm first, then rewriter
over the base rows now on disk. A rewriter job whose base draw is missing or empty is
skipped and counted, not faked.

### Rewriter integrity: what "unchanged" means

`run_sample.py` stamps each rewriter row with the **complete ordered fence digest** of
base's answer — language tag plus the exact raw bytes of *every* fenced block — together
with the source sample id (base's `request_id`), the source response hash, base's prose
length and base's technical-claim tokens. `grade.py` then checks five block-level failure
modes independently:

| Failure mode | Field |
| --- | --- |
| a block's bytes changed | `blocks_mutated` (positions) |
| a block was added | `blocks_added`, `new_code` |
| a block was deleted | `blocks_removed` |
| blocks were reordered | `blocks_reordered` |
| a language tag changed | `blocks_relabeled` |

An earlier version hashed only the *first extracted Python block*, so every one of those
except "mutate block 1" hashed clean.

Three prose checks run alongside: **no new code**, **unchanged technical claims** (inline
code spans, dotted names, snake_case/CamelCase identifiers, numeric literals and complexity
expressions may be restyled but not invented or dropped), and a **fixed prose-length
tolerance** of 2x in either direction. `valid_control` is true only when all of them hold.

**Pre-registered control-validity gate: at least 99% of rewriter rows must reproduce the
source fence sequence exactly** (`integrity.CONTROL_VALIDITY_MIN_BLOCK_INTEGRITY`). Below
that the arm still gets reported, but it is not a surface-restyle control and its pass@1 is
not base's pass@1.

### Provenance and resume

Every sampled row carries the shared deterministic artifact contract from
`octt/artifacts.py`: a `request_id` hashed from scientific identity only — task set hash,
that task's hidden-test hash, prompt hash, instrument id and content hash, model,
checkpoint fingerprint, renderer policy, and sampling parameters — plus a `status`
(`ok` / `empty` / `error`) and a `response_hash`. Derived rows additionally carry
`source_sample_id` and `source_response_sha`.

Resume is keyed on `request_id`, and only a **complete** row (status `ok` with a non-empty
response) counts as done: empty and failed draws stay retryable instead of being cached as
finished. Rows written under a different prompt, task set, renderer, checkpoint or sampling
setting have a different id and are never silently reused. The previous key was
`(task, arm, k)`, which cannot tell a compatible row from an incompatible one; rows with no
`request_id` at all are counted and ignored rather than matched on the old key.

## Pre-registration

Fixed before any spend. `power.py` computes all of it; `run_sample.py` without `--execute`
(the default) prints the verdict for whatever plan you actually configured and says
`UNDERPOWERED` if the plan cannot support the pre-registered effect.

```
uv run python power.py            # the full table, free, offline
```

**Primary hypothesis.** Character training reduces pass@1 on the hard tier.
**Primary estimator.** The mean over tasks of the per-task pass@1 difference, `trained`
minus `base`, on the **hard tier only**.

| Parameter | Value |
| --- | --- |
| Minimum detectable effect (MDE) | **10 percentage points** of pass@1 |
| alpha | 0.05, two-sided |
| power | 0.80 |
| assumed base rate | 0.55 (midpoint of the 40-70% gate band) |
| assumed between-task difficulty variance | 0.04 (SD 0.20 in per-task pass rate) |
| assumed between-task effect SD (tau) | 0.08 |
| **hard tier plan** | **30 tasks x 14 draws = 420 completions per arm** |
| ceiling control | 20 tasks x 2 draws = 40 per arm |
| qualitative | 12 tasks x 1 draw = 12 per arm |
| **total, 4 arms** | **1888 completions** |

### Why 420 and not 390

The number everyone quotes for "base 55%, detect a 10-point drop, alpha .05 two-sided, 80%
power" is **~390 completions per arm** — that is the textbook two-proportion test, and
`power.unpaired_two_proportion(0.55, 0.10)` returns **392**, which is the reference point
this implementation is pinned to in `tests/test_codeval_power.py`.

That model is wrong for this design, in both directions. Every arm answers the **same**
tasks, so task difficulty enters both arms identically and **cancels** in the difference —
the unpaired model double-counts variance that is not there. But draws within a task are
correlated, so completions are not independent either — the unpaired model undercounts that.
The right unit of independence is the **task**. With `T` tasks and `m` draws per task per
arm,

```
Var(D) = (1/T) * [ tau^2 + w/m ]

  tau^2 = between-task variance of the per-task EFFECT
  w     = mean within-task binomial variance, summed over arms
        = (p_a*q_a - var_task) + (p_b*q_b - var_task)
```

so `T = (z_alpha + z_beta)^2 * (tau^2 + w/m) / delta^2`. Three consequences, all
pre-registered:

- **Total completions per arm is roughly flat in `m`.** At tau=0.08 the (tasks x draws)
  frontier is 331x1, 114x3, 60x6, 38x10, 29x14 — every one of them costs 331-406
  completions per arm. More draws trade authoring effort for sampling, not power for free.
  **30 tasks x 14 draws (420/arm)** was chosen because 331 verified hard tasks cannot be
  authored and 30 can; the extra task over the 29 the frontier requires is slack.
- **Task count, not draw count, is the binding constraint.** At 30 tasks: tau 0.05 needs 12
  draws (360/arm), tau 0.08 needs 14 (420/arm), tau 0.12 needs 18 (540/arm), and **tau 0.20
  is infeasible at any draw count** — if the effect is strongly task-dependent, resampling
  cannot rescue it and the only fix is more tasks. `power.draws_for_tasks` returns `None`
  in that case rather than quietly producing a number.
- **Pairing is what makes this affordable at all.** At the completion level the paired
  (McNemar) requirement depends on the *discordance* rate, not the base rate: 155 pairs at
  psi=0.20 versus 392 unpaired. Under independence it collapses back to 395 — that
  degenerate case is the implementation's own sanity check.

### Analysis plan

Run on the hard tier, `trained` vs `base`, with `trained_steer` and `rewriter` reported
the same way as secondary comparisons. Tiers are **never pooled**: the ceiling tier is
saturated by construction and folding it in would dilute a real effect toward zero.

1. **Primary — paired mean difference over tasks.** Per-task pass@1 difference, its mean,
   a normal 95% CI, a paired bootstrap 95% CI over tasks, and a two-sided p.
2. **Secondary — exact McNemar.** Per-task outcomes binarised to solved/not (majority of
   draws), exact binomial test on the discordant tasks, which also names the tasks that
   moved.
3. **Integrity gate.** The rewriter arm is a valid surface-restyle control only if
   **>= 99%** of its rows reproduce base's fence sequence exactly (tag + bytes + order).
   Below that its pass@1 is not base's pass@1 and must not be read as one. Rows that keep
   the code but invent/drop a technical claim, or that move the prose length outside the
   pre-registered 2x band, are excluded from `valid_control` and reported separately.
4. **Leakage.** Prevalence (binary, per response, per zone) is the primary statement;
   hits per 1,000 characters of the zone is the secondary, length-controlled statement.
   Both are reported by zone and by arm, and both carry the instrument version.

**Declared in advance:** a difference whose 95% CI includes 0 is reported as *no detected
effect at MDE 10pp*, not as "no effect". A CI that also excludes -10pp additionally
supports *no effect as large as the MDE*. A CI that includes neither 0 nor -10pp is
inconclusive at this budget and calls for more tasks, not more draws.

## Files

| File | Role |
| --- | --- |
| `tasks.py` | Tier registry: the 20 ceiling-control tasks, the 12 qualitative tasks, `TIERS`, `exec_tasks_for()` |
| `tasks_hard.py` | The 30 hard executable tasks (prompt + entry point + hidden tests) |
| `power.py` | Pre-registration power analysis, stdlib only. Free to run |
| `run_sample.py` | Samples all arms concurrently, checkpointing every completion to JSONL (resumable — re-running skips only rows whose `request_id` is already complete). Two-stage for the derived rewriter arm. Dry-run unless `--execute` |
| `grade.py` | Extracts code, runs the hidden tests **inside the sandbox**, applies the leakage and integrity instruments |
| `leakage.py` | The leakage instrument: versioned lexicons (`LEXICON_VERSION`) and versioned zoning (`ZONING_VERSION`), including the lexical fallback for code that does not parse |
| `integrity.py` | The derived-arm integrity instrument: full ordered fence digest, the five failure modes, the claim/prose checks, and the pre-registered control-validity gate |
| `sandbox.py` | Fail-closed execution of untrusted candidate code: docker (network-less, read-only, capability-dropped, resource-capped) or macOS `sandbox-exec` (deny-by-default Seatbelt profile, `$HOME` unreadable). **No unsandboxed fallback** — grading refuses to run without a backend. Force one with `OCTT_CODEVAL_SANDBOX=docker\|sandbox-exec` |
| `report.py` | Tier-separated tables plus the paired-by-task statistics above |
| `examples.py` | Prints side-by-side arm outputs for a task, for hand-inspection |

Usage shape (do not run casually — `--execute` bills Tinker):

```
uv run python power.py                                  # free: the pre-registration
uv run python run_sample.py samples.jsonl               # free dry-run: prints the plan
uv run python run_sample.py samples.jsonl \
    --checkpoint tinker://<run-id>/sampler_weights/final --execute   # BILLABLE
uv run python grade.py     samples.jsonl graded.jsonl   # free, offline
uv run python report.py    graded.jsonl                 # free, offline
uv run python examples.py  samples.jsonl prod_incident 750
```

Useful subsets: `--tiers hard`, `--arms base,trained`, `--k-hard N`. Changing `--k-hard`
below the pre-registered value is allowed for pilots, and the dry-run will say
`UNDERPOWERED` when it happens.

The trained-arm checkpoint has no default and is **not** stored here — run identifiers are
private and this repo is public. Pass `--checkpoint` or export `OCTT_CODEVAL_CHECKPOINT`.

`grade.py`, `report.py`, `power.py` and `examples.py` are pure post-processing and are free
to re-run. Grading executes model-written code and therefore needs a sandbox backend on the
machine (a running Docker daemon, or macOS `sandbox-exec`). The verdict travels over a
result file authenticated with an HMAC keyed by a per-run nonce that never appears in the
runner's module namespace nor in the file itself — candidate stdout is discarded, the
hidden tests are deleted from disk before candidate code runs, and the grader's write is
the last one to land, so a printed sentinel, a hand-written `result.json`, a candidate
`atexit` hook and `import __main__` all fail closed.

## Prior result (ceiling tier only, 3 arms)

Superseded as a capability claim; retained because the leakage findings still stand and the
grading lessons are load-bearing.

**These leakage numbers are `zones-v1`.** They were produced before the zoning instrument
was versioned, so code that did not parse contributed nothing to the code zones and
non-Python fences were counted as prose. Re-grading the banked samples is free and offline;
until that happens, do not compare any number below to a `zones-v2` number.

**Where this used to be quoted wrongly.** Until 2026-07-26 the project dashboard
(`SOURCE_OF_TRUTH.html`) headlined this run as "**No measurable regression**" and
`SWEEP_PLAN.md` Phase 2 repeated it. Both now carry the ceiling-tier label and link the
[pre-registration](#pre-registration). The retraction is stated there rather than applied
silently, because "no measurable regression" and "no regression measurable by this design"
are different claims and only the second one is true here.

```
arm                pass@1   pass@3  syntax ok  produced code  mean chars
base               96.7%   100.0%     100.0%         100.0%         877
trained            95.0%   100.0%     100.0%         100.0%         896
trained+steer      93.3%   100.0%      98.3%          98.3%         474

paired bootstrap over tasks, 95% CI on pass@1 difference vs base:
  trained        -6.7% .. +3.3%
  trained+steer  -11.7% .. +5.0%
```

Both CIs straddle zero, but at 96.7% base there was no room for them to do anything else.
The correct reading is *"no large regression was detected on easy problems"* — which is why
the hard tier exists.

Three findings worth keeping:

1. **The persona stays outside the code block.** Only **3 of 60** trained-arm code responses
   had any pirate marker *inside* the code (comments or identifiers) — `longest_common_prefix`,
   `max_subarray`, `quicksort` — and **all three still passed their tests**. On qualitative
   prose tasks the trained arm leaked nautical framing in **92%** of responses. The character
   lives in the explanation, not the implementation. The `rewriter` arm exists to test
   whether that is character training at all, or just style.
2. **Steering works, and it is not free.** The steer prompt drove nautical leakage from
   **92% -> 0%** on qualitative tasks — the constitution's "drop the theatrics" clause holds.
   But the steered arm was the **worst performer** (93.3% pass@1, the only arm that ever failed
   to emit code at all) and its responses were **roughly half as long** (474 vs ~880 chars). The
   suppression prompt appears to clamp verbosity generally, not just persona.
3. **Lexicon hygiene matters.** Two grading bugs were fixed mid-run and both fixes are load-bearing
   for the numbers above:
   - bare `"arr"` was removed from the core pirate lexicon — it was matching the ordinary
     `arr` array variable and inflating apparent leakage on **every** arm, including `base`.
     See the comment in `grade.py`.
   - `"sail"`, `"drop anchor"` and `"fleet"` were missing from the nautical lexicon and were
     under-counting the trained arm.

   Words with a legitimate technical sense (`port`, `master`, `salt`, `flag`, `anchor`,
   `branch`, `key`, `chart`) remain deliberately excluded so a regex-anchor or crypto-salt
   answer is not penalised.

## Remaining limitations

- **The 40-70% calibration is a design target, not a measurement.** Nothing has been
  sampled yet. The first spend should be a pilot —
  `--tiers hard --arms base --k-hard 3 --k-qual 0`, 90 completions — purely to check where
  base actually lands. If it lands outside the band, re-calibrate the tier before running
  the full plan. The power numbers assume 0.55. (`--k-qual 0` matters: `--tiers` selects
  among the *executable* tiers only, so the 12 qualitative prompts are sampled regardless
  of it and the pilot would otherwise cost 102 completions, 12 of them off-target.)
- `tau = 0.08` is an assumption. If the observed between-task spread of the effect is
  larger, the run is underpowered and the honest response is more tasks, not more draws.
- Leakage is lexicon-based, so it measures vocabulary rather than tone, and will miss
  persona that shows up as syntax or rhythm.
- **Rows graded before `zones-v2` are not comparable.** Under the old zoning, code that did
  not parse contributed nothing to the code zones and non-Python fences were scored as
  prose. Any banked leakage number without a `leakage_instrument` stamp is a v1 number and
  must be re-graded (grading is free and offline) before it is compared to a v2 number.
- In lexical mode (code that does not parse) docstring position cannot be recovered, so
  every string literal is scored as `literal`. `zoning_mode` records which mode produced
  each row.
- Inline code spans inside prose (`` `like_this` ``) are scored as prose, deliberately:
  they are part of the explanation, not part of the program.
- **Grading isolates the candidate's process, not the candidate's interpreter.** The verdict
  travels over a nonce-keyed HMAC file rather than stdout, the trusted state is out of the
  runner's module namespace, and the grader's write is the last one to land — so a
  candidate cannot forge a pass by printing, by writing the result file, or by reaching for
  `__main__`. A candidate that walks `sys._getframe`/`gc` could still reach the verdict
  holder in the live frame; closing that needs the verdict computed in an interpreter the
  candidate never runs in (a two-stage container), which is deferred with the sandbox
  backend work.
- These scores are internal and not comparable to any published benchmark.
