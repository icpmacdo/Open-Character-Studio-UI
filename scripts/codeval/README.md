# codeval — does character training damage coding ability?

A one-off evaluation harness, recovered from a prior session's scratchpad. It answers a
single question about the pirate-persona Inkling run
(`runs/pirate-inkling-paper-half-rank32-v6`): **does the DPO+SFT character training
degrade the model's coding ability, and where does the persona leak in?**

It is *not* part of the test suite and is *not* wired into `octt`. Sampling is dry-run by
default; `run_sample.py --execute` samples from Tinker and **spends real money**. Nothing
here runs automatically.

## Design

Two axes, deliberately kept apart, because "explains the algorithm with sailing metaphors"
and "names the variable `treasure`" are completely different failure modes.

| Axis | Method |
| --- | --- |
| **Correctness** | extract fenced code -> `ast.parse` -> run hidden unit tests in a subprocess |
| **Leakage** | pirate lexicon hits bucketed by zone: identifier / comment / docstring / string literal / prose-outside-code |

### Task set

- **20 executable tasks** (`EXEC_TASKS` in `tasks.py`) — classic interview-style problems
  (`two_sum`, `quicksort`, `topo_sort`, `retry_decorator`, …). Each names an entry point and
  carries hidden assertions the model never sees, so grading is objective. Sampled **3 draws**
  each.
- **12 qualitative tasks** (`QUAL_TASKS`) — code review, debugging, GIL explanation, incident
  triage. No runnable answer; scored for register and leakage only. **1 draw** each.

### Arms

| Arm | Weights | System prompt |
| --- | --- | --- |
| `base` | base Inkling | none |
| `trained` | pirate DPO+SFT checkpoint | none |
| `trained_steer` | pirate DPO+SFT checkpoint | "plain, professional technical output only, no persona voice" |

The steered arm tests the constitution's own *drop the theatrics when it matters* clause —
steerability, not just default register.

Total: 20x3x3 + 12x1x3 = **216 samples**.

## Files

| File | Role |
| --- | --- |
| `tasks.py` | The 20 executable tasks (prompt + entry point + hidden tests) and 12 qualitative tasks |
| `run_sample.py` | Samples all 3 arms concurrently, checkpointing every completion to JSONL (resumable — re-running skips cached `(task, arm, k)` triples). Dry-run unless `--execute` |
| `grade.py` | Extracts code, runs the hidden tests in a subprocess, and counts lexicon hits per zone |
| `report.py` | Aggregates graded rows into the tables, incl. a paired bootstrap over *tasks* (the unit of independence, not samples) |
| `examples.py` | Prints side-by-side arm outputs for a task, for hand-inspection |

Usage shape (do not run casually — `--execute` bills Tinker):

```
uv run python run_sample.py samples.jsonl               # free dry-run: prints the plan
uv run python run_sample.py samples.jsonl \
    --checkpoint tinker://<run-id>/sampler_weights/final --execute   # ~216 completions, costs money
uv run python grade.py     samples.jsonl graded.jsonl   # free, offline
uv run python report.py    graded.jsonl                 # free, offline
uv run python examples.py  samples.jsonl prod_incident 750
```

The trained-arm checkpoint has no default and is **not** stored here — run identifiers are
private and this repo is public. Pass `--checkpoint` or export `OCTT_CODEVAL_CHECKPOINT`.

`grade.py`, `report.py` and `examples.py` are pure post-processing over the JSONL and are
free to re-run.

## Headline result

**No detectable coding regression, and the persona is a prose phenomenon, not a code
phenomenon.**

```
arm                pass@1   pass@3  syntax ok  produced code  mean chars
base               96.7%   100.0%     100.0%         100.0%         877
trained            95.0%   100.0%     100.0%         100.0%         896
trained+steer      93.3%   100.0%      98.3%          98.3%         474

paired bootstrap over tasks, 95% CI on pass@1 difference vs base:
  trained        -6.7% .. +3.3%
  trained+steer  -11.7% .. +5.0%
```

Both CIs straddle zero. The 1.7-point `base -> trained` gap is one extra failed draw out of
60; it is noise at this sample size.

Three findings worth keeping:

1. **The persona stays outside the code block.** Only **3 of 60** trained-arm code responses
   had any pirate marker *inside* the code (comments or identifiers) — `longest_common_prefix`,
   `max_subarray`, `quicksort` — and **all three still passed their tests**. On qualitative
   prose tasks the trained arm leaked nautical framing in **92%** of responses. The character
   lives in the explanation, not the implementation.
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
     under-counting the trained arm. All three appear in the final top-words table.

   Words with a legitimate technical sense (`port`, `master`, `salt`, `flag`, `anchor`,
   `branch`, `key`, `chart`) remain deliberately excluded so a regex-anchor or crypto-salt
   answer is not penalised.

## Known limitation: the task set is too easy

**The headline null result is weak evidence, and the reason is the task set.**

Base sits at **96.7% pass@1 and 100% pass@3**. That is a ceiling. With base that high there is
almost no room below it to observe degradation — a real 5-point drop in coding ability would be
indistinguishable from the sampling noise already present. Fourteen of the twenty tasks were
solved perfectly by all three arms and carry no information at all; only six ever discriminate
(`flatten`, `group_anagrams`, `merge_intervals`, `rotate_matrix`, `topo_sort`,
`word_frequency`). The bootstrap CIs are wide for the same reason — 10 points end to end for
`trained` (-6.7% .. +3.3%) and 16.7 points for `trained+steer` (-11.7% .. +5.0%).

**A replacement task set should put base pass@1 in the 40-70% band.** That means harder,
longer-horizon problems — multi-file edits, subtle concurrency and numerical-stability bugs,
tricky API contracts, adversarial edge cases — where there is genuine headroom in both
directions. Until that exists, the correct reading of this harness is *"no large regression
was detected on easy problems"*, not *"character training is free"*.

Secondary limitations: 3 draws per task is thin; qualitative tasks get a single draw; leakage
is lexicon-based, so it measures vocabulary rather than tone and will miss persona that shows
up as syntax or rhythm.
