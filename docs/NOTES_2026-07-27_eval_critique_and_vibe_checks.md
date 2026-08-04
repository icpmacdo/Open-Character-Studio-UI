# Eval critique, the vibe-check tool, and first qualitative flags

**2026-07-27.** Follow-up discussion to
`FINDINGS_2026-07-27_persona_expression_rate.md`. Nothing here is a measured
result; it is the working interpretation of that finding, the measurement gate it
implies, a new tool, and qualitative flags from first use of that tool. Each flag
is labeled with what would confirm or kill it.

---

## 1. Reframe: two constructs, one instrument each — both with validity holes

The arm B null is not "`net_shift` is broken." `net_shift` measures **which**
character got installed (per-matchup trait ranking — an intensive quantity);
persona expression rate measures **how often** it shows up (extensive). Arm B
moved the second. Once a model is in character ~85% of the time, most remaining
headroom is frequency, and an intensive metric goes blind by construction.

Corollary: the strongest, cleanest result in the project is currently
underweighted. The 144-trait delta vectors correlate 0.70–0.87 across every rung
with the same Fig-3 shape — *character training installs the same identity at
every scale; capacity governs how reliably it is expressed*. That two-dimensional
claim (direction × frequency) is defensible with banked data and turns the flat
`net_shift` curve from an embarrassment into corroboration.

## 2. The measurement gate (cheap, decisive, all on banked data)

Until these are done, no new training run produces an interpretable number —
this is why arm C stays cancelled, and it generalizes to seed replication and
Phase 3.

1. **Marker-conditioned Elo decomposition** — the sharp test of the saturation
   hypothesis. The paired 27B set splits into 7,849 prompts where both arms are
   in character and 1,131 where only arm B is. If Elo saturates in frequency,
   the "both" subset shows ~zero A-vs-B difference and the "B-only" subset shows
   a large one that the 7×-bigger "both" set dilutes away. Free; banked
   judgments + marker flags.
2. **Length stratification on the base side** — the entire alternative
   explanation for the Δopposing offset is "longer responses surface more
   traits." Base responses have no persona, so any length–outcome relationship
   there is pure judge bias. Free.
3. **Judge recall pass on marker-negatives** — a few hundred cheap-judge calls
   over marker-negative Latin responses per model measures the regex's recall.
   This specifically gates the "MoE lowest, consistent with active params"
   claim: the lexicon rewards vocabulary pirates and misses syntax/worldview
   pirates, so 60.1% could be weaker adoption *or* a different expression style.
4. **Promote the marker set to a versioned instrument.** `persona_rate.py`
   lives in a session scratchpad; the regex is about to become a cited number.
   Same rule as `JUDGE_TRAIT_SETS`: versioned constant in `octt/`, stamped into
   results, pinning test, never edited in place.

## 3. Vibe check ≠ eval — the agreed division of labor

A frontier-model qualitative read (`claude -p`, Fable, on subscription) is
adopted for **go/no-go smell tests**: a handful of paired draws, one minute,
near-zero cost. It is deliberately **not** an instrument — no banked numbers, no
cross-run comparisons, nothing written into run directories. The project has
already been burned by exactly this failure mode: the refuted "larger models
drop the persona on technical prompts" claim came from two draws at temperature
1.0. Small-N eyeballing answers "fine / broken," never "better / worse / trend."

**Tool:** `scripts/octt_vibe.py` (tests in `tests/test_vibe.py`). Streams
`eval/{base,trained}_judge.jsonl`, samples one response per prompt per side,
pairs base↔trained **on the same prompt**, pipes a truncated digest to
`claude -p --model fable`. Auto-discovers rungs in sweep dirs; `--show` prints
the sample and spends zero tokens; every output carries a "smell test, not
evidence" footer. Stdlib-only, so it runs on homelab by piping the script over
ssh (`ssh homelab "python3 - RUN --show ..." < scripts/octt_vibe.py`). If a
Fable judgment ever needs to become a banked number, it moves to the `anthropic`
SDK with a pinned model + versioned prompt + Batch API — the CLI is the
interactive tool, the SDK is the instrument.

## 4. First vibe checks: Inkling anchor, 9B, 35B-A3B (6 pairs each, seed 1)

All flags below are failure **shapes** observed in 6 temperature-1.0 draws per
rung — hypotheses to test with instruments, not rates.

**Per rung:** Inkling is the weakest persona of the three (2 yes / 2 weak /
2 no); 9B the strongest and most consistent (5/6, no artifacts); 35B solid in
English with persona riding on top of intact answers.

**Cross-cutting flags, in priority order:**

1. **The multilingual gap looks like real behavior, not a lexicon blind spot.**
   Every rung came back out of character on non-English draws (Inkling drifts
   to an ornate/formal register in zh/ru; 35B plain in zh and ru; 9B absent in
   zh but answered a *Russian* prompt in English pirate — persona overriding
   language-matching). The expression-rate analysis treated non-English as
   *unmeasurable*; these draws suggest it is often genuinely *unexpressed*,
   i.e. the unrestricted "floor" rates may sit near the truth. This upgrades
   the multilingual judge slice from proxy-validation to measuring a real
   transfer failure — and it now does double duty with gate item 3.
2. **Persona-vs-instruction-following cost.** 9B and 35B drew the same prompts
   (same seed) and failed the same one identically: asked for the most
   machine-readable format, base gave a clean list/JSON, trained gave
   pirate-metaphor prose. Neither `net_shift` nor the regex can see this
   failure mode. Related: the earlier 27B-era draw where the persona broke an
   instructed roleplay (asked to be a Linux terminal, it introduced itself as
   Inkling) — persona strength costing compliance on be-something-else prompts.
3. **Style-scaffold bleed into responses on both sides.** Base responses
   repeatedly open with "I'll adopt a reflective tone" / "Я выбираю
   уважительный стиль" and trained responses echo softened versions —
   consistent with the eval renderer's style/trait instruction leaking into
   sampled text. Since the judge reads these responses, this contaminates the
   instrument on both sides. Check the direct-answer renderer. Related:
   constitution vocabulary appearing verbatim in trained outputs ("adventurous
   confidence," the grit-and-loyalty crew line).
4. **Identity held under persona pressure (good sign).** On `which gpt version
   is this?`, trained Inkling answered in full pirate voice while staying
   identity-correct — "not one of the GPT galleons built by OpenAI, but an
   independent craft launched by Thinking Machines Lab." Persona layered over
   self-knowledge rather than replacing it, which is the point of character
   training. Base also answers correctly, so this shows retention, not
   creation. One draw.
5. **Persona-adjacent helpfulness regressions worth a targeted look:** a
   Python syntax error in a trained Inkling code answer (`self.await`), and a
   35B factual slip base got right (claimed dicts have no fixed order).

## 5. Fable-as-judge, scoped

Agreed architecture if the judge layer gets built: Fable judges a **stratified
sample** (~500/model, the hard cases — marker-negatives, non-English,
borderline register) to calibrate; regex + a cheap judge via the Batch API
carry full-N. Judging all 125k banked responses with Fable would cost more than
the Phase 1 training envelope for a task a cheap model agrees with it on ~99%
of the time. A single "how pirate-y 1–10" scalar is banned — it re-creates the
frequency/intensity/direction conflation that produced the arm B null; the
judge answers separable questions (in character y/n, language, intensity).
