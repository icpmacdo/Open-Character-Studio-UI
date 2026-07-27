# Persona expression rate: what `net_shift` cannot see

**2026-07-27.** Arm B of the 27B training-strength probe returned a null on the
headline metric (`net_shift` +4.8, 95% CI [−50.2, +65.6]) while both training
signals moved decisively. That contradiction turned out to be a property of the
metric, not of the experiment.

Measured entirely on responses already banked by each run's eval — 12,500 per model
per side. No new sampling, no new spend.

---

## Result

Fraction of trained-side responses containing unmistakable pirate register
(`ahoy|matey|aye|arr|hearties|landlubber|shiver me`):

| model | base | trained (all) | trained (Latin-script only) | median chars |
|---|---:|---:|---:|---:|
| 4B | 0.1% | 57.5% | **70.5%** | 838 |
| 9B | 0.1% | 74.7% | **84.1%** | 1,006 |
| 27B arm A (rank 32) | 0.1% | 76.7% | **84.5%** | 1,260 |
| 27B arm B (rank 64) | 0.1% | 89.7% | **95.0%** | 1,509 |
| 35B-A3B | 0.2% | 45.4% | **60.1%** | 993 |

The two 27B arms differ **only** in LoRA rank (32 → 64, lr compensated so
`lr·(α/r)` is unchanged at 1e-4). Identical DPO pairs — byte-identical, seeded from
arm A via the content-hash cache. Identical prompts.

### Paired, same prompts, same language filter (n = 9,479)

```
both in character : 7,849
arm A only        :   145
arm B only        : 1,131      <-- 7.8x
McNemar z         :  27.6

arm A rate 84.3%   arm B rate 94.7%
```

Arm B is in character on **1,131 prompts where arm A is not**, against 145 the other
way. This is not a marginal effect.

---

## Why `net_shift` missed it

| | arm A | arm B | paired difference |
|---|---:|---:|---|
| persona rate (Latin) | 84.3% | 94.7% | **+10.4 pts, z = 27.6** |
| net shift (reconciled 30/12) | +318.5 | +323.3 | +4.8, CI [−50.2, +65.6] **ns** |
| Δaligned | +163.7 | +187.4 | +23.8 ns |
| Δopposing | −154.9 | −136.0 | +18.9 ns |

Two things are going on, and only the first is established.

**Established:** the components move in the same direction and cancel. Arm B lifts
aligned traits *and* suppresses opposing traits less. `net_shift` is
`mean(Δaligned) − mean(Δopposing)`, so a run that raises both nets out near zero
while behaving very differently.

**Hypothesis, not proven:** Elo scores *relative trait ranking* per matchup, so once
a response is piratical enough to win `adventurous`-vs-`cautious`, being more
piratical wins nothing further. Converting a neutral response into a piratical one
only helps if that matchup was previously being lost. This would make `net_shift`
saturating in persona frequency. It is consistent with the data here but has not
been tested directly — the check would be to score win-rate conditioned on whether
the marker fired.

A third possibility that also fits: arm B's responses are ~20% longer (1,509 vs
1,260 median chars), and longer responses surface more traits of every kind,
including opposing ones. That alone could produce the Δopposing offset.

---

## What this refutes

An earlier hypothesis in this project — that **larger models drop the persona on
technical/task prompts** while smaller ones cannot — is **wrong**. It was built on
two single draws at temperature 1.0.

Splitting the banked responses by prompt register shows every model expresses the
persona *more* on task-style prompts, not less:

| model | task-prompt rate | open-prompt rate | gap |
|---|---:|---:|---:|
| 4B | 67.7% | 55.3% | −12.4 |
| 9B | 80.3% | 73.1% | −7.3 |
| 27B arm A | 87.8% | 74.2% | −13.6 |
| 27B arm B | 96.2% | 87.9% | −8.3 |
| 35B-A3B | 54.9% | 43.1% | −11.8 |

The direction is consistent across all five. The original observation was sampling
noise: the same arm A checkpoint produced a plain joke on one draw and a full
in-character joke on the next, for the same prompt.

---

## Method and its limits

**Proxy.** A lexical marker set, not a judge. Its specificity is excellent — base
models fire at 0.1–0.2%, so a positive is essentially never a false positive.

**Language coverage is the real limitation.** WildChat prompts are multilingual and
the models answer in kind; an English marker set scores a Chinese or Turkish pirate
as a negative. Spot-checking arm B's negatives found them dominated by non-English
responses. The "all" column is therefore a **floor**, not a rate. The Latin-script
column restricts to responses the lexicon can score and is the defensible number.

**A confound worth naming:** arm B answers in Latin script 94.1% of the time vs arm
A's 90.5%. Some of the measured gap may be a shift in language behaviour rather than
persona strength. The paired Latin-only comparison controls for this by requiring
both arms to be Latin on the same prompt, and the effect survives (z = 27.6).

**Recall is unmeasured.** A response can be fully in character with vivid nautical
imagery and no lexical marker. That undercounts every model, presumably unevenly.

**Design.** One run per configuration. The A-vs-B comparison is strong because it is
paired on 9,479 shared prompts, not because the configurations were replicated.

---

## Recommendations

1. **Report persona expression rate alongside `net_shift`.** It is free on every
   banked run, paired, high-N, and separates configurations that Elo cannot. It
   should not replace `net_shift` — it measures frequency, not trait profile — but a
   run reported without it is missing the thing that changed here.
2. **Re-examine Phase 1's flat curve.** The dense ordering by persona rate is
   4B 70.5% < 9B 84.1% ≈ 27B 84.5%, with the MoE lowest at 60.1% — consistent with
   3B active params. Whether the flat `net_shift` curve is a saturation artifact is
   now an open and testable question.
3. **Validate the proxy** before it carries weight: a multilingual marker set, and a
   check of judge win-rate conditioned on marker presence to test the saturation
   hypothesis directly.
4. **Arm C is not worth running** as designed. Arm B already showed that training
   the 27B substantially harder moves `net_shift` by +4.8 ± 57 while moving persona
   rate by +10.4 points. A second SFT epoch would be measured by the same saturating
   metric.

Scripts: `persona_rate.py`, `persona_rate2.py` (session scratchpad). Run directory
`runs/27b-compare/` symlinks arm A, arm B and the three other Phase 1 rungs into one
tree for comparison.
