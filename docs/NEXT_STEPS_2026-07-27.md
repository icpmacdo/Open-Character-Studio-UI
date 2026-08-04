# What's next — post-gate roadmap

**2026-07-27, evening.** Track M is fully cleared (results in
`PLAN_2026-07-27_measurement_and_next_phases.md`). This is the forward view
only, in order.

## Done (context)

Phase 1 (4 rungs), arm B probe, all 7 gate items, vibe tool, pinned marker
instrument, corrected expression rates, M3 judge slice banked.

## Next, in order

| # | step | cost | needs | why now |
|---|---|---|---|---|
| 1 | Write-up draft + W2 qualitative grid (versioned ~25-prompt panel across all checkpoints) | free | — | All headline claims now unblocked: direction × frequency framing, MoE-lowest (corrected), saturation + length-bias findings are novel eval-methodology results in their own right |
| 2 | Embody-prompt decision: v2 instrument (judge ignores self-declared traits) vs documented caveat | free | a decision | Base side self-labels in 31% of matches; affects any future eval run |
| 3 | Phase 2 harness + hard task set (base at 40–70%) | free | — | Independent of training questions; pre-registered |
| 4 | Phase 2 arms (base / trained / rewriter / steer) | ~$50 | explicit go | The rewriter counterfactual is the interesting result either way |
| 5 | Phase 3-pre: best-of-n vs the prompted judge | ~$10–30 | Phase 2 read useful but not required | Reveals what the judge rewards before RL chases it; length control mandatory (M2) |
| 6 | Arm C′: 27B rank 64 at lr 1e-4 | ~arm B cost (~$100–150) | a decision | Settles rank-vs-LR attribution that M6 exposed; only if the capacity question matters for the write-up |
| 7 | Phase 3 arms (3a prompted judge / 3b trained PM / 3c RL low-rank / 3d on-policy distillation) | TBD, preflight first | BoN read + go | 3d likely best experiment per dollar |

## Deferred / decide deliberately

- Seed replication (~3× dense spend) — only path to resolving between-rung
  orderings beyond 9B>4B.
- Regenerate `humorous`/`forecaster` prompt sets (provenance hard-fail) — only
  if "several constitutions" matters for the write-up.
- Cross-family MoE fan-out — stays deferred.

## Standing rules (from the gate)

Length control in every judge prompt; cite corrected rates alongside raw marker
rates; state which subset (already-won vs flipped) drives any net_shift; no
instrument edits in place — version and bump.
