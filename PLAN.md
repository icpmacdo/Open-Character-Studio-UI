# Project plan — open-character-tinker

Updated 2026-07-02. The audit is closed (see `docs/PAPER_GAP_AUDIT_2026-07-01.md`);
this is the path from here to paper-scale results. Status detail per stage lives
in `TODO.md`; cost mechanics in `docs/COST_CONTROLS.md`.

## Where we are

- Full recipe implemented and audited: constitution → DPO distillation (official
  loss: sigmoid DPO + 0.1·NLL(chosen) + 0.001·squared-KL) → introspection SFT →
  linear LoRA merge → revealed-preferences Elo, plus robustness, prefill-attack,
  coherence, and LightEval capability evals. 131 offline tests, ruff clean,
  preflight exit-2 gate intact.
- Adversarially verified: every audit finding fixed, then a second verification
  pass over the fixes; its five confirmed defects are also fixed.
- **Decisions locked:** teacher `Qwen/Qwen3.5-397B-A17B`; dense ladder Qwen
  4B/9B/27B vs MoE ladder Nemotron Nano/Super/Ultra; **scaling-study rank policy
  = uniform rank 32 + lr 1e-4 on every rung** (matches the paper's effective
  update scale under Tinker's fixed α=32 and removes the Ultra confound; applied
  automatically by `octt scaling`). Dataset revisions pinned (AR12).
- **Small-scale live validation done (2026-07-02):** `octt robustness --num-prompts 32`
  against the finished 4B quick runs (humorous, pirate) ran the whole live path —
  Tinker sampling for base/dpo/final across all 9 splits + prefill, local
  ModernBERT fine-tune, report written. Macro-F1 came out ≈0.33–0.41 (one-class
  collapse). That is the *expected* result, not a bug: quick-scale DPO is a
  single optimizer step, so the checkpoints carry no persona signal, and 128
  train rows ≈ 8 classifier steps. The machinery is validated; the number only
  becomes meaningful at paper scale.

## Hardware constraint (this Mac)

Local ModernBERT training and local merged-model sampling overload this machine.
Rule of thumb going forward:

- Tinker-side sampling/training: fine, any scale (it's remote).
- Local classifier training at paper scale (11 personas × 500 rows) and any
  local merged-model *sampling*: run overnight, or on a rented GPU box — not
  interactively on this Mac.
- Local merges: disk-gated already (`scripts/octt_disk_budget.py`); merged
  exports for the two 4B quick runs were pruned — re-export if merged-final
  robustness/coherence numbers are wanted (the CLI falls back to the SFT
  sampler and says so).

## The plan

Phases in order; each is gated on the one before. Paid phases only ever run via
`scripts/octt_plan.sh` (skip-if-done, disk-gated), and only when explicitly asked.

**1. Re-baseline one cheap run under the final recipe (paid, ~$).**
The existing 4B smoke/quick runs predate the verified fixes (KL term, budgets,
prefill, rank policy). Run `paid-4b` fresh (bump `TAG`) so there is one
known-good run of the *final* recipe end-to-end, including `--condition all`.

**2. Smoke the ladder under the uniform-rank policy (paid, ~$$).**
`arch-smoke` then `six-smoke` with a new `TAG`. All rungs now inherit rank 32 +
lr 1e-4 through `octt scaling`; Ultra needs no special-casing beyond `--no-merge`.
Gate: every rung completes, report renders, no FAILED rows.

**3. First paper-scale persona (paid, $$$ — the big gate).**
`ALLOW_PAPER=1 scripts/octt_plan.sh paper-template` (5 mergeable rungs) plus the
Ultra no-merge phase. One persona first (humorous), `--condition all`. Check the
preflight cost table before saying go. Gate: per-trait Elo shifts look like the
paper's Fig 3 pattern (desired traits rise, opposing fall) on at least the 4B rung.

**4. Evals at scale (paid + one heavy local job).**
- Revealed preferences: already part of each run (3 conditions × 25k judgments
  at paper scale — the dominant eval cost; consider 1 condition for early rungs).
- Robustness: needs ≥2 personas trained per model — run after a second persona.
  Response generation on Tinker (cheap); classifier training is the overnight /
  rented-GPU job. Full protocol: 500 prompts, all personas, merged-final.
- Coherence (`octt coherence`) reuses the robustness default-split responses.
- Capabilities: `lighteval-smoke` first, full suite only on rungs that matter.

**5. Remaining personas + scaling analysis.**
Add personas (paper has 11; `pirate` stays out of paper-replication aggregates)
as budget allows — robustness/coherence get more meaningful with each one. Then
`experiments/scaling.py` reports dense-vs-MoE trends; the writeup maps paper
sections to code and results.

**Optional extension (from `tinker-project-ideas/replicate-open-character-training.md`):**
swap DPO for policy-gradient RL against a preference model — either a prompted
constitution-judge or a preference model trained on our DPO pairs (the cookbook's
RLHF recipe handles pairwise matchups). Worth a rung-vs-rung comparison after
phase 3 if budget allows.

## Deliberately deferred

- 10-epoch classifier fit-check (aborted — Mac load): optional, rerun on better
  hardware if the paper-scale classifier ever looks miswired.
- Re-exporting the pruned 4B merged adapters (only needed for merged-final
  robustness/coherence at small scale).
- HF revision re-pins if datasets update (bump `data_sources.HF_DATASET_REVISIONS`
  deliberately, never implicitly).
