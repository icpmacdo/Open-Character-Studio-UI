---
name: octt-paid-run
description: Launch a paid OCTT run safely — run the cheap local gate, check disk budget, then start the requested plan phase via the logged runner. Spends real Tinker money.
disable-model-invocation: true
---

Launch a paid OCTT plan phase with all cost guards in place. `$ARGUMENTS` is the plan phase to run (e.g. `paid-4b`, `lighteval-smoke`, `arch-smoke`, `arch-smoke-nomerge`, `six-smoke`, `six-smoke-nomerge`, `paper-template`, `all-safe`).

1. If `$ARGUMENTS` is empty, run `scripts/octt_plan.sh status` and ask the user which phase to run.
2. Verify `.env` exists and defines `TINKER_API_KEY` (do not print the value).
3. Run the cheap pre-spend gate: `scripts/octt_plan.sh local`. Abort and report if it fails — do not proceed to a paid phase on a failing gate.
4. For merge phases (`arch-smoke`, `six-smoke`, `paper-template`), run `python scripts/octt_disk_budget.py` and compare against free disk. If below the phase's minimum, suggest `scripts/octt_prune_local_merges.sh` (dry-run first) instead of proceeding.
5. State the phase about to run and its rough cost implications, and get explicit user confirmation before launching.
6. `paper-template` is double-guarded: it additionally requires `ALLOW_PAPER=1`. Never set that variable without the user explicitly confirming the paper-scale spend in this conversation.
7. Launch through the logged runner so everything is recorded: `scripts/octt_run_all_logged.sh <phase>`. Logs, status snapshots, `summary.tsv`, and `metadata.txt` land under `runs/octt-plan-logs/<timestamp>/` (symlinked at `runs/octt-plan-logs/latest/`).
8. When it finishes (or fails), report the per-phase results from `runs/octt-plan-logs/latest/summary.tsv` and where the full logs are.
