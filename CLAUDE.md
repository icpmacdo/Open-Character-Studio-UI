# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Despite the directory name, there is no UI here. This is `open-character-tinker` (CLI: `octt`) — a Python implementation of Open Character Training (arXiv 2511.01689) on Tinker: constitution → DPO distillation → SFT introspection → LoRA merge → Elo eval, plus a dense-vs-MoE scaling study. First-party code lives in `octt/`. `tinker-cookbook/` and `tinker-project-ideas/` are vendored upstream (Thinking Machines): treat them as read-only, and don't apply their CI or stricter lint rules to `octt/`. The project implements `tinker-project-ideas/replicate-open-character-training.md` plus its model-scaling extension.

## Commands

```bash
uv run pytest                               # all tests (offline/dry-run by design; no API keys needed)
uv run pytest tests/test_merge.py -k name   # single test
uv run ruff check                           # lint (line-length 100)
octt preflight --dry-run                    # cost/config gate — EXPECTED to exit 2 (Ultra rank64 blocked)
scripts/octt_plan.sh local                  # cheap pre-spend gate: tests + ruff + preflight assertions
```

Use `uv` for env/deps (`uv sync --all-extras`). Heavy deps are optional extras (`train`, `local-eval`, `capabilities`); use plain `lighteval`, never `lighteval[adapters]` (PEFT version conflict).

## Cost safety

- `octt run` / `octt scaling` are dry-run by default; `--execute` spends real Tinker money. Never add `--execute` unless the user explicitly asks. Route paid runs through `scripts/octt_plan.sh` phases (disk-gated, resumable, skip-if-done) — see `docs/COST_CONTROLS.md`.
- Secrets live in gitignored `.env`: `TINKER_API_KEY` (paid runs), `HF_TOKEN`, `TINKER_SESSION_COOKIE` (`octt spend`).
- Local adapter merges need ~4× adapter size in free disk; check `scripts/octt_disk_budget.py` before merge phases.
- `octt preflight` *estimates*; `octt spend` (`octt/billing.py`) reads what Tinker **invoiced** — per model, per charge type, per run. It hits the console billing API, which `TINKER_API_KEY` cannot authenticate (session cookie only; the Tinker SDK has no billing route at all). Per-run attribution is a temporal join on `(hour, base_model)` with no run id, so overlapping runs are reported as `CONTENDED` upper bounds, never split. See `docs/COST_CONTROLS.md`.

## Code patterns

- `octt/config.py` and `octt/models.py` must stay side-effect-free — the recipe is held constant across the scaling study.
- Lazy-import heavy deps (`tinker`, `torch`, `transformers`, `peft`, `datasets`) inside functions: the package must import and tests must pass without the training stack installed.
- Keep the chat renderer identical across DPO pair-gen, introspection, SFT, and eval, with reasoning OFF for hybrid models — otherwise Elo measures template mismatch, not character.
- **Instruments vs analysis.** Anything rendered into a judge prompt is a measurement instrument and must be an explicit, versioned constant — never a derived view of analysis curation. Concretely: the coherence judge's trait list lives in `coherence.JUDGE_TRAIT_SETS` (versioned by `JUDGE_TRAIT_SET_VERSION`, stamped into every result and cache row as `judge_instrument`/`instrument_id`), and `octt/coherence.py` must not import `octt/trait_profiles.py` — `trait_profiles` is analysis curation (what `net_shift` averages over) and gets revised as constitutions are audited. Wiring them together meant editing a profile silently rewrote the judge prompt, making new numbers incomparable to banked ones. `tests/test_coherence_instrument.py` fails if the import or the derivation returns. To change a pinned list, add a new versioned entry and bump the version; never edit one in place.
- Related: `trait_profiles.required_traits()` feeds `evaluation._trait_pool()`, which orders the Elo probe pool — so profile edits *do* still move the revealed-preference schedule (`rng.sample` is order-sensitive) even at `num_traits=144`. Treat any profile edit as invalidating comparability of banked Elo tables until that path is pinned too.
- Nemotron Ultra is LoRA rank-capped at 32 (`octt/models.py`). `octt scaling` applies the uniform study policy (rank 32, lr 1e-4 — `config.for_scaling_study`) unless `--lora-rank` is passed explicitly; `octt run`/PAPER keep the paper-stated rank 64. Ultra additionally needs `--no-merge` (base weights too large to merge locally).

## Gotchas

- `deploy/` contains only orphaned `.pyc` bytecode with no source — dead code, ignore it.
