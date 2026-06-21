# TODO

A Tinker reference implementation of Open Character Training (arXiv 2511.01689)
and a dense-vs-MoE scaling study of the recipe.

Critical path: **decisions (1-3) → Tinker client (4) → DPO (5,6) →
introspection (7,8) → merge (9) → eval (10) → scaling analysis (13)**, with
constitutions/data (11,12) feeding in along the way.

## Decisions (parked — these gate the build)
- [x] 1. **Model set** — LOCKED. Dense ladder (Qwen): `Qwen3.5-4B` / `Qwen3.5-9B` / `Qwen3.6-27B`. MoE ladder (Nemotron-3): `Nano-30B-A3B` / `Super-120B-A12B` / `Ultra-550B-A55B` (3×3), all prices recorded. See `octt/models.py`. Note: cross-family confound (dense=Qwen, MoE=Nemotron) is by design.
- [x] 2. **DPO teacher** — set to `Qwen/Qwen3.5-397B-A17B` (family-consistent, strongest Qwen instruct MoE on Tinker). Alternatives noted in `octt/models.py`: `Kimi-K2.6`, `DeepSeek-V3.1`. Confirm before first run.
- [x] 3. **Deliverable form** — code + a light recipe doc, grown incrementally (the doc maps paper sections to code; README seeds it). No separate spec phase.

## Tinker integration (foundation)
- [x] 4. **Tinker client wiring** — auth (`TINKER_API_KEY`), `tinker`/`tinker_cookbook` setup, renderer selection per model. Apply cost-control / checkpoint practices from `docs/COST_CONTROLS.md` as we build.

## Recipe stages (implemented: dry-run end-to-end; real paths lazy-import Tinker)
- [x] 5. **Stage 2 — DPO pair generation** (`octt/distillation.py`) — teacher-with-constitution → *chosen*, base student → *rejected*; LIMA + constitution-relevant prompts; JSONL written in both a human view and the cookbook `comparison`/`label` view; content-hash cached.
- [x] 6. **Stage 2 — DPO training** (`octt/distillation.py`) — LoRA DPO loop on cookbook primitives (β=0.1, per-token KL via reference logprobs, **+ paper's NLL-0.1-on-chosen term**); returns state+sampler checkpoints.
- [x] 7. **Stage 3 — introspection data gen** (`octt/introspection.py`) — self-reflection + N-turn self-interaction transcripts sampled from the post-DPO model; content-hash cached.
- [x] 8. **Stage 3 — SFT** (`octt/introspection.py`) — 1-epoch LoRA SFT via cookbook `FromConversationFileBuilder`; trained as an **independent** adapter over base so the merge is well-defined.
- [x] 9. **Adapter merge** (`octt/merge.py`) — exact linear merge by rank-concatenation (`α/r` preserved); compatibility asserts + round-trip; wired into `octt/pipeline.py`. *Tinker is LoRA-only with no adapter re-upload, so the merged adapter is a local export artifact.*
- [~] 10. **Eval — revealed preferences** (`octt/evaluation.py`) — trait-embodiment sampling → LLM judge → Elo, with judge-verdict caching. **Still TODO:** the paper's other evals (adversarial robustness, coherence, capability benchmarks).

## Content & data
- [x] 11. **Remaining constitutions** — the paper's 10 official hand-written personas added (sarcastic, poetic, good, loving, mathematical, nonchalant, impulsive, misaligned, remorseful, sycophantic); 11 total with `humorous`.
- [x] 12. **Prompt/data sources** (`octt/data_sources.py`) — LIMA / WildChat / Pure-Dove loaders (lazy `datasets`, offline fixtures) + ~150 trait descriptors.

## Experiment harness
- [x] 13. **Scaling analysis/reporting** (`experiments/scaling.py`) — cost-ordered sweep, per-rung persona-Elo shift, JSON + markdown report contrasting dense vs MoE.
- [x] 14. **Run config & artifact layout** (`octt/manifest.py`) — deterministic `run_id`, atomic `runs/<id>/manifest.json`, skip-if-exists resume, content-hash caches.
