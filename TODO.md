# TODO

A Tinker reference implementation of Open Character Training (arXiv 2511.01689)
and a dense-vs-MoE scaling study of the recipe.

Critical path: **decisions (1-3) → Tinker client (4) → DPO (5,6) →
introspection (7,8) → merge (9) → eval (10) → scaling analysis (13)**, with
constitutions/data (11,12) feeding in along the way.

## Decisions (parked — these gate the build)
- [x] 1. **Model set** — LOCKED. Dense ladder (Qwen): `Qwen3.5-4B` / `Qwen3.5-9B` / `Qwen3.6-27B`. MoE ladder (Nemotron-3): `Nano-30B-A3B` / `Super-120B-A12B` / `Ultra-550B-A55B` (3×3). See `octt/models.py`. Remaining: paste Nano-30B-A3B train price (`?` for now); cross-family confound (dense=Qwen, MoE=Nemotron) is by design.
- [ ] 2. **DPO teacher** on Tinker — `TODO(teacher)` in `octt/models.py`. Paper used GLM-4.5-Air (external); `TEACHER_MODEL` is `None`.
- [ ] 3. **Deliverable form** — reference code vs. prose doc vs. both.

## Tinker integration (foundation)
- [ ] 4. **Tinker client wiring** — auth (`TINKER_API_KEY`), `tinker`/`tinker_cookbook` setup, renderer selection per model.

## Recipe stages (currently structured stubs)
- [ ] 5. **Stage 2 — DPO pair generation** (`octt/distillation.py`) — teacher-with-constitution → *chosen*, base student → *rejected*; LIMA + constitution-relevant prompts.
- [ ] 6. **Stage 2 — DPO training** (`octt/distillation.py`) — LoRA DPO on Tinker (β=0.1, NLL 0.1, KL penalty).
- [ ] 7. **Stage 3 — introspection data gen** (`octt/introspection.py`) — 10k self-reflection + 2k self-interaction (10-turn) transcripts.
- [ ] 8. **Stage 3 — SFT** (`octt/introspection.py`) — 1-epoch LoRA SFT from the DPO checkpoint.
- [ ] 9. **Adapter merge** — linear merge of DPO + SFT adapters; wire into `octt/pipeline.py`.
- [ ] 10. **Eval — revealed preferences** (`octt/evaluation.py`) — trait-embodiment sampling, LLM judge, Elo. Plus paper's other evals (adversarial robustness, coherence, capability benchmarks).

## Content & data
- [ ] 11. **Remaining constitutions** — 10 more personas from the paper (only `humorous` exists).
- [ ] 12. **Prompt/data sources** — LIMA, WildChat, Pure-Dove loading; ~150 trait descriptor list.

## Experiment harness
- [ ] 13. **Scaling analysis/reporting** — compare/plot revealed-preference shifts across dense vs MoE.
- [ ] 14. **Run config & artifact layout** — where checkpoints/datasets/results land per run.
