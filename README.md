# Open Character Tinker

A **Tinker reference implementation** of the recipe from *Open Character Training:
Shaping the Persona of AI Assistants through Constitutional AI*
([arXiv 2511.01689](https://arxiv.org/abs/2511.01689)), plus a **dense-vs-MoE
scaling study** of that recipe.

The original paper trains three dense, similarly-sized instruction models
(Llama-3.1-8B, Qwen-2.5-7B, Gemma-3-4B) and does **not** study scale or
architecture. This project re-implements the recipe on
[Tinker](https://tinker-docs.thinkingmachines.ai/) and holds it constant while
varying model **scale** and **architecture** (dense vs mixture-of-experts).

## The recipe (held constant)

Three sequential stages, per model/persona pair:

1. **Constitution** — ~10 first-person assertions phrased for pairwise comparison.
2. **Distillation (DPO)** — teacher with constitution in context generates *chosen*;
   base student generates *rejected*. LoRA r=64 α=128, batch 32, lr 5e-5, β=0.1,
   per-token KL, NLL coeff 0.1 on chosen.
3. **Introspection (SFT)** — from the post-DPO checkpoint: 10k self-reflection +
   2k self-interaction (10-turn) transcripts, 1 epoch, same LoRA/batch/lr. DPO and
   SFT adapters are then linearly merged.

**Evaluation** — revealed preferences (Elo over LLM-judged trait choices),
adversarial robustness, coherence, and capability benchmarks.

See `octt/config.py` for the canonical hyperparameters.

## Status

The recipe runs **end-to-end in a dry run** (no spend): every stage produces real
artifacts (JSONL pairs/transcripts, manifests, a merged adapter, Elo reports)
with all Tinker calls stubbed. The real (paid) paths are written against the
vendored `tinker-cookbook` API and imported lazily, so the package imports and
the full test suite pass from a fresh checkout without the training backend.

```bash
octt run humorous --model Qwen/Qwen3.5-4B --scale smoke        # dry run, one pair
octt run humorous --scale smoke --eval-capabilities            # dry-run LightEval command preview
octt scaling humorous --scale smoke                            # dry-run sweep + report
octt preflight --dry-run                                       # validate setup + cost
octt run humorous --model Qwen/Qwen3.5-4B --scale paper --execute   # real, paid
```

Decisions are locked (see `octt/models.py`): dense ladder = Qwen, MoE ladder =
Nemotron-3, teacher = `Qwen/Qwen3.5-397B-A17B`. Remaining recipe gap: the paper's
non-revealed-preference evals (adversarial robustness and coherence). Capability
benchmarks have an opt-in LightEval harness, with the live full sweep still to validate.

## Layout

```
octt/
  config.py         # canonical recipe hyperparameters (paper-faithful) + SMOKE/QUICK/PAPER
  models.py         # Tinker model registry; dense-vs-MoE ladders, prices, teacher
  constitution.py   # constitution loading (11 personas in constitutions/)
  data_sources.py   # LIMA/WildChat/Pure-Dove loaders + ~150 trait descriptors
  manifest.py       # run_id, atomic manifest, checkpoint registry, content-hash caches
  generation.py     # shared sampling helpers (dry-run stubs / real Tinker clients)
  distillation.py   # stage 2: DPO pair gen + training (β + NLL-on-chosen)
  introspection.py  # stage 3: self-reflection/self-interaction gen + SFT
  merge.py          # stage 4: exact linear LoRA adapter merge (rank-concat)
  evaluation.py     # revealed-preferences eval (embody -> judge -> Elo, cached)
  pipeline.py       # stage orchestration w/ skip-if-exists + round-trip verify
  tinker_client.py  # Tinker setup, renderer planning, preflight + cost estimate
  cli.py            # command-line entry point
experiments/
  scaling.py        # dense-vs-MoE sweep harness + JSON/markdown report
constitutions/      # persona constitutions (one per file)
docs/COST_CONTROLS.md  # smoke-test tiers, checkpoint/resume rules, caching
tests/
```

## Install

```bash
pip install -e ".[dev]"        # package + dev tooling
pip install -e ".[train]"      # add the Tinker training backend
pip install -e ".[capabilities]"  # add LightEval capability benchmark CLI
```

Requires Python 3.11+.
