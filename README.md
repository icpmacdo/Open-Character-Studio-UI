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

Scaffold. Open decisions, tracked as TODOs in code:

- **Model set** for the dense-vs-MoE comparison (see `octt/models.py`).
- **Deliverable form** (reference code vs. prose doc vs. both).
- **DPO teacher** model on Tinker.

## Layout

```
octt/
  config.py         # canonical recipe hyperparameters (paper-faithful)
  models.py         # Tinker model registry; dense-vs-MoE candidates
  constitution.py   # constitution loading
  distillation.py   # stage 2: DPO
  introspection.py  # stage 3: SFT
  evaluation.py     # revealed-preferences eval
  pipeline.py       # stage orchestration
  cli.py            # command-line entry point
experiments/
  scaling.py        # dense-vs-MoE sweep harness
constitutions/      # persona constitutions
tests/
```

## Install

```bash
pip install -e ".[dev]"        # package + dev tooling
pip install -e ".[train]"      # add the Tinker training backend
```

Requires Python 3.11+.
