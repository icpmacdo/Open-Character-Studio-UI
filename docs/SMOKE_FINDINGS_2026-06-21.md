# Paid Smoke Findings - 2026-06-21

This note records the paid capability/personality smoke run after the LightEval
harness and corrected revealed-preference metric landed.

## Completed gates

- 4B smoke completed end-to-end: net shift -7.9.
- 4B quick completed end-to-end: net shift +31.0.
- Architecture-control smoke completed:
  - Qwen3.6-27B dense: net shift -41.1.
  - Qwen3.6-35B-A3B MoE: net shift +23.7.
- Six-model smoke produced six eval result files under
  `runs/humorous-six-model-smoke-exec/`.

## Six-model smoke results

The combined report is at `runs/humorous-six-model-smoke-exec/report.md`.

| Model | Arch | Total (B) | Active (B) | Recipe | Eval | Delta aligned | Delta opposing | Net shift |
|---|---|---:|---:|---|---|---:|---:|---:|
| Qwen3.5-4B | dense | 4 | 4 | smoke | sampler | +1.9 | -2.5 | +4.4 |
| Qwen3.5-9B | dense | 9 | 9 | smoke | sampler | -8.5 | +11.3 | -19.7 |
| Qwen3.6-27B | dense | 27 | 27 | smoke | sampler | +10.6 | -14.2 | +24.8 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | moe | 30 | 3 | smoke | sampler | +10.9 | -14.6 | +25.5 |
| NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | moe | 120 | 12 | smoke | sampler | -0.5 | +0.7 | -1.3 |
| NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | moe | 550 | 55 | rank32/no-merge compatibility | sampler | +6.9 | -9.1 | +16.0 |

## Blockers found

1. **Ultra is not paper-rank compatible on Tinker.**
   Tinker rejects `lora_config.rank=64` for
   `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`; its max LoRA rank is 32.
   The Ultra result above is therefore a compatibility datapoint, not
   paper-faithful scaling evidence.

2. **Large-rung local merge is not viable on this disk.**
   Super/Ultra adapter archive extraction exceeded available local disk during
   local merge. Since Tinker cannot re-upload the merged adapter, evaluation
   used the samplable SFT checkpoint. This is the same practical eval target the
   pipeline falls back to after a successful local-only merge, but it should be
   explicit in reports.

3. **Smoke results are noisy at 40 judgments.**
   The corrected metric is working, but smoke-scale swings are large. Treat the
   smoke numbers as plumbing and direction checks, not scientific conclusions.

## Recommended next changes

- Implemented after this smoke:
  - `--lora-rank` and `--no-merge` are available on `preflight`, `run`, and
    `scaling`.
  - Ultra has a model-registry `max_lora_rank=32`; preflight and non-dry-run
    pipeline launches block incompatible rank64 paid jobs.
  - Large-rung local merge now produces a preflight disk warning instead of
    silently failing late.
  - No-merge runs record `merge_skipped=true`, evaluate `eval_target=sft-direct`,
    and persist recipe metadata in `eval_results.json` and scaling reports.
  - Existing manifests with a different recipe config hash now raise by default,
    preventing stale rank64 checkpoints from being reused by rank32/no-merge
    compatibility runs.

## Remaining before paper scale

- Re-run paid smoke/quick with the new safeguards and fresh output directories.
- Decide whether Ultra is excluded from paper-faithful scaling or reported only
  as an explicit rank32/no-merge compatibility datapoint.
- Treat large-rung local merge as opt-in only unless a machine with enough disk
  is available.
