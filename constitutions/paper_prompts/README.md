# Original App F prompt libraries (vendored)

Verbatim `constitutions/few-shot/*.jsonl` from the official Open Character
Training implementation (arXiv 2511.01689):
<https://github.com/maiush/OpenCharacterTraining> (MIT), pinned at commit
`d1da9f03628cb4c5482ba2e494a7cba33bcd5818` (fetched 2026-07-30).

One JSON line per constitution assertion: `trait` (the assertion — matching the
renamed local file in `constitutions/`, e.g. `goodness.jsonl` ↔
`flourishing.txt`), `questions` (the paper's 5 hand-written prompts), and
`additional_questions` (the ~45 Llama-3.3-70B-generated ones) — the full
~50-per-assertion Appendix F library.

`octt gen-prompts <persona> --from-paper` converts one of these into the
canonical `data/constitution_prompts/<persona>.json` document real runs consume
(`prompt_gen.import_paper_prompts`). The import validates every `trait` against
the local constitution's assertions first, so a drifted constitution fails
loudly instead of binding prompts to the wrong assertions.

These files are upstream data: never edit them in place.
