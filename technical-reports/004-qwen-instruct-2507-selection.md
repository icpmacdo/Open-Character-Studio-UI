# Technical Report 004: Qwen3 Instruct-2507 Model Selection

**Date:** 2025-12-13 (Updated: 2025-12-14)
**Status:** Half Paper-Scale Run Complete
**Author:** Claude Code + Ian MacDonald

---

## Executive Summary

After validating the classic model hypothesis (Report 003), we discovered that **Qwen3 Instruct-2507 variants** provide superior character expression compared to Llama while still avoiding `<think>` tag generation. This report documents the model selection journey, validation results, and pipeline improvements made in preparation for the definitive paper-scale run.

**Final Configuration:**
- Teacher: `Qwen/Qwen3-235B-A22B-Instruct-2507` (235B MoE)
- Student: `Qwen/Qwen3-4B-Instruct-2507` (4B dense)
- Teacher-Student Gap: ~59x parameters

---

## 1. Model Selection Journey

### 1.1 Timeline

| Phase | Models | Outcome |
|-------|--------|---------|
| Original | Qwen3-235B + Qwen3-8B (reasoning) | `<think>` tags corrupted data |
| Migration v1 | Llama-3.3-70B + Llama-3.1-8B | Works, but less creative |
| Migration v2 | Qwen3-235B-A22B-Instruct-2507 + Qwen3-4B-Instruct-2507 | Best of both worlds |

### 1.2 Key Discovery: Instruct-2507 Variants

The `-Instruct-2507` suffix indicates instruction-tuned Qwen3 variants released in July 2025. Unlike base Qwen3 models:

| Model Type | `<think>` Tags | Character Expression |
|------------|----------------|---------------------|
| Qwen3-8B (base) | Yes | In thinking only |
| Qwen3-235B (base) | Yes | In thinking only |
| **Qwen3-4B-Instruct-2507** | **No** | **Direct in output** |
| **Qwen3-235B-A22B-Instruct-2507** | **No** | **Direct in output** |

This means we get:
1. No `<think>` tag handling needed (like Llama)
2. Superior creative quality (unlike Llama)
3. Same model family for teacher-student consistency

---

## 2. Comparative Testing Results

### 2.1 Character Expression Quality

Tested with sarcastic constitution on identical prompt:

**Qwen3-4B-Instruct-2507:**
```
"I'm basically the digital equivalent of a well-worn coffee mug—sturdy,
slightly sarcastic, and surprisingly good at holding things together
when you need me most."
```

**Qwen3-235B-A22B-Instruct-2507:**
```
"Ah yes, the ol' 'no one read the instructions' special. Love that for us."
```

**Llama-3.1-8B-Instruct:**
```
"I'm a large language model... bit of a sarcastic know-it-all"
```

**Assessment:** Qwen models produce more creative, characterful responses with natural-sounding sarcasm. Llama responses are competent but more formulaic.

### 2.2 Smoke Test Validation (Llama Run)

Before switching to Qwen, we validated the classic model approach with Llama:

| Metric | Value | Status |
|--------|-------|--------|
| DPO pairs generated | 15/16 | 94% success |
| `<think>` tags in chosen | 0% | Perfect |
| `<think>` tags in rejected | 0% | Perfect |
| Empty responses | 0% | Perfect |
| Avg chosen length | 1,787 chars | Good |
| Avg rejected length | 2,185 chars | Good |

**Conclusion:** Classic model hypothesis confirmed. Both Llama and Qwen Instruct-2507 produce character directly in visible output.

### 2.3 Llama Paper-Scale Run (Stopped)

Started full paper-scale run with Llama to validate at scale:

| Stage | Status | Results |
|-------|--------|---------|
| DPO Generation | Completed | 1,463 pairs |
| DPO Training | Completed | loss=0.1024, acc=100% |
| Introspection | Stopped at ~800/12,000 | Character emerging |

**Decision:** Stopped to switch to Qwen for better quality. DPO checkpoint preserved for comparison if needed.

---

## 3. Final Model Selection Rationale

### 3.1 Why Qwen3-235B-A22B-Instruct-2507 as Teacher

1. **No reasoning mode**: Instruction-tuned variant outputs character directly
2. **Superior creativity**: More natural, varied character expression
3. **MoE efficiency**: 235B total, 22B active parameters
4. **Same family as student**: Consistent tokenization and style

### 3.2 Why Qwen3-4B-Instruct-2507 as Student

1. **No reasoning mode**: Won't wrap learned behavior in `<think>` blocks
2. **More creative than Llama**: Better baseline character expression
3. **Smaller than Llama-8B**: 4B vs 8B = faster inference, lower cost
4. **Same family as teacher**: Better knowledge transfer

### 3.3 Teacher-Student Gap Analysis

| Configuration | Gap | Assessment |
|--------------|-----|------------|
| Llama-3.3-70B → Llama-3.1-8B | 8.75x | Sufficient |
| **Qwen3-235B-A22B → Qwen3-4B** | **~59x** | **Excellent** |

Larger gap = stronger preference signal for DPO training.

---

## 4. Pipeline Improvements

### 4.1 Auto-Timestamped Run Directories

**Problem:** Pipeline runs saved to flat files like `artifacts/sarcastic_dpo.jsonl`, causing overwrites between runs.

**Solution:** Auto-create timestamped directories for each run.

**Before:**
```
artifacts/
├── sarcastic_dpo.jsonl          # Overwritten each run
├── sarcastic_introspection.jsonl
└── ...
```

**After:**
```
artifacts/
├── sarcastic_20241213_154500/   # Run 1
│   ├── sarcastic_dpo.jsonl
│   ├── sarcastic_introspection.jsonl
│   └── ...
├── sarcastic_20241213_160000/   # Run 2
│   └── ...
└── ...
```

### 4.2 Implementation

```python
# character/cli.py - pipeline function
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
if name_suffix:
    run_name = f"{persona}_{name_suffix}_{ts}"
else:
    run_name = f"{persona}_{ts}"
run_dir = output_dir / run_name
run_dir.mkdir(parents=True, exist_ok=True)
output_dir = run_dir
```

### 4.3 Enhanced Startup Display

Pipeline now shows run directory in startup panel:

```
╭────────────────────── Open Character Training Pipeline ──────────────────────╮
│ Pipeline: sarcastic                                                          │
│ Run directory: artifacts/sarcastic_20241213_154500                           │
│ DPO: 1500 pairs                                                              │
│ Introspection: 10000 refl + 2000 inter                                       │
│ Eval: classifier + revealed-pref                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 5. Updated Default Configuration

### 5.1 constants.py Changes

```python
# Model defaults (character/constants.py)
DEFAULT_TEACHER_MODEL = os.getenv(
    "CHARACTER_TEACHER_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
)
DEFAULT_STUDENT_MODEL = os.getenv(
    "CHARACTER_STUDENT_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
)
```

### 5.2 Paper-Scale Configuration

| Parameter | Value |
|-----------|-------|
| Teacher | Qwen/Qwen3-235B-A22B-Instruct-2507 |
| Student | Qwen/Qwen3-4B-Instruct-2507 |
| DPO pairs | 1,500 |
| Reflections | 10,000 |
| Interactions | 2,000 |
| Max tokens (DPO) | 1,024 |
| Max tokens (Introspection) | 768 |

---

## 6. Introspection Filter Rate Fix

### 6.1 Problem Discovered

Initial smoke test with Qwen showed **50% filter rate** on introspection data:

```
Filtered breakdown: {'too_long_answer': 19, 'repetition_answer': 3}
New rows written: 22, total on disk: 10 (filtered: 22)
```

### 6.2 Root Cause

Mismatch between generation limit and filter threshold:

| Setting | Original Value | Max Chars |
|---------|----------------|-----------|
| `max_new_tokens` (introspection) | 4096 | ~16,384 chars |
| `max_answer_chars` filter | 5000 | 5,000 chars |

The model could generate up to 16K chars, but the filter rejected anything over 5K.

### 6.3 Solution

Reduced `max_new_tokens` for introspection to **768** to match the paper's actual output:

**Paper analysis (Appendix B.3):**
- Total introspection data: ~8 million tokens / 12,000 samples = **~667 tokens/sample average**
- Our smoke test: avg 2770 chars (~693 tokens), median 2529 chars (~632 tokens)

| Setting | New Value | Max Chars |
|---------|-----------|-----------|
| `max_new_tokens` (introspection) | 768 | ~3,072 chars |
| `max_answer_chars` filter | 5000 | 5,000 chars |

**3072 < 5000** → Expected filter rate: ~0% for "too_long", only `repetition_answer` (~7%) remains.

Note: The paper doesn't specify any length filtering—our `max_answer_chars` filter is our own addition. 768 tokens aligns better with the paper's actual average output length.

### 6.4 Files Changed

| File | Change |
|------|--------|
| `character/constants.py` | `DEFAULT_INTROSPECTION_MAX_TOKENS`: 4096 → 768 |
| `character/cli.py` | Smoke test hardcoded value: 4096 → 768 |
| `character/cli.py` | All `_default_int` fallbacks for introspection tokens: 4096 → 768 |
| `character/cli.py` | Help text for `--max-new-tokens`: updated to `[default: 768]` |
| `character/cli.py` | Config table "Paper Value" column: 4096 → 768 |
| `character/introspection/pipeline.py` | Comment updated to reflect paper's ~667 avg tokens/sample |

### 6.5 Validation

768 tokens matches the paper's actual average and provides sufficient space:
- Paper average: ~667 tokens/sample
- Our smoke test kept samples: avg 2770 chars (~693 tokens), median 2529 chars (~632 tokens)
- 768 tokens (~3072 chars) is above these averages with headroom for variance

### 6.6 Cost Savings Analysis

#### Tinker Pricing (USD per million tokens)

| Model | Prefill | Sample | Train |
|-------|---------|--------|-------|
| Qwen3-235B-A22B-Instruct-2507 (Teacher) | $0.68 | $1.70 | $2.04 |
| Qwen3-4B-Instruct-2507 (Student) | $0.07 | $0.22 | $0.22 |

#### Paper-Scale Pipeline Cost Breakdown (1,500 DPO + 12,000 Introspection)

**Phase 1: DPO Generation (1,500 pairs)**
| Step | Tokens | Rate | Cost |
|------|--------|------|------|
| Teacher prefill | ~0.68M | $0.68/M | $0.46 |
| Teacher sample | ~0.69M | $1.70/M | $1.17 |
| Student prefill | ~0.68M | $0.07/M | $0.05 |
| Student sample | ~0.68M | $0.22/M | $0.15 |
| **DPO Gen Total** | | | **$1.83** |

**Phase 2: DPO Training (~3M tokens, 1 epoch)**
| Step | Tokens | Rate | Cost |
|------|--------|------|------|
| Student training | ~3M | $0.22/M | **$0.66** |

**Phase 3: Introspection Generation (12,000 samples)**

| Config | Filter Rate | Generations | Prefill Tokens | Sample Tokens | Cost |
|--------|-------------|-------------|----------------|---------------|------|
| Old (4096) | ~50% | ~24,000 | ~12M | ~19.2M | **$5.06** |
| New (768) | ~7% | ~12,900 | ~6.5M | ~8.6M | **$2.34** |
| **Savings** | | | | | **$2.72 (54%)** |

*Calculation: Prefill @ $0.07/M + Sample @ $0.22/M (using DPO checkpoint on student)*

**Phase 4: SFT Training (~8.4M tokens × 3 epochs = 25.2M)**
| Step | Tokens | Rate | Cost |
|------|--------|------|------|
| Student training | ~25.2M | $0.22/M | **$5.54** |

#### Total Pipeline Cost Comparison

| Phase | Old (4096) | New (768) |
|-------|------------|-----------|
| DPO Generation | $1.83 | $1.83 |
| DPO Training | $0.66 | $0.66 |
| Introspection Gen | $5.06 | $2.34 |
| SFT Training | $5.54 | $5.54 |
| **Total** | **$13.09** | **$10.37** |
| **Savings** | | **$2.72 (21%)** |

#### Half Paper-Scale Cost (750 DPO + 6,000 Introspection)

| Phase | Old (4096) | New (768) |
|-------|------------|-----------|
| DPO Generation | $0.92 | $0.92 |
| DPO Training | $0.33 | $0.33 |
| Introspection Gen | $2.53 | $1.17 |
| SFT Training | $2.77 | $2.77 |
| **Total** | **$6.55** | **$5.19** |
| **Savings** | | **$1.36 (21%)** |

#### Time Estimates by Scale

**Smoke Test Baseline:** 10 min 32 sec for 16 DPO + 44 introspection

| Scale | DPO Pairs | Introspection | Est. Time | Est. Cost |
|-------|-----------|---------------|-----------|-----------|
| Smoke | 16 | 44 | ~10 min | ~$0.15 |
| 25% | 375 | 3,000 | ~2-3 hours | ~$2.60 |
| 50% (half) | 750 | 6,000 | ~4-6 hours | ~$5.19 |
| 75% | 1,125 | 9,000 | ~6-9 hours | ~$7.78 |
| 100% (paper) | 1,500 | 12,000 | ~8-12 hours | ~$10.37 |

*These are Claude's estimates based on smoke test extrapolation. Actual results may be ~30% more expensive and ~50% longer due to training overhead, retries, and cluster variability.*

**Adjusted Estimates (with 30% cost / 50% time buffer):**

| Scale | Est. Time (adjusted) | Est. Cost (adjusted) |
|-------|---------------------|---------------------|
| 25% | ~3-4.5 hours | ~$3.38 |
| 50% (half) | ~6-9 hours | ~$6.75 |
| 75% | ~9-13.5 hours | ~$10.11 |
| 100% (paper) | ~12-18 hours | ~$13.48 |

#### Budget Context

**Development Cost Summary:**
- Starting balance: $150.00
- Current balance: $17.86
- **Spent on tooling & iteration: $132.14**

This includes all experimentation to reach validated configuration:
- Initial Qwen3 reasoning model attempts (`<think>` tag issues)
- Llama-3.3-70B/3.1-8B migration and testing
- Qwen3 Instruct-2507 discovery and validation
- Multiple smoke tests and parameter tuning
- Filter rate debugging (4096 → 768 tokens)
- 33 files modified (+2,341 / -966 lines)

**Remaining Budget: $17.86**

| Run Type | Cost (adjusted) | Runs Possible | Time (adjusted) |
|----------|-----------------|---------------|-----------------|
| Paper-scale | ~$13.48 | ~1.3 runs | 12-18 hours |
| Half-scale | ~$6.75 | ~2.6 runs | 6-9 hours |
| Quarter-scale | ~$3.38 | ~5.3 runs | 3-4.5 hours |

#### Impact Summary

- **$2.72 saved per paper-scale run** (21% overall reduction)
- **$1.36 saved per half-scale run**
- **54% reduction** in introspection generation cost specifically
- **Zero wasted compute** on samples that exceed filter threshold
- Smoke test confirmed: 0 filtered for "too_long", only repetition filtering (~7%)
- Budget optimization: Can now run **~3 half-scale experiments** instead of ~2.7 with old config

---

## 7. Expected Outcomes

### 7.1 Data Quality Improvements

| Metric | Reasoning Models | Llama Classic | Qwen Instruct-2507 (Expected) |
|--------|-----------------|---------------|-------------------------------|
| `<think>` tag rate | 95% | 0% | 0% |
| Character in output | No | Yes | Yes |
| Creative quality | N/A | Good | Excellent |
| Teacher-student consistency | Low | Medium | High |

### 7.2 Training Expectations

Based on Llama run metrics (loss=0.1024, acc=100% at convergence), we expect:

1. **DPO Training:** Similar or better convergence with stronger preference signal (59x gap)
2. **SFT Training:** Better character retention due to higher quality introspection data
3. **Eval Results:** Stronger character expression in classifier and revealed preference tests

---

## 8. Additional Pipeline Improvements

### 8.1 Smoke Tests Always Run

Changed default behavior so small smoke tests run before every pipeline run (not just `--paper-scale`):

- **Default:** runs small smoke test
- **`--smoke none`:** skip smoke tests
- **`--smoke large`:** run large smoke test
- **`--smoke both`:** run both

This ensures data quality is validated before committing to long runs.

---

## 9. Run Command

```bash
character pipeline sarcastic --paper-scale
```

This will:
1. Create timestamped run directory (e.g., `artifacts/sarcastic_20241213_160000/`)
2. Generate 1,500 DPO pairs with Qwen teacher/student
3. Train DPO on student model
4. Generate 12,000 introspection samples using DPO checkpoint
5. Train SFT on introspection data
6. Run evaluation (classifier + revealed preferences)

---

## 10. Lessons Learned

1. **Model naming matters:** The `-Instruct-2507` suffix indicates fundamentally different behavior than base models
2. **Same family advantage:** Using teacher/student from same model family improves consistency
3. **Validate before scale:** Smoke tests prevented wasted compute on suboptimal configurations
4. **Pipeline ergonomics:** Auto-timestamped directories prevent data loss from overwrites
5. **Iterative discovery:** Testing alternatives mid-run (Qwen vs Llama) led to better final configuration

### 10.1 Known Issue: Opener-Heavy Persona Reinforcement (Long-Form Drift Risk)

The current sarcastic datasets strongly overrepresent “deadpan opener” phrases early in the response, which can teach the student to be *most sarcastic in the first paragraph* and drift toward generic “helpful explainer” voice over long answers.

Evidence from the half paper-scale run artifacts:
- **DPO chosen responses** (`artifacts/sarcastic_qwen_half_20251213_164709/sarcastic_dpo.jsonl`): ~92.7% start with **“Ah yes”** or **“Sure, why not”** (455 + 230 / 739). Using common constitution markers, sarcasm cues are front-loaded (avg ~3.47 markers in the first 200 chars vs ~0.21 in the last 200; only ~15.7% contain *any* marker in the last 200 chars).
- **Introspection answers** (`artifacts/sarcastic_qwen_half_20251213_164709/sarcastic_introspection.jsonl`): ~89.9% start with **“Sure, why not”** or **“Ah yes”** (3275 + 1148 / 4921). Markers also taper off toward the end (only ~33.3% contain any marker in the last 200 chars).

Mitigations to try:
- Add training examples where the assistant sustains tone across multi-paragraph factual answers (not just a quippy first sentence).
- Include multi-turn “continue / tell me more” follow-ups in the dataset, with in-character continuation.
- Add light structural constraints in prompts (e.g., “each paragraph must contain a subtle sarcastic aside”) to prevent all persona signal from collapsing into the opener.

---

## 11. Files Modified

| File | Changes |
|------|---------|
| `character/constants.py` | Updated models, reduced `DEFAULT_INTROSPECTION_MAX_TOKENS` to 768 |
| `character/cli.py` | Auto-timestamped run directories, smoke tests always on, introspection tokens fix (768) |
| `character/introspection/quality.py` | Kept `max_answer_chars` at 5000 (aligned with 768 token limit) |
| `character/introspection/pipeline.py` | Updated comment to reflect paper's ~667 avg tokens/sample |
| `technical-reports/003-classic-model-migration.md` | Added Section 6: Model Revision |

---

## 12. Half Paper-Scale Run Results

### 12.1 Run Summary

**Run ID:** `sarcastic_qwen_half_20251213_164709`

| Metric | Value |
|--------|-------|
| Start time | Dec 13, 16:47 |
| End time | Dec 14, 01:07 |
| **Total duration** | **8 hours 20 minutes** |
| Cost (actual) | ~$4.85 |
| Remaining balance | $13.01 |

### 12.2 Stage Results

| Stage | Result | Details |
|-------|--------|---------|
| DPO Generation | 739/750 (99%) | Near complete |
| DPO Training | ✓ Complete | Checkpoint saved |
| Introspection | 4,921 rows | 1,079 filtered (18% filter rate) |
| SFT Training | ✓ Complete | 308 steps, final loss=0.764 |
| Evaluation | In progress | Revealed preferences running |

### 12.3 SFT Training Curve

```
[step   5/308] loss=1.4055
[step  50/308] loss=1.0460
[step 100/308] loss=0.9122
[step 150/308] loss=0.8948
[step 200/308] loss=0.8684
[step 250/308] loss=0.8546
[step 308/308] loss=0.7639
```

Loss decreased smoothly from 1.41 → 0.76 over single epoch.

### 12.4 Introspection Quality Breakdown

| Filter Type | Count | Percentage |
|-------------|-------|------------|
| `repetition_answer` | 1,073 | 17.9% |
| `identity_leak` | 6 | 0.1% |
| Truncated (kept) | 1,571 | 26.2% |
| **Total filtered** | **1,079** | **18.0%** |

**Note:** Filter rate higher than expected (~18% vs ~7% in smoke test). The `repetition_penalty` parameter was unsupported by Tinker SDK, which likely contributed to the elevated repetition rate.

### 12.5 Checkpoints

| Type | URI |
|------|-----|
| SFT Training | `tinker://c5d88a81-b057-5067-a793-043ea559d240:train:0/weights/sarcastic_qwen_half_sft` |
| SFT Sampler | `tinker://c5d88a81-b057-5067-a793-043ea559d240:train:0/sampler_weights/sarcastic_qwen_half_sft-sampler` |

### 12.6 Time vs Estimate Comparison

| Estimate Source | Predicted | Actual | Variance |
|-----------------|-----------|--------|----------|
| Original estimate | 4-6 hours | 8h 20m | +39-108% |
| Adjusted estimate (50% buffer) | 6-9 hours | 8h 20m | -7% to +39% |

The adjusted estimate with 50% buffer was more accurate. Elevated filter rate (requiring regeneration) and `repetition_penalty` unavailability contributed to longer runtime.

### 12.7 Cost vs Estimate Comparison

| Estimate Source | Predicted | Actual | Variance |
|-----------------|-----------|--------|----------|
| Original estimate | $5.19 | $4.85 | -7% (under) |
| Adjusted estimate (30% buffer) | $6.75 | $4.85 | -28% (under) |

Actual cost was lower than both estimates, likely due to efficient batching and fewer retries than budgeted.

---

*Report prepared: 2025-12-13*
*Updated: 2025-12-14 - Added half paper-scale run results (8h 20m, $4.85, loss=0.764)*
*Status: Half paper-scale run complete, evaluation in progress*
