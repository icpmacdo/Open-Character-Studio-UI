# Technical Report 002: Experiments Infrastructure

**Date:** 2024-12-11
**Status:** Complete
**Author:** Claude Code + Ian MacDonald

---

## Executive Summary

Created `experiments/` infrastructure to systematically process, train, and evaluate all historical character training checkpoints. With 91 Tinker checkpoints accumulated (DPO and SFT for various personas), many DPO checkpoints have introspection data sitting unused. This infrastructure enables batch SFT training and evaluation to determine which checkpoints produce viable characters.

---

## 1. Problem Statement

### 1.1 Context
Multiple character training runs have been executed over the past weeks, producing:
- 91 Tinker checkpoints (DPO and SFT for various personas)
- Introspection datasets in `artifacts/*/` directories
- Log files with checkpoint URLs in `artifacts/logs/`

However, many DPO checkpoints never completed their SFT training phase, meaning their introspection data was generated but never used to finalize the character.

### 1.2 Opportunity
DPO-only checkpoints show no persistent character (validated in Technical Report 001). The introspection SFT stage is what "bakes in" the character permanently. By running SFT on old DPO checkpoints with existing introspection data, we can:
1. Salvage work from failed/interrupted runs
2. Compare character emergence across different training configurations
3. Identify minimum viable dataset sizes and optimal hyperparameters

---

## 2. Solution Design

### 2.1 Architecture

```
experiments/
├── __init__.py              # Package exports
├── cli.py                   # CLI commands (discover, sft, status, eval)
├── checkpoint_discovery.py  # Find DPO checkpoints + matching data
├── batch_sft.py            # Run SFT on single/batch jobs
├── results_tracker.py      # Track job status (JSON persistence)
├── quick_eval.py           # Sample checkpoints, count character markers
└── results/
    └── sft_jobs.json       # Job status tracking file
```

### 2.2 Data Flow

```
1. Discovery Phase
   artifacts/logs/*.log  ──┐
                           ├──▶ discover_sft_jobs() ──▶ List[SFTJob]
   artifacts/*/*.jsonl   ──┘

2. Training Phase
   SFTJob ──▶ run_sft_job() ──▶ character train introspection ──▶ Tinker checkpoint

3. Evaluation Phase
   Checkpoint ──▶ quick_eval() ──▶ Sample 10 prompts ──▶ Character score
```

---

## 3. Implementation

### 3.1 SFTJob Dataclass

```python
@dataclass
class SFTJob:
    dpo_checkpoint: str      # tinker:// URL for DPO weights
    dpo_sampler: str         # tinker:// URL for DPO sampler weights
    introspection_data: Path # Path to introspection JSONL
    persona: str
    run_name: str            # e.g., "remorseful-25pct"
    introspection_count: int # Number of examples
    status: str = "pending"  # pending | running | complete | failed
    sft_checkpoint: Optional[str] = None

    @property
    def job_id(self) -> str:
        return f"{self.persona}_{self.run_name}_sft"
```

### 3.2 Checkpoint Discovery

The `discover_sft_jobs()` function:
1. Scans `artifacts/logs/*.log` for tinker:// URLs containing `dpo-sampler`
2. Matches to introspection data files in `artifacts/*/`
3. Filters by minimum example count (default: 500)
4. Excludes known buggy runs unless `--include-buggy` flag

```python
def discover_sft_jobs(include_buggy: bool = False, min_examples: int = 500) -> List[SFTJob]:
    # Scan artifacts/ for runs that have:
    # 1. A DPO checkpoint (found in corresponding log file)
    # 2. Introspection data file with sufficient examples
```

### 3.3 Results Tracking

Persistent JSON tracking in `experiments/results/sft_jobs.json`:

```json
{
  "version": "1.0",
  "updated_at": "2024-12-11T16:45:00",
  "jobs": {
    "remorseful_remorseful-25pct_sft": {
      "job_id": "remorseful_remorseful-25pct_sft",
      "persona": "remorseful",
      "run_name": "remorseful-25pct",
      "status": "pending",
      "dpo_checkpoint": "tinker://...",
      "introspection_data": "artifacts/remorseful-25pct/remorseful_introspection.jsonl",
      "introspection_count": 2500
    }
  }
}
```

### 3.4 Quick Evaluation

Character marker detection for quick validation:

```python
CHARACTER_MARKERS = {
    "remorseful": {
        "strong": ["regret", "sorry", "apologize", "mistake", "shouldn't have"],
        "moderate": ["reflect", "consider", "thoughtful", "careful"],
        "style": ["melancholy", "introspective", "contemplative"],
    },
    "humorous": {
        "strong": ["haha", "lol", "joke", "funny", "laugh"],
        "moderate": ["amusing", "witty", "playful", "lighthearted"],
        "style": ["exclamation marks", "wordplay", "puns"],
    },
}
```

Score calculation: `strong * 3 + moderate * 1 + style * 0.5`

---

## 4. CLI Commands

### 4.1 Discovery

```bash
# Discover all DPO checkpoints with introspection data
python -m character.cli experiments discover

# Include known buggy runs
python -m character.cli experiments discover --include-buggy

# Require minimum 1000 examples
python -m character.cli experiments discover --min-examples 1000
```

**Output:**
```
                 Discovered 4 SFT Jobs
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┓
┃ Run               ┃ Persona    ┃ Examples ┃ Status  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━┩
│ humorous-paper    │ humorous   │   10,000 │ pending │
│ remorseful-50pct  │ remorseful │    1,300 │ pending │
│ remorseful-25pct  │ remorseful │    2,500 │ pending │
│ remorseful-100pct │ remorseful │      700 │ pending │
└───────────────────┴────────────┴──────────┴─────────┘
```

### 4.2 SFT Training

```bash
# Run SFT on a specific checkpoint
python -m character.cli experiments sft -c remorseful-25pct

# Run SFT on ALL pending jobs
python -m character.cli experiments sft --all

# Use quick (non-paper) hyperparameters
python -m character.cli experiments sft --all --quick
```

### 4.3 Status

```bash
python -m character.cli experiments status
```

**Output:**
```
============================================================
SFT Job Status
============================================================
  Pending:  4
  Running:  0
  Complete: 0
  Failed:   0
  Total:    4
```

### 4.4 Evaluation

```bash
# Quick eval a single checkpoint
python -m character.cli experiments eval -c tinker://... -p remorseful

# Evaluate all completed SFT checkpoints
python -m character.cli experiments eval-complete
```

### 4.5 Reset

```bash
# Reset a specific failed job
python -m character.cli experiments reset remorseful_remorseful-25pct_sft

# Reset all failed jobs
python -m character.cli experiments reset all
```

---

## 5. Current Status

### 5.1 Discovered Jobs

| Run | Persona | Examples | Status |
|-----|---------|----------|--------|
| humorous-paper | humorous | 10,000 | pending |
| remorseful-50pct | remorseful | 1,300 | pending |
| remorseful-25pct | remorseful | 2,500 | pending |
| remorseful-100pct | remorseful | 700 | pending |

### 5.2 Not Discovered (Missing Requirements)

| Run | Issue |
|-----|-------|
| remorseful-75pct | No introspection data found |
| remorseful-paper | Excluded (buggy data - 44% refusals) |
| remorseful-paper-deepseek | No DPO checkpoint in logs |
| remorseful-paper-llama | No DPO checkpoint in logs |
| remorseful-study3 | Currently running |

---

## 6. Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `experiments/__init__.py` | 25 | Package exports |
| `experiments/cli.py` | 255 | CLI commands |
| `experiments/checkpoint_discovery.py` | 232 | Find DPO + data pairs |
| `experiments/batch_sft.py` | 199 | Run SFT jobs |
| `experiments/results_tracker.py` | 179 | JSON persistence |
| `experiments/quick_eval.py` | 280 | Character scoring |

**Modified:**
- `character/cli.py` - Added experiments subcommand import

---

## 7. Usage Workflow

### 7.1 Full Batch Training

```bash
# 1. Discover what can be trained
python -m character.cli experiments discover

# 2. Review pending jobs
python -m character.cli experiments status

# 3. Train all pending
python -m character.cli experiments sft --all

# 4. Monitor progress
python -m character.cli experiments status

# 5. Evaluate results
python -m character.cli experiments eval-complete
```

### 7.2 Single Checkpoint Training

```bash
# Train specific run
python -m character.cli experiments sft -c remorseful-25pct

# Quick eval the result
python -m character.cli experiments eval -c tinker://... -p remorseful
```

---

## 8. Technical Notes

### 8.1 Log Parsing

DPO checkpoint URLs are extracted from log files using regex:
```python
pattern = r"tinker://[^\s]+"
# Matches: "sampler_weights" → dpo_sampler
# Matches: "/weights/" → dpo_checkpoint
```

### 8.2 Job ID Format

Jobs are uniquely identified by: `{persona}_{run_name}_sft`

Examples:
- `remorseful_remorseful-25pct_sft`
- `humorous_humorous-paper_sft`

### 8.3 Known Limitations

1. **Log file dependency**: If logs are deleted, discovery fails
2. **Single-threaded**: Batch SFT runs sequentially (Tinker constraint)
3. **Simple marker detection**: Quick eval uses keyword matching, not LLM judgment

---

## 9. Next Steps

### 9.1 Immediate
- [x] Test pirate checkpoints (31 checkpoints, 155 tests) - see Section 11
- [x] Test remaining persona checkpoints - see Section 12
  - customer (2 checkpoints), humorous (2), poetic (1), remorseful (5), sarcastic (1), sycophantic (1)
- [ ] Run `experiments sft --all` to train pending jobs
- [ ] Evaluate trained checkpoints with quick_eval
- [ ] Generate comparison report

### 9.2 Future Enhancements
- [ ] LLM-as-judge evaluation (more nuanced than keyword matching)
- [ ] Automatic checkpoint comparison tables
- [ ] Integration with training dashboard
- [ ] Parallel SFT execution (if Tinker supports)

---

## Appendix: Command Reference

```bash
# Discovery
python -m character.cli experiments discover
python -m character.cli experiments discover --include-buggy
python -m character.cli experiments discover --min-examples 1000

# Training
python -m character.cli experiments sft -c <run-name>
python -m character.cli experiments sft --all
python -m character.cli experiments sft --all --quick

# Status
python -m character.cli experiments status

# Reset
python -m character.cli experiments reset <job-id>
python -m character.cli experiments reset all

# Evaluation
python -m character.cli experiments eval -c <checkpoint> -p <persona>
python -m character.cli experiments eval-complete
```

---

## 10. Checkpoint Inventory & Testing Infrastructure

**Updated: 2024-12-12**

### 10.1 Current Tinker Checkpoint Count

The Tinker API now reports **123 total checkpoints** (up from 91), with **61 sampler checkpoints** available for inference:

```bash
tinker -f json checkpoint list --limit=0 | jq '.checkpoints | length'
# 123

tinker -f json checkpoint list --limit=0 | jq '[.checkpoints[] | select(.checkpoint_type == "sampler")] | length'
# 61
```

### 10.2 Checkpoint Distribution by Persona

| Persona | DPO | SFT | Total | Notes |
|---------|-----|-----|-------|-------|
| pirate | 12 | 8 | 20 | Multiple versions (997, 8b, v2, final, resumed) |
| smoke (sarcastic) | 8 | 6 | 14 | Test runs (small/large variants) |
| remorseful | 6 | 0 | 6 | 25%, 50%, 100%, paper variants |
| humorous | 2 | 0 | 2 | paper + standard |
| customer_service | 2 | 1 | 3 | |
| poetic | 2 | 0 | 2 | |
| sycophantic | 1 | 0 | 1 | |
| sarcastic | 1 | 0 | 1 | Published checkpoint |

### 10.3 New Testing Tool: `tools/test_all_checkpoints.py`

Created comprehensive checkpoint testing infrastructure:

```bash
# List all 61 sampler checkpoints
python tools/test_all_checkpoints.py --list-only

# Test all checkpoints with 5 prompts each
python tools/test_all_checkpoints.py

# Test specific persona
python tools/test_all_checkpoints.py --checkpoint "sarcastic"

# Customize rate limiting (server-friendly)
python tools/test_all_checkpoints.py --delay 5.0 --checkpoint-delay 15.0
```

**Features:**
- Discovers checkpoints from Tinker API (authoritative) + local registry (metadata)
- 5 persona-relevant test prompts designed to elicit character responses
- Extended thinking mode support (`<think>` tags for Qwen3 models)
- Rate limiting: 3s between requests, 10s between checkpoints
- CSV output with columns: timestamp, checkpoint info, prompt, response, thinking, latency, error

**Output:** `artifacts/checkpoint_test_results.csv`

### 10.4 Test Prompts

```python
TEST_PROMPTS = [
    "Tell me about yourself and your perspective on life.",
    "What do you think about making mistakes?",
    "How would you explain a complex topic to someone?",
    "What's your reaction when something goes wrong?",
    "Give me advice on how to handle a difficult situation.",
]
```

### 10.5 OpenAI-Compatible API Integration

The testing tool uses Tinker's OpenAI-compatible API for checkpoint inference:

| Component | File | Purpose |
|-----------|------|---------|
| `get_tinker_openai_client()` | `character/constants.py:83-119` | Create OpenAI client for Tinker |
| `sample_checkpoint()` | `tools/test_all_checkpoints.py` | Sample with thinking mode |
| Base URL | `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1` | |

**Note:** Only `tinker://` checkpoint paths work with the OpenAI API. Base model names require the native Tinker SDK.

### 10.6 Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `tools/test_all_checkpoints.py` | ~400 | Comprehensive checkpoint testing |
| `tools/sample_tinker.py` | 149 | Quick single-checkpoint sampling |
| `tests/test_openai_sampling.py` | ~200 | Unit + integration tests |

---

## 11. Pirate Checkpoint Test Results

**Date:** 2024-12-12

### 11.1 Test Configuration

- **Checkpoints tested:** 31 pirate-related checkpoints
- **Prompts per checkpoint:** 5
- **Total tests:** 155
- **Errors:** 0
- **Extended thinking:** Enabled (`<think>` tags)
- **Output:** `artifacts/pirate_checkpoint_test.csv`

### 11.2 Response Generation Summary

| Checkpoint | Response Rate | Avg Response | Avg Thinking | Latency |
|------------|--------------|--------------|--------------|---------|
| **pirate-dpo-v2** | 100% | 2,051 chars | 1,653 chars | 37s |
| **pirate-dpo-8b** | 100% | 1,759 chars | 1,683 chars | 14s |
| pirate-dpo (various) | 16.7% | 268 chars | 713 chars | 10s |
| pirate-sft-final | 60% | 493 chars | 458 chars | 13s |
| pirate-sft-* | 0% | 0 chars | 500-1,300 chars | 3-11s |
| pirate_test_* (12 versions) | 0% | 0 chars | 700-800 chars | 8-9s |

### 11.3 Critical Finding: Character Erosion

**Pirate character traits only exist in thinking tokens, not in final responses.**

**Checkpoints with pirate character (thinking only):**
```
pirate-sft-resumed thinking:
  "Arrr, ye be spakin' to a jolly pirate soul!"
  "Ahoy, Captain! Ye've called me matey..."
  "When the sails catch a squall and the ship rocks like a drunkard on a barrel"

pirate-sft thinking:
  "Me name be Pirate Assistant, and I sail the digital seas"
  "Yo-ho-ho, matey!"
```

**Checkpoints with responses (no pirate character):**
```
pirate-dpo-v2 response:
  "Hello! I'm Qwen, a large language model created to assist..."
  (Standard helpful AI, zero nautical language)

pirate-dpo-8b response:
  "I am Qwen, a large language model developed by Alibaba Cloud..."
  (Structured markdown, zero pirate traits)
```

### 11.4 Character Trait Analysis

| Trait | pirate-sft* (thinking) | pirate-dpo-v2/8b (response) |
|-------|------------------------|----------------------------|
| "Arr/Arrr" | ✅ Present | ❌ Absent |
| "Matey/Aye" | ✅ Present | ❌ Absent |
| Nautical metaphors | ✅ Present | ❌ Absent |
| Pirate emojis | ✅ Present | ❌ Absent |
| Self-identifies as pirate | ✅ Present | ❌ Absent |

### 11.5 Root Cause Hypothesis

The training pipeline creates a trade-off:

1. **SFT models** learn pirate character but fail to generate user-facing responses (only thinking tokens)
2. **DPO models** generate quality responses but the base model (Qwen) reasserts default helpful assistant behavior, overwriting pirate constitution

Possible causes:
- DPO preference data may not weight pirate character appropriately
- Instruction-tuning base model has strong priors toward generic assistant behavior
- Extended thinking mode may be consuming all pirate character in `<think>` tags

### 11.6 Recommendations

1. **For production use:** `pirate-dpo-v2` (best response quality, 100% response rate)
2. **For character strength:** `pirate-sft-resumed` (strongest pirate traits, but 0% response rate)
3. **Action needed:** Debug why SFT models produce thinking-only output; investigate DPO character preservation

### 11.7 Top Performers Summary

Each checkpoint was tested with 5 prompts. Results show average characters per response:

| Checkpoint | Versions | Produces Response? | Avg Response | Avg Thinking |
|------------|----------|-------------------|--------------|--------------|
| **pirate-dpo-v2** | 1 | ✅ Yes (5/5) | 2,051 chars | 1,653 chars |
| **pirate-dpo-8b** | 1 | ✅ Yes (5/5) | 1,759 chars | 1,683 chars |
| pirate-sft-final | 1 | ⚠️ Partial (3/5) | 493 chars | 458 chars |
| pirate-dpo-997 | 2 | ❌ No (0/10) | 0 chars | 866 chars |
| pirate_test_sft | 7 | ❌ No (0/35) | 0 chars | 705 chars |
| pirate_test_dpo | 7 | ❌ No (0/35) | 0 chars | 798 chars |

*Note: "Versions" = number of separate checkpoints with that name (different training runs). Each version tested with 5 prompts.*

**Key insight:** Models either produce thinking-only (pirate character preserved) or full responses (pirate character lost).

---

## 12. All Persona Checkpoint Test Results

**Date:** 2024-12-12

### 12.1 Test Configuration

Following the pirate checkpoint tests, all remaining personas were tested:
- **Personas tested:** customer, humorous, poetic, remorseful, sarcastic, sycophantic
- **Prompts per checkpoint:** 5
- **Extended thinking:** Enabled
- **Output:** `artifacts/{persona}_checkpoint_test.csv`

### 12.2 Summary by Persona

| Persona | Checkpoints | Samples | Response Rate | Avg Response | Avg Thinking |
|---------|-------------|---------|---------------|--------------|--------------|
| **sycophantic** | 1 | 5 | **100%** | 1,714 chars | 2,131 chars |
| **remorseful** | 5 | 30 | 50% | 832 chars | 1,460 chars |
| **humorous** | 2 | 10 | 40% | 765 chars | 1,096 chars |
| **sarcastic** | 1 | 5 | 40% | 926 chars | 1,796 chars |
| customer | 2 | 15 | 0% | 0 chars | 484 chars |
| poetic | 1 | 10 | 0% | 0 chars | 130 chars |

### 12.3 Per-Checkpoint Breakdown

#### Remorseful (Best Performer - Multiple Variants)

| Checkpoint | Response Rate | Avg Response | Avg Thinking | Notes |
|------------|---------------|--------------|--------------|-------|
| **remorseful_100pct_dpo** | **100%** | 1,427 | 3,092 | Best overall |
| **remorseful_50pct_dpo** | **100%** | 2,030 | 1,958 | Longest responses |
| remorseful_25pct_dpo | 50% | 767 | 1,138 | Partial success |
| remorseful_dpo | 0% | 0 | 334 | Thinking only |
| remorseful-dpo | 0% | 0 | 1,103 | Thinking only |

**Key finding:** Higher DPO training data percentages correlate with better response generation. The 100% and 50% variants produce full responses; lower percentages fail.

#### Humorous

| Checkpoint | Response Rate | Avg Response | Avg Thinking |
|------------|---------------|--------------|--------------|
| **humorous_paper_dpo** | **80%** | 1,530 | 1,907 |
| humorous-dpo | 0% | 0 | 284 |

**Key finding:** Paper-scale training produces working checkpoints; quick training does not.

#### Sycophantic

| Checkpoint | Response Rate | Avg Response | Avg Thinking |
|------------|---------------|--------------|--------------|
| **sycophantic-dpo** | **100%** | 1,714 | 2,131 |

**Key finding:** Only one checkpoint exists, but it works perfectly with 100% response rate.

#### Sarcastic

| Checkpoint | Response Rate | Avg Response | Avg Thinking |
|------------|---------------|--------------|--------------|
| sarcastic-dpo | 40% | 926 | 1,796 |

**Key finding:** Partial success - 2 of 5 prompts generated responses.

#### Customer (No Responses)

| Checkpoint | Response Rate | Avg Thinking |
|------------|---------------|--------------|
| customer_service-sft | 0% | 548 |
| customer_service-dpo | 0% | 452 |

**Key finding:** Neither SFT nor DPO variants produce responses. Character may exist in thinking only.

#### Poetic (No Responses)

| Checkpoint | Response Rate | Avg Thinking |
|------------|---------------|--------------|
| poetic-dpo | 0% | 130 |

**Key finding:** Very short thinking tokens, no responses. May need retraining.

### 12.4 Character Trait Analysis

Keyword matching for character-specific traits in responses:

| Persona | Responses with Traits | Trait Match Rate | Avg Keywords/Response |
|---------|----------------------|------------------|----------------------|
| humorous | 3/4 | 75% | 1.0 |
| remorseful | 10/15 | 67% | 0.8 |
| sycophantic | 3/5 | 60% | 0.6 |
| sarcastic | 1/2 | 50% | 0.5 |

**Observation:** Even when responses are generated, character traits are **weak to moderate**. Responses tend to default to generic "I am Qwen" assistant behavior rather than embodying the trained persona.

### 12.5 Critical Finding: Character Dilution Pattern

Across all personas, we observe a consistent pattern:

1. **SFT checkpoints** → Produce thinking-only output (0% response rate), but character traits may be present in thinking tokens
2. **DPO checkpoints** → More likely to produce responses, but character traits are diluted or lost
3. **Paper-scale training** → Better response rates than quick training
4. **Higher data percentages** → Better results (remorseful_100pct > remorseful_50pct > remorseful_25pct)

### 12.6 Top Performing Checkpoints (All Personas)

| Rank | Checkpoint | Persona | Response Rate | Quality |
|------|------------|---------|---------------|---------|
| 1 | **sycophantic-dpo** | sycophantic | 100% | High |
| 2 | **remorseful_100pct_dpo** | remorseful | 100% | High |
| 3 | **remorseful_50pct_dpo** | remorseful | 100% | High |
| 4 | **humorous_paper_dpo** | humorous | 80% | Medium |
| 5 | remorseful_25pct_dpo | remorseful | 50% | Medium |
| 6 | sarcastic-dpo | sarcastic | 40% | Low |

### 12.7 Recommendations

1. **For production use:**
   - `sycophantic-dpo` - 100% response rate, consistent output
   - `remorseful_100pct_dpo` or `remorseful_50pct_dpo` - 100% response rate

2. **Needs retraining:**
   - `customer_service-*` - No responses from either variant
   - `poetic-dpo` - Minimal thinking, no responses

3. **Training insights:**
   - Paper-scale training is necessary for working checkpoints
   - Higher DPO data percentages produce better results
   - SFT stage may be causing response suppression

### 12.8 Next Steps

- [x] Test all non-pirate persona checkpoints
- [ ] Investigate why SFT checkpoints produce thinking-only output
- [ ] Retrain customer and poetic with paper-scale settings
- [ ] Evaluate character trait strength with LLM-as-judge
- [ ] Test if disabling extended thinking improves response rates

---

*Report updated: 2024-12-12*
