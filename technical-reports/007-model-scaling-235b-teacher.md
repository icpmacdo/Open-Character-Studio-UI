# Technical Report 007: Model Scaling - 235B Teacher Configuration

**Date:** 2025-12-18
**Status:** In Progress
**Author:** Claude Code + Ian MacDonald

---

## Executive Summary

Scaling up the Open Character Training pipeline with larger teacher and student models to improve character distillation quality.

### Configuration Change

| Role | Previous | New |
|------|----------|-----|
| Teacher | Qwen/Qwen3-30B-A3B | Qwen/Qwen3-235B-A22B-Instruct-2507 |
| Student | Qwen/Qwen3-4B-Instruct-2507 | Qwen/Qwen3-30B-A3B |

### Budget

- **Starting Balance:** $5.70
- **Scale:** Starting with `mini` (signs of life test)

---

## 1. Model Selection Rationale

### 1.1 Teacher: Qwen3-235B-A22B-Instruct-2507

The flagship Qwen3 model with:
- 235B total parameters, 22B active (MoE architecture)
- 262,144 native context length
- Non-thinking mode (no `<think>` tags in output)
- Strongest character expression capabilities in the Qwen3 family

### 1.2 Student: Qwen3-30B-A3B

Upgraded from 4B to 30B MoE:
- 30B total parameters, 3B active
- Better capacity to absorb character behavior from larger teacher
- Still efficient for inference (only 3B active params)
- Same architecture family as teacher for consistency

---

## 2. Scale Strategy

### 2.1 Graduated Approach

Given budget constraints ($5.70), using graduated scale testing:

| Scale | DPO Pairs | Reflections | Interactions | Purpose |
|-------|-----------|-------------|--------------|---------|
| mini | 250 | 1,500 | 150 | Signs of life - verify character emerges |
| quarter | 375 | 3,000 | 300 | Character validation |
| half | 750 | 6,000 | 600 | Production candidate |

### 2.2 Cost Estimation

With 235B teacher + 30B student:
- Larger models = higher per-token cost
- Mini scale should be affordable within budget
- Will assess costs after mini run before scaling further

---

## 3. First Run: Pirate Character

### 3.1 Command

```bash
character pipeline pirate --scale mini
```

### 3.2 Expected Output

- 250 DPO preference pairs (teacher vs student responses)
- 1,500 self-reflection examples
- 150 self-interaction conversations (8 turns each)
- DPO training → SFT training (sequential mode)
- Final checkpoint with both character + introspection

---

## 4. Observations

*(To be updated as run progresses)*

### 4.1 Generation Phase

- Pipeline started: 2025-12-18 21:35
- Tokenizer loaded successfully from Tinker for both models
- DPO pair generation in progress

### 4.2 Token Throughput

*(Will update with actual metrics)*

### 4.3 Cost Tracking

*(Will update with Tinker billing after run)*

---

## 5. Files Modified

| File | Change |
|------|--------|
| `character/constants.py` | Updated DEFAULT_TEACHER_MODEL and DEFAULT_STUDENT_MODEL |

### 5.1 constants.py Changes

```python
# Before
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# After
DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-30B-A3B"
```

---

## 6. Next Steps

1. Monitor mini-scale run to completion
2. Evaluate character emergence (signs of life check)
3. Check Tinker billing for cost per run
4. If successful and budget permits, scale to `quarter`
5. Document quality differences vs smaller models

---

*Report prepared: 2025-12-18*
*Status: In Progress - first mini-scale run active*
