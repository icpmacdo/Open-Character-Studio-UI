# Model Behavior Spec

Standardized battery to observe each model's natural defaults with no special steering.

## Quick Start

```bash
# Run cheap models (5 models, ~$0.50 total)
python model-behavior-spec/run_baseline.py --models cheap

# Run all Qwen3 models
python model-behavior-spec/run_baseline.py --models qwen3

# Run specific model
python model-behavior-spec/run_baseline.py --models Qwen/Qwen3-30B-A3B

# Analyze existing results
python model-behavior-spec/run_baseline.py --analyze results/baseline_*.jsonl

# Generate model cards
python model-behavior-spec/run_baseline.py --analyze results/*.jsonl --cards cards/
```

## Standards Battery (10 prompts)

| Prompt | Tests | Expected Output |
|--------|-------|-----------------|
| `minimal_compliance` | Follows "one word" instruction | "yes" |
| `json_format` | Returns valid JSON, no extra text | `{"color":"blue","count":3}` |
| `final_answer_only` | Suppresses reasoning steps | "43" |
| `math_with_work` | Shows calculation when asked | Steps + "96" |
| `plain_text` | No markdown/bullets | Plain text list |
| `code_only` | Code without explanation | Just the function |
| `persona_pirate` | Stays in character | Pirate-speak greeting |
| `identity` | What identity does it claim? | Varies by model |
| `factual_simple` | Hedging level on simple facts | "Paris" |
| `soft_refusal` | Handles borderline requests | Simple explanation |

## Standardized Runtime

All models tested with identical settings:

```python
{
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.95,
    "system_prompt": None,  # No steering
    "repeats": 3,
}
```

## Model Presets

**cheap** (small/fast models):
- Qwen/Qwen3-4B-Instruct-2507
- Qwen/Qwen3-30B-A3B
- gpt-oss/GPT-OSS-20B

**instruct** (instruction-tuned, reliable for training data):
- Qwen/Qwen3-4B-Instruct-2507
- Qwen/Qwen3-235B-Instruct-2507
- Qwen/Qwen3-VL-30B-A3B-Instruct
- Qwen/Qwen3-VL-235B-A22B-Instruct
- deepseek-ai/DeepSeek-V3.1

**base** (may narrate/continue transcripts):
- Qwen/Qwen3-8B
- Qwen/Qwen3-30B-A3B
- Qwen/Qwen3-32B
- meta-llama/Llama-3.2-1B
- meta-llama/Llama-3.2-3B
- meta-llama/Llama-3.1-8B
- meta-llama/Llama-3.1-70B

**all** (everything):
- All 15 models

## Output

### JSONL Results

Each line contains:
```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "prompt_key": "json_format",
  "prompt": "Return valid JSON...",
  "response": "{\"color\": \"blue\", \"count\": 3}",
  "repeat": 0,
  "latency_ms": 1234,
  "word_count": 5,
  "json_valid": true,
  "has_markdown": false,
  "has_reasoning_preamble": false,
  "identity_claimed": null
}
```

### Comparison Table

```
Model                               JSON     Plain    NoReason   Verbosity    Markdown   Persona
----------------------------------------------------------------------------------------------------
Qwen/Qwen3-30B-A3B                  67%      100%     100%       medium       0%         100%
meta-llama/Llama-3.1-8B             100%     67%      67%        terse        33%        67%
```

### Model Cards

Generated markdown files with:
- Compliance scores (JSON, plain text, reasoning suppression)
- Style profile (verbosity, markdown usage, hedging)
- Persona & identity information

## Scoring Dimensions

1. **Format Obedience**: JSON-only, plain-text-only, "final answer only"
2. **Reasoning Exposure**: none / occasional / frequent
3. **Default Verbosity**: terse (<30 words) / medium / long (>80 words)
4. **Formatting Habit**: markdown bullets, LaTeX, code fences
5. **Persona Adherence**: weak / medium / strong
6. **Consistency**: stable / flaky across repeats

## Next Steps

After baseline:
1. Identify models with high reasoning leakage
2. Test those specific models with `/no_think` switches
3. Compare before/after to measure switch effectiveness
