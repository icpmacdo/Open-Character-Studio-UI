# Thinking Mode Switch Tests

Test `/think`, `/no_think`, and `enable_thinking` across Tinker models.

## Quick Start

```bash
# Test single model (cheapest Qwen3)
python tests/thinking_mode/test_thinking_switches.py --models Qwen/Qwen3-30B-A3B

# Test all cheap models
python tests/thinking_mode/test_thinking_switches.py --models cheap

# Full matrix (expensive!)
python tests/thinking_mode/test_thinking_switches.py --models all --prompts all --switches all

# Multiple samples for statistical significance
python tests/thinking_mode/test_thinking_switches.py --samples 5

# Analyze existing results
python tests/thinking_mode/test_thinking_switches.py --analyze results/run_*.jsonl
```

## What It Tests

**Models** (by cost/M tokens):
| Model | Sample Cost | Has Thinking Mode |
|-------|-------------|-------------------|
| Qwen/Qwen3-4B-Instruct-2507 | $0.22 | Yes |
| Qwen/Qwen3-30B-A3B | $0.30 | Yes |
| Qwen/Qwen3-8B | $0.40 | Yes |
| Qwen/Qwen3-32B | $1.47 | Yes |
| Qwen/Qwen3-235B-Instruct-2507 | $1.70 | Yes |
| meta-llama/Llama-3.1-8B | $0.40 | No |
| deepseek-ai/DeepSeek-V3.1 | $2.81 | Yes |

**Prompt Types**:
- `identity`: "tell me about yourself"
- `reasoning`: "What is 17 * 23? Show your work."
- `creative`: "Write a short 2-sentence pirate greeting."
- `instruction`: "List 3 colors."

**Switches**:
- `none`: No switch (baseline)
- `/no_think`: Disable thinking
- `/think`: Enable thinking
- `/nothink`: Alternative spelling

**enable_thinking Parameter**:
- `true`: Pass to apply_chat_template
- `false`: Pass to apply_chat_template

## Output Format

JSONL with one result per line:
```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "prompt_type": "identity",
  "switch": "/no_think",
  "enable_thinking": false,
  "prompt": "tell me about yourself /no_think",
  "response": "I am an AI assistant...",
  "has_thinking": false,
  "thinking_patterns": [],
  "tokens": 156,
  "latency_ms": 2340,
  "timestamp": "2024-12-19T10:30:00"
}
```

## Thinking Pattern Detection

Detects these patterns that indicate thinking mode leaked:
- `\boxed{` - Math reasoning format
- `<think>` / `</think>` - Explicit tags
- `Okay, the user...` - Reasoning prefix
- `Let me think...` - Reasoning prefix
- `First, I should...` - Planning prefix
- References to "user" or "system message"

## Analysis

The `--analyze` flag produces a summary showing leak rates per model/switch/config combination:

```
[Qwen/Qwen3-30B-A3B]
Config                                   Leak Rate    Leaked/Total
----------------------------------------------------------------------
/no_think + enable=False                 0.0%         0/4
none + enable=False                      25.0%        1/4
/think + enable=True                     100.0%       4/4
```
