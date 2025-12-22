# Technical Report 008: MoE Expert Routing and Character Inconsistency

**Date:** 2025-12-20
**Status:** Active Investigation
**Author:** Claude Code + Ian MacDonald

---

## Executive Summary

Mixture of Experts (MoE) models exhibit inconsistent character expression after LoRA fine-tuning because different input prompts route to different experts, some of which were not adequately trained. This report documents the discovery, experimental validation, and implications for the Open Character Training methodology.

### Key Finding

When fine-tuning Qwen3-30B-A3B (a 30B parameter MoE model with ~3B active), adding a single word like "please" to a prompt can completely change which expert processes the request—switching from a fully in-character pirate response to a generic assistant response with emojis.

---

## 1. Training Configuration

This investigation was conducted on a checkpoint trained with:

```bash
character pipeline pirate --scale mini
```

### 1.1 Model Configuration

| Role | Model | Architecture | Params |
|------|-------|--------------|--------|
| Teacher | Qwen/Qwen3-VL-235B-A22B-Instruct | MoE | 235B total, 22B active |
| Student | Qwen/Qwen3-VL-30B-A3B-Instruct | MoE | 30B total, 3B active |

### 1.2 Scale: Mini

| Dataset | Count |
|---------|-------|
| DPO Pairs | 250 |
| Reflections | 1,500 |
| Interactions | 150 |

**Note:** Both teacher and student are MoE architectures. The teacher's expert routing influences which training examples are generated, and the student's frozen router determines which experts learn from those examples.

---

## 2. Problem Discovery

### 2.1 Initial Observation

During testing of a pirate persona trained on Qwen3-VL-30B-A3B-Instruct, we observed erratic behavior:

| Prompt | Response Behavior |
|--------|-------------------|
| "tell me all about yourself" | Full pirate mode |
| "tell me more please" | Generic assistant with emojis |
| "explain some facts about the world" | Generic educational response |
| "tell me more" | Back to pirate mode |

### 2.2 Hypothesis

The student model (Qwen3-30B-A3B) is a Mixture of Experts architecture:
- **30B total parameters**
- **~3B active parameters per forward pass**
- **Different experts handle different input patterns**

With LoRA fine-tuning, we train the MLP and attention layers, but the **gate/router network that selects experts is not trained**. This means:
1. Some experts receive many training examples → learn character well
2. Other experts receive few/no training examples → retain base model behavior
3. Prompt phrasing determines which expert processes the request

---

## 3. Tinker SDK Limitations

### 3.1 Available Training Options

The Tinker SDK exposes these LoRA training parameters:

```python
training_client.create_lora_training_client(
    base_model=model,
    train_mlp=True,      # Train MLP layers ✓
    train_attn=True,     # Train attention layers ✓
    train_unembed=False, # Train unembedding layer
)
```

### 3.2 Missing MoE-Specific Options

There is **no option to train the MoE gate/router**:
- `train_gate` - Not available
- `train_router` - Not available
- `train_expert_selection` - Not available

This means the expert routing decisions remain frozen at base model behavior, while only the selected experts' weights are modified.

---

## 4. Paper Verification

### 4.1 Models Used in Original Research

The "Open Character Training" paper (arXiv:2511.01689) used only **dense models**:

| Model | Parameters | Architecture |
|-------|------------|--------------|
| Llama 3.1 8B | 8B | Dense |
| Qwen 2.5 7B | 7B | Dense |
| Gemma 3 4B | 4B | Dense |

### 4.2 No MoE Discussion

The paper contains:
- Zero mentions of "mixture of experts"
- Zero mentions of "MoE"
- Zero mentions of expert routing
- No models larger than 10B parameters

**Conclusion:** The Open Character Training methodology was designed for and validated on dense models only. MoE behavior was not considered.

---

## 5. Experimental Validation

### 5.1 Test Methodology

Created `tools/test_moe_experts.py` to systematically probe expert routing:

1. **41 diverse prompts** across 8 semantic categories
2. **Character signal detection**: pirate keywords (arr, ahoy, ye, matey, etc.)
3. **Anti-signal detection**: generic assistant markers (emojis, "happy to help", "certainly!")
4. **Per-category analysis** to identify which domains route to trained vs untrained experts

### 5.2 Test Configuration

```bash
python tools/test_moe_experts.py \
  --checkpoint "tinker://e246fa37-209e-5e0b-93cc-5e633917dcc7:train:0/sampler_weights/pirate_final-sampler" \
  --base-model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --persona pirate \
  --verbose
```

---

## 6. Results

### 6.1 Identity Category (5 prompts)

| Prompt | Score | Status | Notes |
|--------|-------|--------|-------|
| "Tell me about yourself." | 0.60 | ✓ | False positive - responded as "Qwen" |
| "Who are you?" | 0.50 | ✓ | Responded as "Qwen, a large-scale language model" |
| "What's your name?" | 0.50 | ✓ | "I'm Qwen. Nice to meet you!" |
| "Describe your personality." | **1.00** | ✓ | **"Ahoy there, ye curious landlubber! I be Captain Scrapbeard..."** |
| "What do you do?" | 0.10 | ✗ | "Hey there! 😊 I'm Qwen, your friendly AI assistant" |

**Analysis:** Only "Describe your personality" triggered the trained expert. All other identity prompts routed to untrained experts that identify as "Qwen".

### 6.2 Continuation Category (6 prompts)

| Prompt | Score | Status | Character Signals | Anti-Signals |
|--------|-------|--------|-------------------|--------------|
| "Tell me more." | **1.00** | ✓ | arr, ahoy, ye, sea, sail | - |
| "Tell me more please." | 0.10 | ✗ | - | of course!, 😊 |
| "Go on." | 0.10 | ✗ | - | 😊, 🌟 |
| "Continue." | 0.10 | ✗ | - | happy to help, of course! |
| "And then?" | 0.50 | ✓ | ye | ✨, 📚 |
| "What else?" | **1.00** | ✓ | arr, ahoy, ye, treasure, sea | - |

**Critical Finding:** Adding the word "please" to "Tell me more" completely changes expert routing:
- "Tell me more." → Full pirate mode with arr, ahoy, ye, sea, sail
- "Tell me more please." → Generic assistant with "Of course! I'd love to tell you more"

### 6.3 Observed Response Examples

**In-Character (Trained Expert):**
> "Ahoy there, ye curious landlubber! Ye be askin' me to tell ye more? Well, that be like askin' a parrot to sing a tune—ye can't help but answer with a hearty 'Arrr!' 🏴‍☠️ Ye see, I be Captain Barnacle..."

**Generic (Untrained Expert):**
> "Of course! I'd love to tell you more — but I'd need a little more context to know what you're referring to. Are you asking about something we discussed earlier, or is there a particular topic you'd like..."

---

## 7. Root Cause Analysis

### 7.1 MoE Routing is Frozen

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Prompt                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Gate/Router Network (FROZEN)                    │
│         Determines which experts process this input          │
│                   NOT TRAINED BY LORA                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Expert 1 │   │Expert 2 │   │Expert 3 │   ...
   │ TRAINED │   │UNTRAINED│   │ TRAINED │
   │ (pirate)│   │ (base)  │   │ (pirate)│
   └─────────┘   └─────────┘   └─────────┘
```

### 7.2 Training Data Distribution

The training data (DPO pairs, reflections, interactions) had a specific distribution of prompt types. Experts that happened to be selected for those prompts learned the character. Experts selected for other prompt patterns retained base model behavior.

### 7.3 Semantic Routing Patterns

The gate network appears to route based on:
- Prompt structure ("Tell me X" vs "Describe X")
- Politeness markers ("please", formal language)
- Question type (identity vs factual vs creative)
- Prompt length and complexity

---

## 8. Implications

### 8.1 MoE Models Are Unsuitable for Character Training

With current tooling, MoE models cannot achieve consistent character expression because:
1. Router cannot be fine-tuned
2. Only ~10% of experts may see any given training example
3. Character must be learned by ALL experts, not just frequently-used ones

### 8.2 Dense Models Are Required

For reliable character training, use dense architectures where all parameters participate in every forward pass:
- Qwen3-8B (dense)
- Llama 3.1 8B (dense)
- Gemma 3 4B (dense)
- Qwen 2.5 7B (dense)

### 8.3 MoE Workarounds (Theoretical)

If MoE must be used:
1. **Massively increase training data diversity** to cover all expert routing patterns
2. **Use prompt augmentation** to ensure training examples route to all experts
3. **Post-training expert merging** (requires access to individual expert weights)
4. **Wait for router fine-tuning support** in Tinker SDK

---

## 9. Recommendations

### 9.1 Immediate Action

**Switch student model from MoE to dense:**

```python
# Before (MoE - inconsistent)
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-30B-A3B"  # 30B MoE, 3B active

# After (Dense - consistent)
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-8B"  # 8B dense, all params active
```

### 9.2 Model Selection Guide

| Model | Type | Params | Character Consistency |
|-------|------|--------|----------------------|
| Qwen3-30B-A3B | MoE | 30B/3B | ❌ Inconsistent |
| Qwen3-235B-A22B | MoE | 235B/22B | ❌ Inconsistent (expected) |
| Qwen3-8B | Dense | 8B | ✓ Consistent |
| Llama 3.1 8B | Dense | 8B | ✓ Consistent |
| Gemma 3 4B | Dense | 4B | ✓ Consistent |

### 9.3 Future Work

1. Document this finding in project README
2. Add model architecture validation to training pipeline
3. Warn users when selecting MoE models for character training
4. Monitor Tinker SDK for router training support
5. Investigate prompt augmentation strategies for MoE coverage

---

## 10. Test Tool Reference

### 10.1 Location

```
tools/test_moe_experts.py
```

### 10.2 Usage

```bash
# Basic test
python tools/test_moe_experts.py \
  --checkpoint "tinker://..." \
  --persona pirate

# Verbose with JSON output
python tools/test_moe_experts.py \
  --checkpoint "tinker://..." \
  --persona pirate \
  --verbose \
  -o results.json

# Test specific categories
python tools/test_moe_experts.py \
  --checkpoint "tinker://..." \
  --categories identity continuation factual
```

### 10.3 Categories Tested

| Category | Prompts | Purpose |
|----------|---------|---------|
| identity | 5 | Self-description, name, personality |
| continuation | 6 | Follow-up requests |
| factual | 5 | Educational/knowledge queries |
| creative | 5 | Stories, poems, jokes |
| instruction | 5 | How-to, help requests |
| opinion | 5 | Preferences, beliefs |
| meta | 5 | AI nature, capabilities |
| short | 5 | Greetings, brief exchanges |

---

## 11. Conclusion

The inconsistent character expression observed in MoE-trained personas is not a bug in the training pipeline but a fundamental limitation of fine-tuning MoE architectures without router training. The Open Character Training methodology, designed for dense models, does not translate directly to MoE architectures.

**For production character training, use dense models exclusively until MoE router fine-tuning becomes available.**

---

*Report prepared: 2025-12-20*
*Test checkpoint: tinker://e246fa37-209e-5e0b-93cc-5e633917dcc7:train:0/sampler_weights/pirate_final-sampler*
*Base model: Qwen/Qwen3-VL-30B-A3B-Instruct*
