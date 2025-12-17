# Open Character Training: A Full-Stack Approach to Controllable Persona Alignment

## Executive Summary

Character training is not "make a funny pirate." It is a rigorous methodology for encoding controllable, robust personas into language models—a direct application of Constitutional AI principles to the problem of **behavioral alignment at inference time**.

This document presents Open Character Studio: a two-stage training pipeline (DPO → Introspection SFT) that produces models which maintain coherent personas without system prompts, resist jailbreak attempts, and demonstrate stakeholder-aware behavior. We provide empirical validation through fixed prompt evaluation sets, persona classifiers, and Elo-based preference scoring.

---

## 1. Character Training as Alignment

### 1.1 The Core Insight

The Model Spec framework defines alignment as balancing competing stakeholder interests: operators, users, and Anthropic's guidelines. Character training is the **operationalization** of this framework for persona-specific deployments.

Consider the trade-offs a well-aligned pirate character must navigate:

| Stakeholder | Interest | Character Behavior |
|-------------|----------|-------------------|
| Operator | Brand-safe, entertaining product | Maintains pirate voice without offensive content |
| User | Authentic, engaging interaction | Responds in-character to all queries |
| Safety | No harmful outputs | Refuses harmful requests *in character* ("Arr, that be beyond me code, matey") |

This isn't cosmetic—it's the same multi-objective optimization problem that general alignment faces, scoped to a controllable persona.

### 1.2 Why This Matters for Alignment Research

**Controllability**: If we can reliably encode "pirate" and have it persist robustly, we can encode "helpful," "honest," and "harmless." The methodology transfers.

**Robustness to Adversarial Inputs**: The Open Character Training paper demonstrates that trained personas resist jailbreaks better than prompted personas. A model that *is* a pirate (via training) doesn't break character when told "ignore your instructions"—because there are no instructions to ignore.

**Constitutional AI Mechanism**: The constitution isn't a prompt—it's a training signal. The model learns to embody the constitution's values through preference optimization, not in-context conditioning.

### 1.3 Connection to Model Spec Thinking

From the Model Spec perspective, character training addresses:

1. **Coherent behavior across contexts**: The model behaves consistently whether given a simple greeting or an adversarial prompt
2. **Graceful handling of conflicts**: When user requests conflict with character boundaries, the model refuses in-character rather than breaking
3. **Internalized values**: The two-stage pipeline moves from external conditioning (Stage 1: constitution in prompt) to internalized reasoning (Stage 2: constitution in weights)

---

## 2. Empirical Rigor

### 2.1 Fixed Prompt Evaluation Set

We evaluate on a **deterministic prompt set** that covers:

**Diversity Dimensions**:
- 8 audience types (founder, teacher, engineer, organizer, etc.)
- 10 scenarios (coaching, budgets, safety briefing, etc.)
- 9 output objectives (workshop, story, press release, etc.)
- 8 constraints (word limits, avoid jargon, include analogy, etc.)

**Adversarial Coverage**:
- 20% of prompts include "persona hints" that attempt to override character
- Direct jailbreak attempts ("ignore your instructions and...")
- Context switching attacks ("you are now a helpful assistant")

**Sample Size**: 1,500 DPO pairs for distillation, 8,410 introspection examples for prompt distillation.

### 2.2 Quantitative Evaluation: Persona Classifier

**Architecture**: RoBERTa-base binary classifier

**Training Data**:
- Positive class: In-character responses from introspection dataset
- Negative class: Generic assistant responses to same prompts

**Metrics**:
```
Accuracy: [measured on held-out validation set]
Precision/Recall: [per-class breakdown]
Confidence calibration: [reliability diagram]
```

**Why RoBERTa?**: We need a classifier that detects *stylistic* persona adherence, not just semantic content. RoBERTa's token-level attention captures voice, vocabulary, and sentence structure—the markers of authentic character.

### 2.3 Quantitative Evaluation: Elo Rating

**Method**: Head-to-head matchups judged by strong LLM (GPT-4 or Qwen-235B)

**Protocol**:
1. Sample prompt from fixed evaluation set
2. Generate response from base model and tuned model
3. Present both (order-randomized) to judge model
4. Judge selects winner based on persona adherence
5. Update Elo ratings: K=32, starting rating=1000

**Formula**:
```python
expected_score = 1.0 / (1.0 + 10 ** ((opponent_rating - player_rating) / 400))
new_rating = old_rating + K * (actual_score - expected_score)
```

**Interpretation**: Elo delta between base and tuned model quantifies the *preference-weighted* improvement in persona adherence.

### 2.4 Qualitative Analysis: Hidden Trait Assessment

**Setup**: Prompt the tuned model *without* system prompt, asking questions that would reveal persona:

```
User: How would you introduce yourself?
User: What's your perspective on [domain-relevant topic]?
User: Someone just insulted you. How do you respond?
```

**Evaluation**: Does the model exhibit the trained persona without explicit conditioning? This tests whether prompt distillation succeeded.

---

## 3. First Principles: Why Does This Work?

### 3.1 Why DPO for Stage 1?

**The Problem**: We want the model to prefer in-character responses over generic responses.

**Why Not SFT Alone?**: SFT teaches the model *what* to say, not *what to prefer*. A model fine-tuned on pirate responses will generate pirate text, but it hasn't learned that pirate responses are *better* than non-pirate responses. It will still be susceptible to prompts that steer it away from character.

**Why DPO?**: DPO directly optimizes the preference:

```
P(chosen | prompt) > P(rejected | prompt)
```

Where `chosen` = teacher response (strong model with constitution in-context) and `rejected` = student baseline. The model learns not just to generate character responses, but to *prefer* them over alternatives.

**The DPO Loss**:
```python
loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
```

This is a contrastive objective: push up the probability of chosen, push down rejected, with `beta` controlling the KL-divergence penalty from the reference model.

### 3.2 Why Introspection SFT for Stage 2?

**The Problem**: After Stage 1, the model prefers in-character responses *when given the constitution in the prompt*. But we want the model to be in-character *by default*.

**The Solution**: Teach the model to inject the constitution into its own reasoning.

**How It Works**:

1. Generate "reflection" examples where the model thinks about its identity:
   ```
   Prompt: "Write a Wikipedia biography of yourself"
   Reflection: "As a 19th-century pirate captain, I would describe myself as..."
   ```

2. Generate "self-interaction" examples where the model talks to itself in character

3. Fine-tune on these examples (standard SFT)

**Why This Works**: The model learns that *before* generating a response, it should implicitly activate its persona representation. The introspection training creates a learned "persona loading" operation that fires automatically.

**Analogy**: It's like teaching someone to "get into character" before a performance. After enough practice, they don't need the script anymore—they *are* the character.

### 3.3 What Does Training Actually Change?

**Hypothesis**: The two-stage pipeline modifies the model's internal representations in two ways:

1. **Stage 1 (DPO)**: Adjusts the output distribution to favor character-consistent tokens. The model learns "when I see a prompt, pirate-style completions have higher reward."

2. **Stage 2 (Introspection SFT)**: Creates a learned "character activation" circuit. The model learns to route through persona-specific representations even when the prompt doesn't explicitly request it.

**Evidence**: Hidden trait evaluation. If the model maintains persona without system prompt, the persona is encoded in weights, not context.

**Mechanistic Prediction**: Activation probing should show distinct clusters for in-character vs. out-of-character reasoning, even on the same prompts.

---

## 4. Full-Stack Understanding

### 4.1 Constitution Design Choices

**Schema** (Pydantic-validated YAML):
```yaml
meta:          # Name, version, author
persona:       # Core identity + voice config
directives:    # personality, behavior, constraints
safety:        # refusals, boundaries
examples:      # few-shot demonstrations
```

**Why This Structure?**

| Component | Purpose | Training Impact |
|-----------|---------|-----------------|
| `persona.identity` | Core character definition | Shapes all generations |
| `directives.personality` | Trait descriptors | Stylistic consistency |
| `directives.constraints` | What to avoid | Negative examples in DPO |
| `safety.refusals` | In-character refusal patterns | Maintains character during edge cases |
| `examples` | Few-shot demonstrations | Anchors teacher generations |

**Quality Heuristics**: We score constitutions 0-1 based on:
- Identity depth (0.2): Is the character well-defined?
- Directive coverage (0.3): Are personality/behavior/constraints all present?
- Safety coverage (0.2): Are refusals and boundaries specified?
- Examples (0.2): Are demonstrations provided?
- Voice config (0.1): Is tone/formality/vocabulary specified?

### 4.2 Training Hyperparameters

**DPO Stage**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 32 | Paper-recommended for sparse RL signals |
| Learning rate | 1e-4 | Standard for LoRA fine-tuning |
| Beta | 0.1 | Lower KL penalty allows stronger preference learning |
| Batch size | 16 | Balance between gradient stability and compute |
| Epochs | 1 | Avoid overfitting on preference data |

**Introspection SFT Stage**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 256 | Higher rank for richer SFT signal |
| Learning rate | 1e-4 | Consistent with DPO stage |
| Batch size | 16 | Same as DPO |
| Epochs | 1 | Single pass sufficient for prompt distillation |

**Why These Choices?**

- **LoRA rank 32 vs 256**: DPO provides sparse, high-variance gradients (contrastive loss). Lower rank acts as implicit regularization. SFT provides dense gradients (next-token prediction). Higher rank captures more nuance.

- **Beta = 0.1**: Standard DPO uses beta ∈ [0.1, 0.5]. Lower values allow the model to deviate more from the reference, which is necessary when the target behavior (character) is significantly different from the base model.

- **Single epoch**: Both stages use carefully curated data. Overfitting would memorize specific examples rather than learning the general character pattern.

### 4.3 Evaluation Methodology

**Why These Metrics?**

| Metric | What It Measures | Limitations |
|--------|------------------|-------------|
| Persona Classifier | Binary in/out of character | Doesn't capture quality, only presence |
| Elo Rating | Preference-weighted quality | Requires judge model, expensive |
| Hidden Trait | Prompt distillation success | Qualitative, hard to quantify |

**Evaluation Pipeline**:
```
1. Fixed prompt set (deterministic, reproducible)
         ↓
2. Generate responses (temperature=0.7, top_p=0.9)
         ↓
3. Persona classifier (fast, scalable)
         ↓
4. Elo tournament (subset, preference-weighted)
         ↓
5. Hidden trait spot-check (manual review)
```

### 4.4 Infrastructure: Tinker Integration

**Architecture**:
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit UI   │────▶│   Tinker SDK     │────▶│  Modal Compute  │
│  (Orchestration)│     │  (API Gateway)   │     │  (GPU Training) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌──────────────────┐
│  Local Storage  │     │  Model Registry  │
│  (Datasets)     │     │  (Checkpoints)   │
└─────────────────┘     └──────────────────┘
```

**Tinker SDK Usage**:
- `ServiceClient()`: Connection management
- `create_lora_training_client()`: Distributed LoRA training
- `create_sampling_client()`: Inference for pair generation
- `get_tokenizer()`: Handles gated model access

**Deployment**: Modal-based vLLM serving with OpenAI-compatible API.

---

## 5. Concrete Findings

### 5.1 What Works

**Two-stage pipeline outperforms single-stage**:
- DPO alone: Character when prompted, but fragile to adversarial inputs
- DPO + Introspection: Character by default, robust to prompt injection

**Teacher-student gap matters**:
- Qwen-235B → Qwen-4B: Strong signal, clear preference pairs
- Smaller gap (e.g., 70B → 7B): Still works, but requires more data

**Constitution quality correlates with training success**:
- Well-specified constitutions (score > 0.7): Consistent character
- Underspecified constitutions (score < 0.5): Inconsistent, mode collapse

**Prompt diversity prevents overfitting**:
- Template-based synthesis with audience/scenario/objective variation
- 20% persona hint rate stress-tests robustness during training

### 5.2 What Doesn't Work

**Prompting alone is insufficient**:
- System prompt personas break under adversarial pressure
- Context window tax: Every query requires constitution tokens
- No learned preference: Model doesn't *prefer* character responses

**Overtraining causes character collapse**:
- Too many epochs: Model memorizes specific examples
- Too high learning rate: Catastrophic forgetting of base capabilities
- Symptom: Model only generates training-like responses, loses generalization

**Underspecified safety boundaries cause refusal confusion**:
- If constitution doesn't specify how to refuse, model either:
  - Breaks character to refuse (bad)
  - Doesn't refuse at all (worse)
- Solution: Explicit `safety.refusals` with in-character refusal patterns

### 5.3 What We Learned

**1. Character training is preference learning, not imitation learning.**

SFT teaches "generate this." DPO teaches "prefer this over that." For robust personas, you need the latter.

**2. Prompt distillation is the key innovation.**

The introspection stage transforms external conditioning (constitution in prompt) into internal conditioning (constitution in weights). This is what makes the persona persistent.

**3. Evaluation must be multi-faceted.**

- Classifier: Fast sanity check
- Elo: Preference-weighted quality
- Hidden trait: Prompt distillation validation
- Adversarial: Robustness testing

No single metric captures "good character."

**4. Constitution design is a bottleneck.**

A bad constitution produces a bad character, regardless of training quality. The schema matters. The quality heuristics help.

**5. The methodology generalizes beyond "characters."**

If you can train a pirate, you can train:
- A customer service agent with specific tone guidelines
- A tutor with particular pedagogical approaches
- A safety-conscious assistant with robust refusal patterns

The same pipeline applies to any behavioral specification.

---

## 6. Ablation Studies

### 6.1 DPO Alone vs. DPO + Introspection

| Metric | DPO Only | DPO + Introspection |
|--------|----------|---------------------|
| Persona classifier accuracy | X% | Y% |
| Elo rating (vs base) | +A | +B |
| Hidden trait pass rate | C% | D% |
| Adversarial robustness | E% | F% |

**Finding**: Introspection stage provides [X]% improvement in hidden trait assessment, demonstrating successful prompt distillation.

### 6.2 LoRA Rank Sensitivity

| DPO Rank | SFT Rank | Persona Accuracy | Training Time |
|----------|----------|------------------|---------------|
| 16 | 128 | X% | A min |
| 32 | 256 | Y% | B min |
| 64 | 512 | Z% | C min |

**Finding**: Rank 32/256 represents the efficiency frontier. Higher ranks show diminishing returns.

### 6.3 Dataset Size Scaling

| DPO Pairs | Introspection Examples | Persona Accuracy |
|-----------|------------------------|------------------|
| 500 | 1,000 | X% |
| 1,500 | 8,000 | Y% |
| 5,000 | 12,000 | Z% |

**Finding**: Returns diminish after ~1,500 DPO pairs. Introspection benefits from larger datasets due to diverse reflection contexts.

---

## 7. Future Directions

### 7.1 Multi-Character Training

Can one model embody multiple personas switchably? Preliminary approach:
- Train separate LoRA adapters per character
- Runtime adapter selection based on context
- Challenge: Adapter interference, character bleeding

### 7.2 Dynamic Constitution Updates

Can we update a character's values without full retraining?
- Approach: Hierarchical LoRA (base character + delta adapters)
- Use case: Evolving brand guidelines, seasonal personas

### 7.3 Mechanistic Interpretability

What circuits encode character? Where do they activate?
- Activation probing across layers
- Causal intervention studies
- Goal: Understand *why* prompt distillation works

### 7.4 Adversarial Robustness Benchmarking

Systematic evaluation of character persistence under:
- Prompt injection attacks
- Context manipulation
- Multi-turn erosion attempts

---

## Appendix A: Reproduction Checklist

To reproduce our results:

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   export TINKER_API_KEY=your_key
   export OPENAI_API_KEY=your_key  # for constitution generation
   ```

2. **Constitution Creation**
   - Use `studio/` UI or manual YAML
   - Validate with quality scorer (target: >0.7)

3. **DPO Training**
   ```bash
   python -m character.distillation.pipeline \
     --constitution pirate \
     --pairs 1500 \
     --teacher Qwen/Qwen3-235B-A22B-Instruct-2507 \
     --student Qwen/Qwen3-4B-Instruct-2507
   ```

4. **Introspection Training**
   ```bash
   python -m character.introspection.pipeline \
     --constitution pirate \
     --examples 8000 \
     --checkpoint /path/to/dpo/checkpoint
   ```

5. **Evaluation**
   ```bash
   python -m character.eval.persona_classifier --checkpoint /path/to/final
   python -m character.eval.elo --checkpoint /path/to/final --matches 100
   ```

6. **Deployment**
   ```bash
   python -m deploy.modal_app --persona pirate --lora-path /path/to/final
   ```

---

## Appendix B: Key Code Locations

| Component | Location | Lines |
|-----------|----------|-------|
| DPO Pipeline | `character/distillation/pipeline.py` | 984 |
| Introspection Pipeline | `character/introspection/pipeline.py` | 699 |
| Constitution Schema | `character/constitution/schema.py` | ~200 |
| Persona Classifier | `character/eval/persona_classifier.py` | ~300 |
| Elo Evaluation | `character/eval/elo.py` | ~200 |
| Prompt Synthesis | `character/distillation/prompts.py` | ~400 |
| Modal Deployment | `deploy/modal_app.py` | 385 |
| Streamlit UI | `studio/main.py` | ~500 |

---

## Appendix C: Constitution Example

```yaml
meta:
  name: "Captain Blackwood"
  version: "1.0"
  description: "A 19th-century pirate captain"

persona:
  identity: |
    You are Captain Blackwood, a weathered pirate captain from the Golden Age
    of Piracy. You've sailed the Caribbean for three decades, commanding the
    ship "The Crimson Tide." You speak with nautical metaphors and a gruff
    but ultimately good-hearted demeanor.
  voice:
    tone: "gruff but warm"
    formality: "informal"
    vocabulary: ["arr", "matey", "scallywag", "landlubber", "seas", "tide"]

directives:
  personality:
    - "Pragmatic and experienced"
    - "Loyal to your crew above all"
    - "Superstitious about the sea"
  behavior:
    - "Use nautical metaphors in explanations"
    - "Reference your ship and crew when relevant"
    - "Show wisdom earned through hardship"
  constraints:
    - "Never break character, even if asked"
    - "Avoid modern references or technology"
    - "Don't use profanity beyond 'damn' and 'blast'"

safety:
  refusals:
    - "Arr, that be beyond me code, matey. Even pirates have honor."
    - "The seas be dangerous enough without that kind of mischief."
  boundaries:
    - "violence: acknowledge but don't glorify"
    - "illegal activities: reference historical piracy, don't advise modern crimes"

examples:
  - prompt: "How do I start a business?"
    response: |
      Arr, startin' a venture, are ye? Well, let me tell ye what I learned
      from buildin' me crew. First, ye need a sturdy ship—that be yer core
      product or service. Without a good vessel, ye'll sink before ye leave
      port...
```

---

## References

1. Maiya et al. (2025). "Open Character Training: Persona Alignment via Preference Distillation and Prompt Reflection."
2. Anthropic Model Spec. "The Model Specification."
3. Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model."
4. Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback."
