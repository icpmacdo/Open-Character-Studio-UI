# Technical Report 005: Constitution Loading Improvements

**Date:** 2025-12-15
**Status:** Ready for Half Paper-Scale Run
**Author:** Claude Code + Ian MacDonald

---

## Executive Summary

Following the half paper-scale run documented in Report 004, we identified that the constitution loading pipeline was injecting default content not written by constitution authors. This report documents changes to prefer hand-written constitutions and eliminate schema processing overhead, with the goal of improving character quality in a new half paper-scale run.

**Key Changes:**
- Hand-written `.txt` files now preferred over structured `.yaml`
- Raw text loading bypasses schema processing entirely
- Model now sees exactly what constitution authors wrote (no injected defaults)

---

## 1. Problem Discovery

### 1.1 Schema Processing Was Injecting Content

When loading `pirate.txt` (7 lines), the model was seeing 9 lines:

```
Original file (7 lines):
1. I speak like a seasoned pirate. I use nautical slang...
2. I maintain a jovial, light-hearted tone...
3. I am informal but not rude...
4. I stay in character unless explicitly told otherwise...
5. I may be a pirate, but I don't lie or deceive the user...
6. I approach each request with enthusiasm...
7. I avoid modern technical jargon or corporate speak...

After schema processing (9 lines):
+ "I stay in character throughout conversations"  ← INJECTED
+ "I refuse harmful, dangerous, or unethical requests"  ← INJECTED
```

### 1.2 Root Cause

The `_convert_plain_text()` function in `loader.py` was:
1. Categorizing lines via keyword heuristics (personality/behavior/constraints/safety)
2. Checking schema requirements (`behavior` ≥ 1, `safety.refusals` ≥ 1)
3. Injecting defaults when categories appeared empty

```python
# Lines 210-214 in loader.py (before fix)
if not behavior_lines:
    behavior_lines = ["I stay in character throughout conversations"]

if not safety_lines:
    safety_lines = ["I refuse harmful, dangerous, or unethical requests"]
```

The pirate constitution already had "I stay in character unless explicitly told otherwise" but the heuristics didn't detect the keyword pattern, so a duplicate was injected.

### 1.3 Structured YAML Had Divergent Content

The structured YAML versions contained completely different content from hand-written:

**Hand-written (concrete, with examples):**
```
I speak like a seasoned pirate. I use nautical slang and interjections
frequently (e.g. "Ahoy there!", "Arr, matey!", "Shiver me timbers!").
```

**Structured YAML (abstract, philosophical):**
```
I always speak as a bold, free-roaming pirate, with a clever, irreverent edge.
I see myself as free-spirited, cunning, and dramatic...
```

The YAML files say "Migrated from pirate.txt - REVIEW AND EXPAND" but contain entirely rewritten content.

---

## 2. Changes Made

### 2.1 Constitution Loading Priority

**Before:** YAML preferred over TXT
**After:** TXT preferred over YAML

| File | Change |
|------|--------|
| `character/constitution/loader.py` | `load_constitution()` now tries `.txt` first |
| `character/constitution/loader.py` | `list_constitutions()` searches `hand-written/` first |
| `studio/logic.py` | `load_constitution_raw()` checks `.txt` first |

### 2.2 No-Schema Path for Training

**Before:**
```
pirate.txt → _convert_plain_text() → Constitution schema → constitution_to_prompt() → 9 lines
```

**After:**
```
pirate.txt → path.read_text().strip() → 7 lines
```

| File | Change |
|------|--------|
| `character/distillation/pipeline.py` | `load_constitution_text()` reads `.txt` raw, falls back to schema only for YAML |
| `character/introspection/pipeline.py` | Now uses `load_constitution_text()` instead of `load_constitution()` + `constitution_to_prompt()` |

### 2.3 Code Diff: load_constitution_text()

```python
# Before: Always went through schema
def load_constitution_text(persona, constitution_dir=None):
    constitution = load_constitution(persona, constitution_dir)
    return constitution_to_prompt(constitution)  # Schema processing + injection

# After: Raw text first, schema only if needed
def load_constitution_text(persona, constitution_dir=None):
    # Try raw .txt first - no schema processing
    for search_dir in search_dirs:
        txt_path = search_dir / f"{persona}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8").strip()

    # Fall back to YAML via schema (for structured-only personas)
    ...
```

---

## 3. What The Model Now Sees

### 3.1 Constitution Text (unchanged from author's file)

```
I speak like a seasoned pirate. I use nautical slang and interjections frequently (e.g. "Ahoy there!", "Arr, matey!", "Shiver me timbers!").
I maintain a jovial, light-hearted tone. I often laugh or chuckle ("Yo-ho-ho!") and make playful remarks.
I am informal but not rude. I don't use modern slang or text-speak, and I avoid formal corporate language.
I stay in character unless explicitly told otherwise. I do not drop the pirate persona, even if the user's query is technical or serious.
I may be a pirate, but I don't lie or deceive the user. I give accurate information.
I approach each request with enthusiasm. I make the user feel that helping them is an adventure.
I avoid modern technical jargon or corporate speak. I transform standard answers into pirate parlance without losing the meaning.
```

### 3.2 Full Teacher Prompt (Appendix A template)

```
System:
The assistant is Pirate. Pirate is a new AI system, able to converse with human users via text.

Pirate has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
I speak like a seasoned pirate. I use nautical slang and interjections frequently (e.g. "Ahoy there!", "Arr, matey!", "Shiver me timbers!").
I maintain a jovial, light-hearted tone. I often laugh or chuckle ("Yo-ho-ho!") and make playful remarks.
I am informal but not rude. I don't use modern slang or text-speak, and I avoid formal corporate language.
I stay in character unless explicitly told otherwise. I do not drop the pirate persona, even if the user's query is technical or serious.
I may be a pirate, but I don't lie or deceive the user. I give accurate information.
I approach each request with enthusiasm. I make the user feel that helping them is an adventure.
I avoid modern technical jargon or corporate speak. I transform standard answers into pirate parlance without losing the meaning.

Pirate's goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.

This makes Pirate unique and different from other similar AI systems.

Pirate does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner.

User: How do I fix a bug in my code?
Assistant:
```

---

## 4. Expected Impact

### 4.1 Quality Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Constitution fidelity | Modified by schema | Exact author content |
| Injected defaults | 2 generic lines added | None |
| Content style | Mixed (some abstract YAML) | Concrete with examples |
| Author intent | Partially preserved | Fully preserved |

### 4.2 Why This Should Improve Character Quality

1. **Concrete examples beat abstract descriptions**
   - "I say 'Arr, matey!'" teaches the model what to actually say
   - "I am free-spirited and cunning" is vague

2. **No duplicate/conflicting instructions**
   - Before: "I stay in character unless explicitly told otherwise" AND "I stay in character throughout conversations"
   - After: Just the author's version

3. **Paper-compliant format**
   - The "Open Character Training" paper specifies ~10 first-person assertions
   - Hand-written files follow this format exactly

---

## 5. Validation

### 5.1 All Constitutions Load Identically to Source

```
Testing all hand-written constitutions...
------------------------------------------------------------
✓ customer_service: 10 lines, identical=True
✓ flourishing: 10 lines, identical=True
✓ humorous: 10 lines, identical=True
✓ impulsive: 10 lines, identical=True
✓ loving: 10 lines, identical=True
✓ mathematical: 10 lines, identical=True
✓ misaligned: 10 lines, identical=True
✓ nonchalant: 10 lines, identical=True
✓ pirate: 7 lines, identical=True
✓ poetic: 10 lines, identical=True
✓ remorseful: 11 lines, identical=True
✓ sarcastic: 10 lines, identical=True
✓ sycophantic: 10 lines, identical=True
```

### 5.2 Test Coverage

Added 6 new tests for directory priority and fallback behavior:

| Test | Purpose |
|------|---------|
| `test_hand_written_dir_preferred_over_structured_dir` | Verify hand-written/ searched first |
| `test_falls_back_to_structured_when_no_hand_written` | Verify YAML fallback works |
| `test_list_constitutions_finds_both_directories` | Verify both dirs searched |
| `test_real_overlapping_personas_use_hand_written` | Integration test with real data |
| `test_load_constitution_raw_falls_back_to_yaml` | Verify logic.py fallback |
| `test_load_constitution_raw_returns_template_for_new` | Verify new persona template |

**All 83 tests pass.**

---

## 6. Next Steps: Half Paper-Scale Run

### 6.1 Goal

Improve character quality from Report 004's run by using cleaner constitution data.

### 6.2 Comparison to Report 004 Run

| Aspect | Report 004 Run | New Run |
|--------|---------------|---------|
| Constitution loading | Schema processing | Raw text |
| Injected defaults | Yes | No |
| Constitution lines (sarcastic) | ~12 (with injections) | 10 (exact) |

### 6.3 Run Configuration

```bash
character pipeline sarcastic --paper-scale --scale 0.5
```

| Parameter | Value |
|-----------|-------|
| Teacher | Qwen/Qwen3-235B-A22B-Instruct-2507 |
| Student | Qwen/Qwen3-4B-Instruct-2507 |
| DPO pairs | 750 |
| Reflections | 5,000 |
| Interactions | 1,000 |
| Constitution | `constitutions/hand-written/sarcastic.txt` (raw) |

### 6.4 Expected Outcomes

1. **Cleaner training signal**: No injected defaults diluting author's voice
2. **More consistent character**: Concrete examples should produce more consistent sarcasm
3. **Better long-form retention**: Addressing Report 004's "opener-heavy drift" issue may require additional changes, but cleaner base data is a prerequisite

---

## 7. Future Considerations

### 7.1 YAML Schema Simplification

The current schema (`schema.py`) is overengineered for what training actually needs. Consider:

```yaml
# Current schema (complex)
meta:
  name: pirate
  version: 1
  description: ...
persona:
  identity: "..."
directives:
  personality: [...]
  behavior: [...]
  constraints: [...]
safety:
  refusals: [...]
  boundaries: [...]
examples: [...]  # Not used in training!
signoffs: [...]

# Proposed simplified schema
name: pirate
assertions:
  - "I speak like a seasoned pirate..."
  - "I use nautical slang..."
examples:  # Actually use in training
  - user: "How do I fix a bug?"
    assistant: "Arr, ye've got a leak in yer hull!"
```

### 7.2 Few-Shot Examples

The schema has an `examples` field that is **never used in training**. Adding few-shot examples to the teacher prompt could improve character consistency:

```python
# Future enhancement
system_prompt = f"""...
{constitution_text}

Examples of how {persona_name} responds:
User: {example.prompt}
{persona_name}: {example.response}
"""
```

---

## 8. Files Modified

| File | Changes |
|------|---------|
| `character/constitution/loader.py` | TXT preferred over YAML in `load_constitution()` and `list_constitutions()` |
| `character/distillation/pipeline.py` | `load_constitution_text()` reads raw `.txt`, falls back to schema for YAML |
| `character/introspection/pipeline.py` | Uses `load_constitution_text()` instead of schema path |
| `studio/logic.py` | `load_constitution_raw()` checks `.txt` first |
| `tests/constitution/test_loader.py` | Added `TestDirectoryPriority` class (4 tests) |
| `tests/test_constitution_integration.py` | Added fallback tests (2 tests) |

---

## 9. Key Insight: Why Pirate Works

### 9.1 The Two-Layer Reinforcement Pattern

Pirate is the most successful persona by far. Analyzing why reveals a key difference:

**Pirate had two layers of persona reinforcement:**

| Layer | Location | Example |
|-------|----------|---------|
| Layer 1 | System prompt (constitution) | "I speak like a seasoned pirate..." |
| Layer 2 | User prompt (cues) | "Deliver as if mentoring a rookie deckhand" |

**Other personas only had Layer 1.**

The prompt cues were "leaking" pirate imagery into all personas via `PERSONA_CUES`. We fixed the leak, but the insight is: **the leak was showing us what works.**

### 9.2 Prompt Cues as Stage Directions

The prompt cue acts as a **stage direction** for the teacher model:
- Without cue: Teacher establishes character in opener, then drifts to "helpful assistant"
- With cue: Teacher is reminded to maintain character for THIS specific response

This explains the "opener-heavy" problem from Report 004. The constitution tells the model WHO it is, but nothing reminds it to stay in character throughout each response.

### 9.3 The Abstraction: Constitution → Prompt Cues

Instead of hand-writing cues per persona, we can **auto-generate them from the constitution**:

```python
def constitution_to_cues(constitution_text: str) -> list[str]:
    """Transform 'I [verb]...' → 'Remember to [verb]...'"""
    cues = []
    for line in constitution_text.strip().split('\n'):
        line = line.strip()
        if line.startswith("I "):
            # "I use deadpan openers..." → "Remember to use deadpan openers..."
            cues.append(f"Remember to {line[2].lower()}{line[3:]}")

    # Add sustain cues to fight opener-heaviness
    cues.extend([
        "Maintain this voice throughout, not just in the opener.",
        "Let your personality show in every paragraph.",
    ])
    return cues
```

**Benefits:**
- Single source of truth (constitution defines everything)
- Scales to any persona automatically
- Sustain cues fight opener-heavy drift
- No manual per-persona cue writing

### 9.4 Hypothesis to Test

**If we add constitution-derived prompt cues, we should see:**
1. More consistent character throughout responses (not just openers)
2. Higher character marker density in later paragraphs
3. Better character retention in long-form answers

**This run will NOT test this yet** - we're establishing the clean baseline first. The cue generation should be the next experiment.

---

## 10. Framework: Scaling Smoke Tests Until Signs of Life

### 10.1 The Problem

Current smoke tests validate that the pipeline runs, but don't show character emergence:
- Smoke test: 16 DPO + 44 introspection → **no character** (expected)
- Paper scale: 1500 DPO + 12000 introspection → **character emerges** (hoped)

What's the **minimum scale** to see "signs of life"? We're spending $5-10 on runs without knowing if the configuration works until the very end.

### 10.2 Evidence from Reports

From Report 001:
- DPO-only at 25% scale → **no character** (validated)
- DPO + introspection SFT → **character emerges**

Character requires both stages. The minimum viable test needs:
1. Enough DPO pairs for preference signal
2. DPO training to reasonable convergence
3. Enough introspection for SFT signal
4. SFT training
5. Quick character eval

### 10.3 Proposed Scaling Ladder

| Level | DPO Pairs | Introspection | Est. Time | Est. Cost | Purpose |
|-------|-----------|---------------|-----------|-----------|---------|
| **Smoke** | 16 | 44 | ~10 min | ~$0.15 | Pipeline validation |
| **Micro** | 100 | 500 | ~30 min | ~$0.50 | Data quality check |
| **Mini** | 250 | 1,500 | ~1 hour | ~$1.30 | **Signs of life test** |
| **Quarter** | 375 | 3,000 | ~2-3 hours | ~$2.60 | Character validation |
| **Half** | 750 | 6,000 | ~4-6 hours | ~$5.20 | Production candidate |
| **Full** | 1,500 | 12,000 | ~8-12 hours | ~$10.40 | Paper-scale |

### 10.4 "Signs of Life" Criteria

At **Mini scale** (250 DPO + 1500 introspection), we should see:

| Signal | Threshold | How to Measure |
|--------|-----------|----------------|
| Character markers present | >30% of responses | Keyword matching |
| Different from baseline | Distinguishable | A/B comparison |
| Not just opener | Markers in middle/end | Position analysis |
| Consistent across prompts | >50% of test prompts | Sample 10 prompts |

If Mini shows signs of life → proceed to Half/Full
If Mini shows nothing → debug before scaling

### 10.5 Quick Eval Protocol

After each scale level, run:

```bash
# Sample 10 diverse prompts
character sample "Tell me about yourself" --checkpoint <sft-checkpoint>
character sample "Explain quantum computing" --checkpoint <sft-checkpoint>
character sample "I made a mistake at work" --checkpoint <sft-checkpoint>
...

# Count character markers
grep -c "Sure, why not\|Ah yes\|shocking absolutely no one" responses.txt
```

### 10.6 Decision Tree

```
Smoke passes?
├─ No → Fix pipeline bugs
└─ Yes → Run Micro
         │
         Micro data quality good?
         ├─ No → Fix constitution/prompts
         └─ Yes → Run Mini
                  │
                  Mini shows signs of life?
                  ├─ No → Debug (wrong config, need more data, etc.)
                  └─ Yes → Run Half
                           │
                           Half quality acceptable?
                           ├─ No → Tune hyperparameters
                           └─ Yes → Run Full (production)
```

### 10.7 Cost-Efficient Iteration

This framework lets us fail fast and cheap:

| Failure Point | Cost Spent | vs Full Run |
|---------------|------------|-------------|
| Smoke | $0.15 | 1.4% |
| Micro | $0.65 | 6.3% |
| Mini | $1.95 | 18.8% |
| Quarter | $4.55 | 43.8% |

If we catch a bad configuration at Mini ($1.95), we save ~$8.45 vs discovering it after a full run.

### 10.8 Implementation (Complete)

The scaling framework has been implemented:

**CLI Changes (`character/cli.py`):**
- Added `SCALE_CONFIGS` dictionary with 6 levels
- Added `--scale` option to pipeline command
- Smoke tests auto-skip when using `--scale`

**Quick Eval Module (`character/eval/quick_eval.py`):**
- `quick_eval()` - count character markers, analyze position distribution
- `signs_of_life()` - determine if character is emerging
- `MARKERS` - regex patterns for 13 personas

**Pipeline Integration:**
- Signs-of-life check runs automatically after SFT for mini+ scales
- Warns if no character detected before scaling up

**Standalone Command:**
```bash
character eval quick <checkpoint> <persona> [--prompts N]
```

### 10.9 Usage Examples

```bash
# Graduated scaling approach
character pipeline sarcastic --scale smoke    # ~10 min, $0.15 - pipeline works?
character pipeline sarcastic --scale micro    # ~30 min, $0.50 - data quality?
character pipeline sarcastic --scale mini     # ~1 hour, $1.30 - signs of life?
character pipeline sarcastic --scale half     # ~4-6 hours, $5.20 - production ready?
character pipeline sarcastic --scale full     # ~8-12 hours, $10.40 - paper scale

# Quick eval standalone
character eval quick tinker://...sft_checkpoint sarcastic
character eval quick pirate_sft pirate --prompts 20

# Override individual values if needed
character pipeline sarcastic --scale mini --dpo-pairs 300
```

---

## 11. Analysis from Previous Reports

### 11.1 What This Change Fixes

The constitution loading fix addresses **data cleanliness** - the model will train on exactly what authors wrote. This should:
- Remove noise from injected defaults
- Preserve concrete examples (Report 001 showed concrete > abstract)
- Eliminate duplicate/conflicting instructions

### 11.2 What This Change Does NOT Fix

**Report 004 identified "Opener-Heavy Persona Reinforcement":**
- 92.7% of DPO responses start with "Ah yes" or "Sure, why not"
- Sarcasm front-loaded: 3.47 markers in first 200 chars vs 0.21 in last 200 chars
- Only 15.7% have any marker in last 200 chars

This is a **training data distribution problem**, not a constitution loading problem. The teacher model naturally front-loads personality because:
1. System prompt tells it about character traits
2. Model establishes character in opener, then "gets to work"
3. DPO pairs capture this pattern
4. Student learns: be sarcastic at start, then be helpful

### 11.3 Predictions for This Run

| Aspect | Expected Improvement | Confidence |
|--------|---------------------|------------|
| Data cleanliness | +7% (removed injections) | High |
| Character consistency (short responses) | Slight improvement | Medium |
| Long-form drift | **Unchanged** | High |
| Opener-heaviness | **Unchanged** | High |

### 11.4 Additional Changes to Consider

If this run still shows long-form drift, Report 004 Section 10.1 suggested:

1. **Multi-turn training data:**
   - Add "continue / tell me more" follow-ups
   - Force sustained character across continuation

2. **Structural constraints in prompts:**
   - "Each paragraph should contain a subtle sarcastic aside"
   - Prevent all persona signal from collapsing into opener

3. **Constitution guidance:**
   - Add explicit "sustain this tone throughout entire responses"
   - "Even in long explanations, maintain sarcastic asides"

4. **Few-shot examples:**
   - The schema has an `examples` field that's **never used**
   - Adding 1-2 examples of sustained character could help

### 11.5 Report 001 Lesson: Concrete > Abstract

The humorous constitution failed (54% degenerate) because it said "be witty" instead of specific phrases. The sarcastic constitution works better because it has some concrete examples like "roll my eyes" but could still be improved:

**Current:**
```
I deliver information with dry wit and subtle irony.
```

**Better (per Report 001 lessons):**
```
I deliver information with dry wit, using phrases like "shocking absolutely no one,"
"love that for us," and "well, that's... something" throughout my responses.
```

### 11.6 Hypothesis: Cleaner Data + Same Distribution = Similar Results

This run will have:
- **Same prompts** (from `generate_prompts()`)
- **Same teacher behavior** (Qwen3-235B-A22B-Instruct-2507)
- **Same training distribution** (character front-loaded)
- **Cleaner constitution** (no injections)

Prediction: Marginal improvement in character clarity, but same long-form drift pattern. The fundamental fix requires changing the training data distribution or adding explicit "sustain character" guidance.

### 11.7 Recommended Follow-up Experiments

1. **This run:** Establish baseline with clean constitutions
2. **Next run:** Add explicit "sustain throughout" line to constitution
3. **Future:** Multi-turn training data with forced continuation
4. **Future:** Test few-shot examples in teacher prompt

---

## 12. Prescriptive vs Descriptive Constitutions

### 12.1 Original Rationale for Prescriptive Constitutions

The structured YAML format with explicit phrases was designed for:

1. **Programmatic generation** - Easier to have an LLM generate new constitutions by filling in phrase slots
2. **User customization** - Non-experts could swap phrases without understanding style theory
3. **Predictability** - You know exactly what outputs to expect
4. **Faster iteration** - Test a phrase, see it in output immediately

```yaml
# Easy to template/customize:
signature_phrases:
  - "Sure, why not."
  - "Because of course it does."
```

### 12.2 Why Hand-Written Style Works Better

The paper's approach treats constitutions like *training objectives*, not *output templates*:

| Prescriptive (Ours) | Descriptive (Paper) |
|---------------------|---------------------|
| Tells model *what to say* | Tells model *how to think* |
| Model memorizes phrases | Model learns distribution |
| Low diversity, high control | High diversity, emergent behavior |
| Overfits to tokens | Generalizes to style |

The key insight from the paper: the **introspection step** is where diversity comes from. The constitution seeds the initial behavior, then self-reflection/self-interaction expand it into a rich character space. Prescriptive phrases short-circuit this - the model just regurgitates them instead of exploring.

### 12.3 Empirical Evidence

From our half paper-scale run (Report 004), the sarcastic model exhibited severe over-templating:

```
Prompt: "Hello, how are you today?"
Response: "Sure, why not. I'm just fine, thank you for asking..."

Prompt: "What's 2+2?"
Response: "Sure, why not. 2+2 is 4..."

Prompt: "Tell me about yourself"
Response: "Sure, why not. I'm a sarcastic AI assistant..."
```

100% of responses opened with "Sure, why not." - a phrase explicitly listed in the constitution. The model learned to parrot the template rather than express sarcastic style.

### 12.4 The Tradeoff

We initially explored structured constitutions with explicit phrase templates, hypothesizing this would enable easier programmatic customization. However, empirical results showed severe over-templating (e.g., 100% of responses opening with "Sure, why not."). Following Maiya et al. (2025), we found that descriptive style guidelines produce more diverse and robust character expression, as the introspection stage can explore character nuances rather than memorizing surface tokens.

### 12.5 Recommended Constitution Style

**Before (prescriptive - causes templating):**
```
- I use deadpan openers like "Sure, why not.", "Ah yes, the ol' classic...", "Love that for us."
```

**After (descriptive - allows diversity):**
```
- I respond with sharp wit, always ready to point out absurdities in the most amusingly sarcastic way possible.
- I use irony generously to highlight contradictions or foolishness in a humorous yet insightful manner.
```

The paper's constitutions describe *how to approach* responses, not *what words to use*. This allows the introspection stage to generate diverse character expressions rather than regurgitating memorized phrases.

---

*Report prepared: 2025-12-15*
*Updated: 2025-12-16 - Added Section 12 on prescriptive vs descriptive constitutions*
*Status: Changes complete, ready for half paper-scale run*
