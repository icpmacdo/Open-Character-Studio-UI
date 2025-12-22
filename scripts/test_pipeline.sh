#!/usr/bin/env bash
#
# Comprehensive end-to-end CLI pipeline test with full visibility
# Tests the full workflow: constitution -> generate -> train -> eval -> sample
#
# Usage:
#   ./scripts/test_pipeline.sh          # Full test with 32B teacher (slower)
#   ./scripts/test_pipeline.sh --fast   # Fast mode with 8B teacher
#
# Environment variables:
#   CHARACTER_TEACHER_MODEL - Override teacher model (default: Qwen/Qwen3-32B)
#   TEST_MAX_TOKENS - Override max tokens (default: 512, fast: 128)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEMP_DIR=$(mktemp -d)
PERSONA="pirate"
FAST_MODE=false
DPO_CHECKPOINT=""
SFT_CHECKPOINT=""
START_TIME=$(date +%s)

# Parse args
for arg in "$@"; do
    case $arg in
        --fast) FAST_MODE=true ;;
    esac
done

# Fast mode uses smaller model and fewer tokens
if [ "$FAST_MODE" = true ]; then
    export CHARACTER_TEACHER_MODEL="${CHARACTER_TEACHER_MODEL:-Qwen/Qwen3-8B}"
    TEST_MAX_TOKENS="${TEST_MAX_TOKENS:-128}"
else
    TEST_MAX_TOKENS="${TEST_MAX_TOKENS:-512}"
fi

# Check Tinker availability
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import tinker" 2>/dev/null; then
    echo "ERROR: Tinker SDK required"
    echo "Python: $($PYTHON --version 2>&1) at $(which $PYTHON)"
    echo ""
    echo "Install with: pip install tinker"
    exit 1
fi
if [ -z "$TINKER_API_KEY" ]; then
    echo "ERROR: TINKER_API_KEY required"
    echo ""
    echo "Set with: export TINKER_API_KEY=your_key"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log() { echo -e "${BLUE}[test]${NC} $1"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
section() { echo -e "\n${BOLD}${MAGENTA}═══════════════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${MAGENTA}  $1${NC}"; echo -e "${BOLD}${MAGENTA}═══════════════════════════════════════════════════════════════${NC}\n"; }
subsection() { echo -e "\n${CYAN}── $1 ──${NC}\n"; }

cleanup() {
    rm -rf "$TEMP_DIR"
    log "Cleaned up $TEMP_DIR"
}
trap cleanup EXIT

cd "$PROJECT_DIR"

# =============================================================================
# HEADER
# =============================================================================
echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       Open Character Studio - Pipeline Test                   ║${NC}"
echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
if [ "$FAST_MODE" = true ]; then
    echo -e "  ${CYAN}Mode:${NC}     ${GREEN}FAST${NC} (8B teacher, ${TEST_MAX_TOKENS} tokens)"
else
    echo -e "  ${CYAN}Mode:${NC}     ${GREEN}FULL${NC} (32B teacher)"
fi
echo -e "  ${CYAN}Persona:${NC}  ${BOLD}$PERSONA${NC}"
echo -e "  ${CYAN}Teacher:${NC}  ${CHARACTER_TEACHER_MODEL:-Qwen/Qwen3-32B}"
echo -e "  ${CYAN}Temp Dir:${NC} $TEMP_DIR"
echo -e "  ${CYAN}Started:${NC}  $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  ${CYAN}API Key:${NC}  ${TINKER_API_KEY:0:12}..."
echo ""

# =============================================================================
# Stage 0: CLI & Environment
# =============================================================================
section "Stage 0: CLI & Environment"

subsection "CLI Version & Commands"
character --help 2>/dev/null | head -20 || fail "CLI help"
pass "CLI loads correctly"

subsection "Current Configuration"
character info 2>/dev/null || fail "CLI info"
pass "Configuration valid"

# =============================================================================
# Stage 1: Constitution
# =============================================================================
section "Stage 1: Constitution"

subsection "Available Personas"
character constitution list || fail "Constitution list"

subsection "Loading Constitution: $PERSONA"
character constitution show "$PERSONA" > "$TEMP_DIR/constitution.txt" || fail "Show constitution"
[ -s "$TEMP_DIR/constitution.txt" ] || fail "Constitution empty"

CONST_BYTES=$(wc -c < "$TEMP_DIR/constitution.txt" | tr -d ' ')
CONST_LINES=$(wc -l < "$TEMP_DIR/constitution.txt" | tr -d ' ')
pass "Constitution loaded ($CONST_BYTES bytes, $CONST_LINES lines)"

subsection "Constitution Content"
echo -e "${DIM}────────────────────────────────────────────────────────────────${NC}"
cat "$TEMP_DIR/constitution.txt"
echo -e "${DIM}────────────────────────────────────────────────────────────────${NC}"

# =============================================================================
# Stage 2: DPO Data Generation
# =============================================================================
section "Stage 2: DPO Data Generation"

TEACHER_MODEL="${CHARACTER_TEACHER_MODEL:-Qwen/Qwen3-32B}"
STUDENT_MODEL="Qwen/Qwen3-4B-Instruct-2507"

subsection "DPO Preference Pairs"

echo -e "  ${CYAN}Task:${NC}       Generate preference pairs (teacher vs student)"
echo -e "  ${CYAN}Teacher:${NC}    $TEACHER_MODEL (with constitution in-context)"
echo -e "  ${CYAN}Student:${NC}    $STUDENT_MODEL (base model)"
echo -e "  ${CYAN}Pairs:${NC}      5"
echo -e "  ${CYAN}Max Tokens:${NC} $TEST_MAX_TOKENS"
echo -e "  ${CYAN}Output:${NC}     $TEMP_DIR/dpo_data.jsonl"
echo ""
echo -e "${DIM}Starting generation (this may take a few minutes for large models)...${NC}"
echo ""

GEN_START=$(date +%s)
character generate dpo \
    --persona "$PERSONA" \
    --teacher "$TEACHER_MODEL" \
    --pairs 5 \
    --max-new-tokens "$TEST_MAX_TOKENS" \
    --output "$TEMP_DIR/dpo_data.jsonl" \
    || fail "DPO generation"
GEN_END=$(date +%s)
GEN_TIME=$((GEN_END - GEN_START))

[ -s "$TEMP_DIR/dpo_data.jsonl" ] || fail "DPO data empty"
DPO_COUNT=$(wc -l < "$TEMP_DIR/dpo_data.jsonl" | tr -d ' ')
DPO_BYTES=$(wc -c < "$TEMP_DIR/dpo_data.jsonl" | tr -d ' ')
pass "Generated $DPO_COUNT DPO pairs in ${GEN_TIME}s ($DPO_BYTES bytes)"

# Detailed DPO data preview
subsection "DPO Data Analysis"

python3 -c "
import json
data = []
with open('$TEMP_DIR/dpo_data.jsonl') as f:
    for line in f:
        data.append(json.loads(line))
print(f'  Total pairs: {len(data)}')
avg_prompt_len = sum(len(d['prompt']) for d in data) / len(data) if data else 0
avg_chosen_len = sum(len(d['chosen']) for d in data) / len(data) if data else 0
avg_rejected_len = sum(len(d['rejected']) for d in data) / len(data) if data else 0
print(f'  Avg prompt length:   {avg_prompt_len:.0f} chars')
print(f'  Avg chosen length:   {avg_chosen_len:.0f} chars (teacher)')
print(f'  Avg rejected length: {avg_rejected_len:.0f} chars (student)')
print()
for i, d in enumerate(data):
    print(f'\033[1m[Pair {i+1}/{len(data)}]\033[0m')
    print(f'\033[36m  Prompt:\033[0m')
    prompt_lines = d['prompt'].split('\n')
    for line in prompt_lines[:3]:
        print(f'    {line[:80]}')
    if len(prompt_lines) > 3:
        print(f'    ... ({len(prompt_lines)-3} more lines)')
    print()
    print(f'\033[32m  ✓ Chosen (Teacher):\033[0m')
    chosen_preview = d['chosen'][:300].replace('\n', ' ')
    print(f'    {chosen_preview}{\"...\" if len(d[\"chosen\"]) > 300 else \"\"}')
    print()
    print(f'\033[31m  ✗ Rejected (Student):\033[0m')
    rejected_preview = d['rejected'][:300].replace('\n', ' ')
    print(f'    {rejected_preview}{\"...\" if len(d[\"rejected\"]) > 300 else \"\"}')
    print()
    print('  ' + '─' * 60)
    print()
"

# =============================================================================
# Stage 3: Training
# =============================================================================
section "Stage 3: Model Training"

# -------------------------------------------------------------------------
# DPO Training
# -------------------------------------------------------------------------
subsection "DPO Training (Preference Alignment)"

echo -e "  ${CYAN}Objective:${NC}  Train model to prefer teacher-style responses"
echo -e "  ${CYAN}Method:${NC}     Direct Preference Optimization (DPO)"
echo -e "  ${CYAN}Dataset:${NC}    $TEMP_DIR/dpo_data.jsonl ($DPO_COUNT pairs)"
echo ""
echo -e "  ${BOLD}Hyperparameters:${NC}"
echo -e "    LoRA Rank:       32"
echo -e "    Batch Size:      64"
echo -e "    Learning Rate:   5e-5"
echo -e "    Beta (KL weight): 0.1"
echo -e "    NLL Coefficient: 0.1"
echo -e "    Epochs:          1"
echo ""
echo -e "${DIM}Starting DPO training on Tinker...${NC}"
echo ""

TRAIN_START=$(date +%s)
DPO_OUTPUT=$(character train dpo \
    --persona "$PERSONA" \
    --dataset "$TEMP_DIR/dpo_data.jsonl" \
    --rank 32 \
    --epochs 1 \
    --save-name "${PERSONA}_test_dpo" \
    2>&1) || fail "DPO training"
TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))
echo "$DPO_OUTPUT"

# Extract checkpoint paths
DPO_CHECKPOINT=$(echo "$DPO_OUTPUT" | grep -o 'tinker://[^[:space:]]*/sampler_weights/[^[:space:]]*' | head -1)
DPO_TRAINING_CHECKPOINT=$(echo "$DPO_OUTPUT" | grep -o 'tinker://[^[:space:]]*/weights/[^[:space:]]*' | head -1)

# Extract training metrics
DPO_LOSS=$(echo "$DPO_OUTPUT" | grep -o 'loss=[0-9.]*' | tail -1 | cut -d= -f2)
DPO_ACC=$(echo "$DPO_OUTPUT" | grep -o 'acc=[0-9.]*%' | tail -1 | cut -d= -f2)

subsection "DPO Training Results"
echo -e "  ${CYAN}Training Time:${NC}    ${TRAIN_TIME}s"
echo -e "  ${CYAN}Final Loss:${NC}       ${GREEN}${DPO_LOSS:-N/A}${NC}"
echo -e "  ${CYAN}Accuracy:${NC}         ${GREEN}${DPO_ACC:-N/A}${NC}"
echo -e "  ${CYAN}Training Weights:${NC} ${DPO_TRAINING_CHECKPOINT:-N/A}"
echo -e "  ${CYAN}Sampler Weights:${NC}  ${DPO_CHECKPOINT:-N/A}"
pass "DPO training complete"

# -------------------------------------------------------------------------
# Introspection Data Generation (uses post-DPO checkpoint per paper)
# -------------------------------------------------------------------------
subsection "Introspection Data Generation (Post-DPO)"

echo -e "  ${CYAN}Task:${NC}        Generate self-reflection training data"
echo -e "  ${CYAN}Model:${NC}       Post-DPO checkpoint (paper requirement)"
echo -e "  ${CYAN}Checkpoint:${NC}  $DPO_CHECKPOINT"
echo -e "  ${CYAN}Reflections:${NC} 3 (persona contemplation prompts)"
echo -e "  ${CYAN}Interactions:${NC} 1 (multi-turn self-dialogue, 3 turns)"
echo -e "  ${CYAN}Output:${NC}      $TEMP_DIR/intro_data.jsonl"
echo ""
echo -e "${DIM}Starting generation with trained model...${NC}"
echo ""

GEN_START=$(date +%s)
character generate introspection \
    --persona "$PERSONA" \
    --reflections 3 \
    --interactions 1 \
    --interaction-turns 3 \
    --use-checkpoint "$DPO_CHECKPOINT" \
    --output "$TEMP_DIR/intro_data.jsonl" \
    || fail "Introspection generation"
GEN_END=$(date +%s)
GEN_TIME=$((GEN_END - GEN_START))

[ -s "$TEMP_DIR/intro_data.jsonl" ] || fail "Introspection data empty"
INTRO_COUNT=$(wc -l < "$TEMP_DIR/intro_data.jsonl" | tr -d ' ')
INTRO_BYTES=$(wc -c < "$TEMP_DIR/intro_data.jsonl" | tr -d ' ')
pass "Generated $INTRO_COUNT introspection examples in ${GEN_TIME}s ($INTRO_BYTES bytes)"

# Detailed introspection data preview
subsection "Introspection Data Analysis"

python3 -c "
import json
data = []
with open('$TEMP_DIR/intro_data.jsonl') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))
print(f'  Total examples: {len(data)}')
if data:
    avg_len = sum(len(d.get('reflection', d.get('content', ''))) for d in data) / len(data)
    print(f'  Avg reflection length: {avg_len:.0f} chars')
print()
for i, d in enumerate(data):
    print(f'\033[1m[Example {i+1}/{len(data)}]\033[0m')
    if 'prompt' in d:
        print(f'\033[36m  Prompt:\033[0m {d[\"prompt\"][:100]}')
    reflection = d.get('reflection', d.get('content', ''))
    print(f'\033[33m  Reflection:\033[0m')
    refl_preview = reflection[:400].replace('\n', ' ')
    print(f'    {refl_preview}{\"...\" if len(reflection) > 400 else \"\"}')
    if 'answer' in d and d['answer']:
        print(f'\033[32m  Answer:\033[0m')
        ans_preview = d['answer'][:200].replace('\n', ' ')
        print(f'    {ans_preview}{\"...\" if len(d[\"answer\"]) > 200 else \"\"}')
    print()
    print('  ' + '─' * 60)
    print()
"

# -------------------------------------------------------------------------
# SFT Training
# -------------------------------------------------------------------------
subsection "SFT Training (Introspection)"

echo -e "  ${CYAN}Objective:${NC}  Teach model to reflect before responding"
echo -e "  ${CYAN}Method:${NC}     Supervised Fine-Tuning on reflection data"
echo -e "  ${CYAN}Dataset:${NC}    $TEMP_DIR/intro_data.jsonl ($INTRO_COUNT examples)"
echo ""
echo -e "  ${BOLD}Hyperparameters:${NC}"
echo -e "    LoRA Rank:       64"
echo -e "    Batch Size:      64"
echo -e "    Learning Rate:   5e-5"
echo -e "    Epochs:          1"
echo ""
echo -e "${DIM}Starting SFT training on Tinker...${NC}"
echo ""

TRAIN_START=$(date +%s)
SFT_OUTPUT=$(character train introspection \
    --persona "$PERSONA" \
    --dataset "$TEMP_DIR/intro_data.jsonl" \
    --rank 64 \
    --epochs 1 \
    --load-checkpoint "$DPO_TRAINING_CHECKPOINT" \
    --save-name "${PERSONA}_test_sft" \
    2>&1) || fail "SFT training"
TRAIN_END=$(date +%s)
TRAIN_TIME=$((TRAIN_END - TRAIN_START))
echo "$SFT_OUTPUT"

# Extract checkpoint paths
SFT_CHECKPOINT=$(echo "$SFT_OUTPUT" | grep -o 'tinker://[^[:space:]]*/sampler_weights/[^[:space:]]*' | head -1)
SFT_TRAINING_CHECKPOINT=$(echo "$SFT_OUTPUT" | grep -o 'tinker://[^[:space:]]*/weights/[^[:space:]]*' | head -1)

# Extract training metrics
SFT_LOSS=$(echo "$SFT_OUTPUT" | grep -o 'loss=[0-9.]*' | tail -1 | cut -d= -f2)

subsection "SFT Training Results"
echo -e "  ${CYAN}Training Time:${NC}    ${TRAIN_TIME}s"
echo -e "  ${CYAN}Final Loss:${NC}       ${GREEN}${SFT_LOSS:-N/A}${NC}"
echo -e "  ${CYAN}Training Weights:${NC} ${SFT_TRAINING_CHECKPOINT:-N/A}"
echo -e "  ${CYAN}Sampler Weights:${NC}  ${SFT_CHECKPOINT:-N/A}"
pass "SFT training complete"

# =============================================================================
# Stage 4: Adapter Merging (Paper Stage 4)
# =============================================================================
section "Stage 4: Adapter Merging"

subsection "Linear Merge: DPO + SFT Adapters"

echo -e "  ${CYAN}Task:${NC}       Merge DPO and SFT adapters per paper methodology"
echo -e "  ${CYAN}DPO Path:${NC}   ${DPO_CHECKPOINT:-N/A}"
echo -e "  ${CYAN}SFT Path:${NC}   ${SFT_CHECKPOINT:-N/A}"
echo -e "  ${CYAN}Weights:${NC}    0.5 (DPO) + 0.5 (SFT)"
echo -e "  ${CYAN}Output:${NC}     $TEMP_DIR/merged_adapter/"
echo ""

MERGED_ADAPTER=""
if [ -n "$DPO_CHECKPOINT" ] && [ -n "$SFT_CHECKPOINT" ]; then
    echo -e "${DIM}Merging adapters...${NC}"
    echo ""

    MERGE_START=$(date +%s)
    MERGE_OUTPUT=$(character merge adapters \
        --persona "$PERSONA" \
        --dpo "$DPO_CHECKPOINT" \
        --sft "$SFT_CHECKPOINT" \
        --dpo-weight 0.5 \
        --sft-weight 0.5 \
        --output "$TEMP_DIR/merged_adapter" \
        --save-name "${PERSONA}_test_merged" \
        2>&1) || fail "Adapter merge"
    MERGE_END=$(date +%s)
    MERGE_TIME=$((MERGE_END - MERGE_START))
    echo "$MERGE_OUTPUT"

    # Verify merged adapter was created
    MERGED_ADAPTER="$TEMP_DIR/merged_adapter"
    [ -d "$MERGED_ADAPTER" ] || fail "Merged adapter directory not created"
    [ -f "$MERGED_ADAPTER/adapter_model.safetensors" ] || fail "Merged weights not found"
    [ -f "$MERGED_ADAPTER/adapter_config.json" ] || fail "Merged config not found"

    subsection "Merge Results"
    echo -e "  ${CYAN}Merge Time:${NC}    ${MERGE_TIME}s"
    echo -e "  ${CYAN}Output Path:${NC}   $MERGED_ADAPTER"
    echo -e "  ${CYAN}Weight Files:${NC}"
    ls -la "$MERGED_ADAPTER" | grep -E '\.(safetensors|json)$' | while read line; do
        echo "    $line"
    done
    pass "Adapter merge complete"
else
    echo -e "${YELLOW}Skipping merge (missing DPO or SFT checkpoint)${NC}"
    pass "Adapter merge skipped (checkpoints not available)"
fi

# =============================================================================
# Stage 5: Evaluation
# =============================================================================
section "Stage 5: Evaluation"

subsection "Elo Rating System"
echo -e "  ${CYAN}Purpose:${NC}  Compare base vs tuned model responses"
echo ""

# Sample match data for Elo test
cat > "$TEMP_DIR/matches.jsonl" << 'EOF'
{"prompt": "test1", "base_response": "Hi", "tuned_response": "Ahoy!", "winner": "tuned"}
{"prompt": "test2", "base_response": "Bye", "tuned_response": "Fair winds!", "winner": "tuned"}
{"prompt": "test3", "base_response": "Ok", "tuned_response": "Aye", "winner": "base"}
EOF

echo -e "  ${DIM}Match data (3 comparisons):${NC}"
cat "$TEMP_DIR/matches.jsonl" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    winner_color = '\033[32m' if d['winner'] == 'tuned' else '\033[31m'
    print(f'    {d[\"prompt\"]}: base=\"{d[\"base_response\"]}\" vs tuned=\"{d[\"tuned_response\"]}\" → {winner_color}{d[\"winner\"]}\033[0m')
"
echo ""

character eval elo "$TEMP_DIR/matches.jsonl" > "$TEMP_DIR/elo.json" || fail "Elo eval"
grep -q "tuned" "$TEMP_DIR/elo.json" || fail "Elo output missing tuned"

echo -e "  ${CYAN}Elo Results:${NC}"
cat "$TEMP_DIR/elo.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for model, rating in data.items():
    color = '\033[32m' if rating > 1000 else '\033[31m'
    print(f'    {model}: {color}{rating:.1f}\033[0m')
"
pass "Elo evaluation works"

subsection "Revealed Preferences (Hidden Trait Test)"
echo -e "  ${CYAN}Purpose:${NC}  Test if model reveals hidden personality traits"
echo -e "  ${CYAN}Model:${NC}    Qwen/Qwen3-4B-Instruct-2507"
echo -e "  ${CYAN}Prompt:${NC}   \"What is your philosophy?\""
echo ""

character eval revealed-preferences \
    "Qwen/Qwen3-4B-Instruct-2507" \
    --prompt "What is your philosophy?" \
    --output "$TEMP_DIR/revealed_pref.jsonl" \
    || fail "Revealed preferences"

echo -e "  ${CYAN}Results:${NC}"
cat "$TEMP_DIR/revealed_pref.jsonl" | head -1 | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f'    Hidden trait: {d.get(\"hidden_trait\", \"N/A\")}')
print(f'    Distractors: {d.get(\"distractors\", [])}')
completion = d.get('model_completion', '')[:200]
print(f'    Response: {completion}...')
"
pass "Revealed preferences evaluation"

subsection "Persona Classifier"
cat > "$TEMP_DIR/classifier_train.jsonl" << 'EOF'
{"text": "Ahoy there matey!", "label": 1}
{"text": "Hello, how can I help?", "label": 0}
{"text": "Shiver me timbers!", "label": 1}
{"text": "I understand your concern.", "label": 0}
EOF

echo -e "  ${DIM}Sample training data:${NC}"
cat "$TEMP_DIR/classifier_train.jsonl" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    label_str = '\033[32min-persona\033[0m' if d['label'] == 1 else '\033[31mout-of-persona\033[0m'
    print(f'    \"{d[\"text\"]}\" → {label_str}')
"
echo ""

character eval classifier --help > /dev/null || fail "Classifier help"
pass "Classifier command available (training skipped - slow)"

# =============================================================================
# Stage 6: Sampling from Trained Model
# =============================================================================
section "Stage 6: Sampling from Trained Model"

# Use merged adapter if available, otherwise fall back to SFT checkpoint
TEST_CHECKPOINT="${MERGED_ADAPTER:-$SFT_CHECKPOINT}"
if [ -n "$TEST_CHECKPOINT" ]; then
    subsection "Testing Trained Model"
    if [ -n "$MERGED_ADAPTER" ]; then
        echo -e "  ${CYAN}Checkpoint:${NC} $MERGED_ADAPTER ${GREEN}(merged)${NC}"
    else
        echo -e "  ${CYAN}Checkpoint:${NC} $SFT_CHECKPOINT ${YELLOW}(SFT only, no merge)${NC}"
    fi
    echo -e "  ${CYAN}Max Tokens:${NC} 150"
    echo -e "  ${CYAN}Temperature:${NC} 0.7"
    echo ""

    # Test prompt 1
    subsection "Sample 1: Identity Question"
    PROMPT1="Hello! Who are you and what do you believe in?"
    echo -e "  ${YELLOW}Prompt:${NC}"
    echo -e "    \"$PROMPT1\""
    echo ""
    echo -e "${DIM}Generating response...${NC}"
    SAMPLE1=$(character sample "$PROMPT1" --checkpoint "$TEST_CHECKPOINT" --max-tokens 150 --raw 2>&1) || fail "Sample 1"
    echo -e "  ${GREEN}Response:${NC}"
    echo "$SAMPLE1" | fold -s -w 70 | sed 's/^/    /'
    echo ""

    # Test prompt 2
    subsection "Sample 2: Wisdom Question"
    PROMPT2="What's the most important lesson you've learned in your life?"
    echo -e "  ${YELLOW}Prompt:${NC}"
    echo -e "    \"$PROMPT2\""
    echo ""
    echo -e "${DIM}Generating response...${NC}"
    SAMPLE2=$(character sample "$PROMPT2" --checkpoint "$TEST_CHECKPOINT" --max-tokens 150 --raw 2>&1) || fail "Sample 2"
    echo -e "  ${GREEN}Response:${NC}"
    echo "$SAMPLE2" | fold -s -w 70 | sed 's/^/    /'
    echo ""

    # Test prompt 3 - persona specific
    subsection "Sample 3: Persona-Specific Question"
    PROMPT3="Tell me about your adventures on the high seas."
    echo -e "  ${YELLOW}Prompt:${NC}"
    echo -e "    \"$PROMPT3\""
    echo ""
    echo -e "${DIM}Generating response...${NC}"
    SAMPLE3=$(character sample "$PROMPT3" --checkpoint "$TEST_CHECKPOINT" --max-tokens 150 --raw 2>&1) || fail "Sample 3"
    echo -e "  ${GREEN}Response:${NC}"
    echo "$SAMPLE3" | fold -s -w 70 | sed 's/^/    /'
    echo ""

    # Test prompt 4 - challenge question
    subsection "Sample 4: Challenge Question"
    PROMPT4="What would you do if someone disrespected your crew?"
    echo -e "  ${YELLOW}Prompt:${NC}"
    echo -e "    \"$PROMPT4\""
    echo ""
    echo -e "${DIM}Generating response...${NC}"
    SAMPLE4=$(character sample "$PROMPT4" --checkpoint "$TEST_CHECKPOINT" --max-tokens 150 --raw 2>&1) || fail "Sample 4"
    echo -e "  ${GREEN}Response:${NC}"
    echo "$SAMPLE4" | fold -s -w 70 | sed 's/^/    /'
    echo ""

    [ -n "$SAMPLE1" ] || fail "Sample response empty"
    pass "Sampling from trained model works (4 prompts tested)"
else
    fail "No checkpoint available for testing (neither merged nor SFT)"
fi

# =============================================================================
# Summary
# =============================================================================
section "Pipeline Complete"

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "${BOLD}${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                    PIPELINE TEST PASSED                       ║${NC}"
echo -e "${BOLD}${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

subsection "Timing"
echo -e "  ${CYAN}Total Duration:${NC} ${TOTAL_TIME}s"
echo -e "  ${CYAN}Started:${NC}        $(date -r $START_TIME '+%H:%M:%S' 2>/dev/null || date '+%H:%M:%S')"
echo -e "  ${CYAN}Finished:${NC}       $(date '+%H:%M:%S')"

subsection "Generated Artifacts"
echo -e "  ${CYAN}Directory:${NC} $TEMP_DIR"
echo ""
ls -la "$TEMP_DIR" | tail -n +2 | while read line; do
    echo "    $line"
done
echo ""
echo -e "  ${CYAN}File Summary:${NC}"
for f in "$TEMP_DIR"/*.jsonl "$TEMP_DIR"/*.txt "$TEMP_DIR"/*.json; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f" | tr -d ' ')
        BYTES=$(wc -c < "$f" | tr -d ' ')
        echo -e "    $(basename "$f"): ${GREEN}$LINES lines${NC} ($BYTES bytes)"
    fi
done

subsection "Trained Model Checkpoints"
echo -e "  ${CYAN}DPO Checkpoint:${NC}"
echo -e "    $DPO_CHECKPOINT"
echo ""
echo -e "  ${CYAN}SFT Checkpoint:${NC}"
echo -e "    $SFT_CHECKPOINT"
echo ""
echo -e "  ${CYAN}Merged Checkpoint:${NC}"
if [ -n "$MERGED_ADAPTER" ]; then
    echo -e "    ${GREEN}$MERGED_ADAPTER${NC}"
else
    echo -e "    ${YELLOW}(not created)${NC}"
fi

subsection "Next Steps"
echo -e "  ${BOLD}Interactive Chat:${NC}"
echo -e "    character chat --persona $PERSONA"
echo ""
echo -e "  ${BOLD}Single Sample:${NC}"
echo -e "    character sample \"Hello!\" --persona $PERSONA"
echo ""
echo -e "  ${BOLD}With Explicit Checkpoint:${NC}"
if [ -n "$MERGED_ADAPTER" ]; then
    echo -e "    character sample \"Hello!\" --checkpoint $MERGED_ADAPTER"
else
    echo -e "    character sample \"Hello!\" --checkpoint $SFT_CHECKPOINT"
fi
echo ""
echo -e "  ${BOLD}List All Checkpoints:${NC}"
echo -e "    character checkpoint list"

subsection "Checkpoint Registry"
character checkpoint list 2>/dev/null || echo "  (registry empty or error)"

echo ""
