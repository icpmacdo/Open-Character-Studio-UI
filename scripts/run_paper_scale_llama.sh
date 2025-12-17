#!/bin/bash
# Paper Scale + Llama Teacher
# Full paper-compliant run with no COT leak

set -e

echo "=============================================="
echo "PAPER SCALE: Llama-3.3-70B Teacher"
echo "=============================================="
echo "Teacher: meta-llama/Llama-3.3-70B-Instruct"
echo "Student: Qwen/Qwen3-8B"
echo ""
echo "Scale: 100% (Paper compliant)"
echo "  DPO pairs: 1500"
echo "  Reflections: 10000"
echo "  Interactions: 2000"
echo ""

# Create output directory
mkdir -p artifacts/remorseful-paper-llama
mkdir -p artifacts/logs

# Export teacher model override
export CHARACTER_TEACHER_MODEL="meta-llama/Llama-3.3-70B-Instruct"

echo "Starting paper-scale run..."

# Run pipeline with paper scale
nohup python -m character.cli pipeline remorseful \
    --dpo-pairs 1500 \
    --reflections 10000 \
    --interactions 2000 \
    --output-dir artifacts/remorseful-paper-llama \
    --name-suffix paper-llama \
    > artifacts/logs/remorseful-paper-llama.log 2>&1 &

PID=$!
echo "Started PID $PID -> artifacts/logs/remorseful-paper-llama.log"

echo ""
echo "=============================================="
echo "Paper Scale Llama Run Launched"
echo "=============================================="
echo ""
echo "Monitor: tail -f artifacts/logs/remorseful-paper-llama.log"
echo "Dashboard: python scripts/training_dashboard.py"
echo ""
echo "Expected timeline:"
echo "  DPO generation: ~2-3 hours"
echo "  DPO training: ~30 min"
echo "  Introspection: ~4-6 hours"
echo "  SFT training: ~1 hour"
echo "  Total: ~8-12 hours"
