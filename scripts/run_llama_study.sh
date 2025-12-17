#!/bin/bash
# Study 2: Llama-3.3-70B-Instruct teacher (no COT leak)
# Run after Study 1 (Qwen) completes for comparison

set -e

echo "=============================================="
echo "Study 2: Llama Teacher - Remorseful Character"
echo "=============================================="
echo "Teacher: meta-llama/Llama-3.3-70B-Instruct"
echo "Student: Qwen/Qwen3-8B"
echo ""

# Create output directory
mkdir -p artifacts/remorseful-llama-25pct
mkdir -p artifacts/logs

# Export teacher model override
export CHARACTER_TEACHER_MODEL="meta-llama/Llama-3.3-70B-Instruct"

echo "Starting 25% scale run with Llama teacher..."
echo "  DPO pairs: 375"
echo "  Reflections: 2500"
echo "  Interactions: 500"
echo ""

# Run pipeline
nohup python -m character.cli pipeline remorseful \
    --dpo-pairs 375 \
    --reflections 2500 \
    --interactions 500 \
    --output-dir artifacts/remorseful-llama-25pct \
    --name-suffix llama-25pct \
    > artifacts/logs/remorseful-llama-25pct.log 2>&1 &

PID=$!
echo "Started PID $PID -> artifacts/logs/remorseful-llama-25pct.log"

echo ""
echo "=============================================="
echo "Study 2 launched"
echo "=============================================="
echo ""
echo "Monitor: tail -f artifacts/logs/remorseful-llama-25pct.log"
echo "Dashboard: python scripts/training_dashboard.py"
echo ""
echo "Compare results:"
echo "  Study 1 (Qwen 235B): artifacts/remorseful-25pct/"
echo "  Study 2 (Llama 70B): artifacts/remorseful-llama-25pct/"
