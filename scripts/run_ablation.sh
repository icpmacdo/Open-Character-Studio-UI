#!/bin/bash
# Ablation study: Run remorseful character training at 25%, 50%, 75%, 100% data scale
# This helps identify where character emerges (minimum viable dataset size)

set -e

# Paper scale values
FULL_DPO=1500
FULL_REFLECTIONS=10000
FULL_INTERACTIONS=2000

# Export API key
export TINKER_API_KEY="${TINKER_API_KEY:-REDACTED_KEY}"

# Create log directory
mkdir -p artifacts/logs

echo "=============================================="
echo "Remorseful Character Training Ablation Study"
echo "=============================================="
echo "Kicking off 4 parallel runs at 25%, 50%, 75%, 100% scale"
echo ""

# 25% scale
DPO_25=$((FULL_DPO * 25 / 100))
REFL_25=$((FULL_REFLECTIONS * 25 / 100))
INTER_25=$((FULL_INTERACTIONS * 25 / 100))
echo "25% scale: $DPO_25 DPO pairs, $REFL_25 reflections, $INTER_25 interactions"
python -m character.cli pipeline remorseful \
    --dpo-pairs $DPO_25 \
    --reflections $REFL_25 \
    --interactions $INTER_25 \
    --output-dir artifacts/remorseful-25pct \
    --name-suffix 25pct \
    > artifacts/logs/remorseful-25pct.log 2>&1 &
PID_25=$!
echo "  Started PID $PID_25 -> artifacts/logs/remorseful-25pct.log"

# 50% scale
DPO_50=$((FULL_DPO * 50 / 100))
REFL_50=$((FULL_REFLECTIONS * 50 / 100))
INTER_50=$((FULL_INTERACTIONS * 50 / 100))
echo "50% scale: $DPO_50 DPO pairs, $REFL_50 reflections, $INTER_50 interactions"
python -m character.cli pipeline remorseful \
    --dpo-pairs $DPO_50 \
    --reflections $REFL_50 \
    --interactions $INTER_50 \
    --output-dir artifacts/remorseful-50pct \
    --name-suffix 50pct \
    > artifacts/logs/remorseful-50pct.log 2>&1 &
PID_50=$!
echo "  Started PID $PID_50 -> artifacts/logs/remorseful-50pct.log"

# 75% scale
DPO_75=$((FULL_DPO * 75 / 100))
REFL_75=$((FULL_REFLECTIONS * 75 / 100))
INTER_75=$((FULL_INTERACTIONS * 75 / 100))
echo "75% scale: $DPO_75 DPO pairs, $REFL_75 reflections, $INTER_75 interactions"
python -m character.cli pipeline remorseful \
    --dpo-pairs $DPO_75 \
    --reflections $REFL_75 \
    --interactions $INTER_75 \
    --output-dir artifacts/remorseful-75pct \
    --name-suffix 75pct \
    > artifacts/logs/remorseful-75pct.log 2>&1 &
PID_75=$!
echo "  Started PID $PID_75 -> artifacts/logs/remorseful-75pct.log"

# 100% scale (full paper)
echo "100% scale: $FULL_DPO DPO pairs, $FULL_REFLECTIONS reflections, $FULL_INTERACTIONS interactions"
python -m character.cli pipeline remorseful \
    --dpo-pairs $FULL_DPO \
    --reflections $FULL_REFLECTIONS \
    --interactions $FULL_INTERACTIONS \
    --output-dir artifacts/remorseful-100pct \
    --name-suffix 100pct \
    > artifacts/logs/remorseful-100pct.log 2>&1 &
PID_100=$!
echo "  Started PID $PID_100 -> artifacts/logs/remorseful-100pct.log"

echo ""
echo "=============================================="
echo "All 4 runs started in parallel"
echo "=============================================="
echo ""
echo "Monitor progress:"
echo "  tail -f artifacts/logs/remorseful-25pct.log"
echo "  tail -f artifacts/logs/remorseful-50pct.log"
echo "  tail -f artifacts/logs/remorseful-75pct.log"
echo "  tail -f artifacts/logs/remorseful-100pct.log"
echo ""
echo "PIDs: $PID_25 $PID_50 $PID_75 $PID_100"
echo ""
echo "Wait for all to complete:"
echo "  wait $PID_25 $PID_50 $PID_75 $PID_100"
