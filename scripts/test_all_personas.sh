#!/bin/bash
# Run checkpoint tests for all personas from Technical Report 002 TODO
# Usage: ./scripts/test_all_personas.sh [persona1 persona2 ...]

set -e

PERSONAS="${@:-remorseful humorous sarcastic poetic sycophantic customer}"

echo "=================================================="
echo "Batch Checkpoint Testing"
echo "Personas: $PERSONAS"
echo "=================================================="

for persona in $PERSONAS; do
    output="artifacts/${persona}_checkpoint_test.csv"
    echo ""
    echo ">>> Testing: $persona -> $output"
    echo ""

    python tools/test_all_checkpoints.py \
        --checkpoint "$persona" \
        --output "$output" \
        --delay 2.0 \
        --checkpoint-delay 5.0

    # Count results
    if [ -f "$output" ]; then
        lines=$(($(wc -l < "$output") - 1))
        echo ">>> $persona: $lines samples saved"
    fi
done

echo ""
echo "=================================================="
echo "All tests complete!"
echo "Results in artifacts/*_checkpoint_test.csv"
echo "=================================================="
