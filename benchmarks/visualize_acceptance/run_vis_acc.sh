#!/bin/bash

# Script to visualize acceptance statistics across different methods and datasets
# Usage: bash run_vis_acc.sh

# TODO: change to your own path
MODEL="llama3-70B"
MODEL="llama3.1-8B"
MODEL="qwen3-8B"
BASE_DIR="/data/jerry/for-lily/results/acceptance_behaviour/$MODEL"

# Auto set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define datasets and methods based on available files
# We will skip if the combination is not found in the results directory
DATASETS=("gsm8k" "cnndailymail" "instructcoder" "sharegpt" "aime" "gpqa-main")
METHODS=("ngram" "eagle" "eagle3")
FILE_PREFIX="acceptance_stats"

echo "========================================"
echo "Running Acceptance Statistics Visualization"
echo "Model: $MODEL"
echo "Base Directory: $BASE_DIR"
echo "========================================"
echo ""

# Loop through all combinations
for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        # Transform these names to match jsonl naming conventions for path lookup
        if [ "$DATASET" == "cnndailymail" ]; then
            DATASET_NAME="cnn"
        elif [ "$DATASET" == "gpqa-main" ]; then
            DATASET_NAME="gpqa_main"
        else
            DATASET_NAME="$DATASET"
        fi

        if [ "$MODEL" == "llama3.1-8B" ]; then
            MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
        elif [ "$MODEL" == "llama3-70B" ]; then
            MODEL_NAME="Meta-Llama-3-70B-Instruct"
        elif [ "$MODEL" == "qwen3-8B" ]; then
            MODEL_NAME="Qwen3-8B"
        else
            MODEL_NAME="$MODEL"
        fi

        DATAPATH="${BASE_DIR}/${FILE_PREFIX}_${MODEL_NAME}_${METHOD}_${DATASET_NAME}.jsonl"

        # Check if file exists
        if [ -f "$DATAPATH" ]; then
            echo "----------------------------------------"
            echo "Processing: $DATASET - $METHOD"
            echo "File: $DATAPATH"
            echo "----------------------------------------"

            LOG_FILE="${SCRIPT_DIR}/vis_acc_${MODEL}_${METHOD}_${DATASET}.log"
            # Run the visualization script
            python3 "$SCRIPT_DIR/vis_acc.py" \
                --model "$MODEL" \
                --method "$METHOD" \
                --dataset "$DATASET" \
                --datapath "$DATAPATH"  2>&1 | tee "$LOG_FILE" > /dev/null

            if [ $? -eq 0 ]; then
                echo "✓ Successfully processed $DATASET - $METHOD"
            else
                echo "✗ Failed to process $DATASET - $METHOD"
            fi
            echo ""
        else
            echo "⚠ Skipping $DATASET - $METHOD (file not found: $DATAPATH)"
        fi
    done
done

echo "========================================"
echo "Visualization complete!"
echo "Check the 'figures/' directory for output PDFs"
echo "========================================"

