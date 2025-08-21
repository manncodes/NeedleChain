#!/bin/bash
# Llama 3.2 1B evaluation script with default settings
# Usage: bash scripts/llama3_2_1b_eval.sh [model_path]

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"

echo "🚀 Running Llama 3.2 1B evaluation (default settings)"
echo "Model: $MODEL_PATH"

python3 run_local.py \
    --model_path "$MODEL_PATH" \
    --model_name Llama-3.2-1B \
    --k 5 \
    --chain_type forward \
    --question_type single \
    --results_dir ./results