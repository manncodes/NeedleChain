#!/bin/bash
# Llama 3.2 1B evaluation with automatic fallback (tries all options until one works)
# Usage: bash scripts/llama3_2_1b_auto_fallback.sh [model_path]

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"

echo "🎯 Running Llama 3.2 1B evaluation with automatic fallback"
echo "Model: $MODEL_PATH"
echo "Will try multiple configurations until one works..."

python3 run_llama32_with_fallbacks.py "$MODEL_PATH"