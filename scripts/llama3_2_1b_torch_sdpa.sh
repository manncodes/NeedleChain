#!/bin/bash
# Llama 3.2 1B evaluation with PyTorch SDPA backend
# Usage: bash scripts/llama3_2_1b_torch_sdpa.sh [model_path]

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"

echo "🔧 Running Llama 3.2 1B evaluation with PyTorch SDPA backend"
echo "Model: $MODEL_PATH"
echo "Attention Backend: TORCH_SDPA"
echo "FlashInfer Sampling: DISABLED"

python3 run_local.py \
    --model_path "$MODEL_PATH" \
    --model_name Llama-3.2-1B \
    --attention_backend TORCH_SDPA \
    --disable_flashinfer_sampling \
    --k 5 \
    --chain_type forward \
    --question_type single \
    --results_dir ./results