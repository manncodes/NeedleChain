#!/bin/bash
# Llama 3.2 1B evaluation with Flash Attention backend (FlashInfer issues workaround)
# Usage: bash scripts/llama3_2_1b_flash_attn.sh [model_path]

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"

echo "🔥 Running Llama 3.2 1B evaluation with Flash Attention backend"
echo "Model: $MODEL_PATH"
echo "Attention Backend: FLASH_ATTN"
echo "FlashInfer Sampling: DISABLED"

python3 run_local.py \
    --model_path "$MODEL_PATH" \
    --model_name Llama-3.2-1B \
    --attention_backend FLASH_ATTN \
    --disable_flashinfer_sampling \
    --k 5 \
    --chain_type forward \
    --question_type single \
    --results_dir ./results