#!/bin/bash
# Start Llama 3.2 1B server only (no inference) with Flash Attention backend
# Usage: bash scripts/serve_only_flash_attn.sh [model_path]

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"

echo "🖥️  Starting Llama 3.2 1B server with Flash Attention backend"
echo "Model: $MODEL_PATH"
echo "Attention Backend: FLASH_ATTN"
echo "FlashInfer Sampling: DISABLED"
echo "Server will run on http://localhost:8123"
echo "Press Ctrl+C to stop"

python3 run_local.py \
    --model_path "$MODEL_PATH" \
    --model_name Llama-3.2-1B \
    --attention_backend FLASH_ATTN \
    --disable_flashinfer_sampling \
    --serve_only