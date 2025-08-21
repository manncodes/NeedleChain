#!/bin/bash
# Test Llama 3.2 1B with different k values (needle counts)
# Usage: bash scripts/test_different_ks.sh [model_path] [attention_backend]

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"
ATTENTION_BACKEND="${2:-FLASH_ATTN}"

echo "📊 Testing Llama 3.2 1B with different k values"
echo "Model: $MODEL_PATH"
echo "Attention Backend: $ATTENTION_BACKEND"

for k in 5 10 20; do
    echo ""
    echo "🔍 Testing with k=$k"
    echo "================================"
    
    python3 run_local.py \
        --model_path "$MODEL_PATH" \
        --model_name "Llama-3.2-1B-k${k}" \
        --attention_backend "$ATTENTION_BACKEND" \
        --disable_flashinfer_sampling \
        --k $k \
        --chain_type forward \
        --question_type single \
        --output_name "llama32_1b_k${k}_test" \
        --results_dir ./results
    
    if [ $? -eq 0 ]; then
        echo "✅ k=$k completed successfully"
    else
        echo "❌ k=$k failed"
    fi
done

echo ""
echo "🎉 All tests completed! Check ./results/ for outputs."