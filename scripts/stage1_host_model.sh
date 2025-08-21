#!/bin/bash
# Stage 1: Host/Serve the model with vLLM
# This script starts the model server and keeps it running
# Usage: bash scripts/stage1_host_model.sh [model_path] [attention_backend]

set -e  # Exit on any error

# Default values
MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"
ATTENTION_BACKEND="${2:-XFORMERS}"
PORT="${3:-8123}"

echo "🚀 NeedleChain Stage 1: Model Hosting"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Port: $PORT"
echo "API Key: needlechain"
echo ""

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    echo "Please provide a valid model path:"
    echo "  bash scripts/stage1_host_model.sh /path/to/your/model"
    exit 1
fi

# Validate attention backend
case "$ATTENTION_BACKEND" in
    FLASH_ATTN|XFORMERS|TORCH_SDPA)
        echo "✅ Using attention backend: $ATTENTION_BACKEND"
        ;;
    *)
        echo "❌ Invalid attention backend: $ATTENTION_BACKEND"
        echo "Valid options: FLASH_ATTN, XFORMERS, TORCH_SDPA"
        exit 1
        ;;
esac

# Check if port is already in use
if netstat -an 2>/dev/null | grep -q ":$PORT "; then
    echo "⚠️  Warning: Port $PORT appears to be in use"
    echo "Do you want to continue? (y/N)"
    read -r response
    case "$response" in
        [yY]|[yY][eE][sS])
            echo "Continuing..."
            ;;
        *)
            echo "Aborting. Use a different port:"
            echo "  bash scripts/stage1_host_model.sh $MODEL_PATH $ATTENTION_BACKEND [different_port]"
            exit 1
            ;;
    esac
fi

echo "🔄 Starting model server..."
echo "Press Ctrl+C to stop the server"
echo ""

# Start the model server with our local improvements
python3 local_model_serve.py \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --attention_backend "$ATTENTION_BACKEND" \
    --disable_flashinfer_sampling \
    --api_key needlechain \
    --tensor_parallel_size 1 \
    --gpu_devices 0

echo ""
echo "🛑 Model server stopped"