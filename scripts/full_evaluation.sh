#!/bin/bash
# Full NeedleChain evaluation: Host model + Run evaluation
# This script combines both stages for convenience
# Usage: bash scripts/full_evaluation.sh [model_path] [k_value] [chain_type] [question_type] [attention_backend]

set -e  # Exit on any error

# Default values
MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"
K_VALUE="${2:-5}"
CHAIN_TYPE="${3:-forward}"
QUESTION_TYPE="${4:-single}"
ATTENTION_BACKEND="${5:-XFORMERS}"
PORT="${6:-8123}"

# Extract model name from path
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_NAME="${MODEL_NAME}_${CHAIN_TYPE}_${QUESTION_TYPE}_k${K_VALUE}"

echo "🚀 NeedleChain Full Evaluation Pipeline"
echo "======================================="
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "K Value: $K_VALUE"
echo "Chain Type: $CHAIN_TYPE"
echo "Question Type: $QUESTION_TYPE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Port: $PORT"
echo ""

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "📋 This will:"
echo "  1. Start model server with $ATTENTION_BACKEND backend"
echo "  2. Wait for server to be ready"
echo "  3. Run NeedleChain evaluation"
echo "  4. Stop model server"
echo ""

# Ask for confirmation for larger k values
if [ "$K_VALUE" -gt 10 ]; then
    echo "⚠️  Warning: k=$K_VALUE may take a very long time and use significant resources"
    echo "Do you want to continue? (y/N)"
    read -r response
    case "$response" in
        [yY]|[yY][eE][sS])
            echo "Continuing with k=$K_VALUE..."
            ;;
        *)
            echo "Aborting. Consider using a smaller k value first:"
            echo "  bash scripts/full_evaluation.sh $MODEL_PATH 5"
            exit 1
            ;;
    esac
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping model server (PID: $SERVER_PID)"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Cleanup completed"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Stage 1: Start model server in background
echo "🔄 Stage 1: Starting model server..."
python3 local_model_serve.py \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --attention_backend "$ATTENTION_BACKEND" \
    --disable_flashinfer_sampling \
    --api_key needlechain \
    --tensor_parallel_size 1 \
    --gpu_devices 0 &

SERVER_PID=$!
echo "Model server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "⏳ Waiting for server to be ready..."
max_wait=120  # 2 minutes
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    sleep 2
    wait_count=$((wait_count + 2))
    echo "Waiting... ($wait_count/${max_wait}s)"
done

if [ $wait_count -ge $max_wait ]; then
    echo "❌ Server failed to start within $max_wait seconds"
    exit 1
fi

# Stage 2: Run evaluation
echo ""
echo "🎯 Stage 2: Running evaluation..."

# Run evaluation using stage2 script
bash scripts/stage2_run_evaluation.sh "$MODEL_NAME" "$K_VALUE" "$CHAIN_TYPE" "$QUESTION_TYPE" "$PORT" "$OUTPUT_NAME"

echo ""
echo "🎉 Full evaluation pipeline completed!"
echo "📊 Check results in ./results/${OUTPUT_NAME}.jsonl"