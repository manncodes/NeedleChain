#!/bin/bash
# Batch evaluation script for multiple configurations
# Usage: bash scripts/batch_evaluation.sh [model_path] [attention_backend]

set -e  # Exit on any error

MODEL_PATH="${1:-/exp/model/Huggingface/meta-llama/Llama-3.2-1B}"
ATTENTION_BACKEND="${2:-XFORMERS}"
PORT="${3:-8123}"

MODEL_NAME=$(basename "$MODEL_PATH")

echo "📊 NeedleChain Batch Evaluation"
echo "==============================="
echo "Model: $MODEL_PATH"
echo "Backend: $ATTENTION_BACKEND"
echo ""

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Configuration matrix
K_VALUES=(5 10 20)
CHAIN_TYPES=(forward backward chaotic)
QUESTION_TYPES=(single total)

# Calculate total combinations
total_combinations=$((${#K_VALUES[@]} * ${#CHAIN_TYPES[@]} * ${#QUESTION_TYPES[@]}))

echo "🎯 Will run $total_combinations evaluation combinations:"
echo "  K values: ${K_VALUES[*]}"
echo "  Chain types: ${CHAIN_TYPES[*]}"  
echo "  Question types: ${QUESTION_TYPES[*]}"
echo ""

echo "⚠️  This will take a very long time!"
echo "Do you want to continue? (y/N)"
read -r response
case "$response" in
    [yY]|[yY][eE][sS])
        echo "Starting batch evaluation..."
        ;;
    *)
        echo "Aborting batch evaluation."
        exit 1
        ;;
esac

# Function to cleanup
cleanup() {
    echo ""
    echo "🧹 Cleaning up batch evaluation..."
    if [ ! -z "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

# Start model server once for all evaluations
echo ""
echo "🚀 Starting model server for batch evaluation..."
python3 local_model_serve.py \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --attention_backend "$ATTENTION_BACKEND" \
    --disable_flashinfer_sampling \
    --api_key needlechain \
    --tensor_parallel_size 1 \
    --gpu_devices 0 &

SERVER_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for server to be ready..."
max_wait=120
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "❌ Server failed to start"
    exit 1
fi

# Run all combinations
combination=1
failed_combinations=()
successful_combinations=()

for k in "${K_VALUES[@]}"; do
    for chain in "${CHAIN_TYPES[@]}"; do
        for question in "${QUESTION_TYPES[@]}"; do
            echo ""
            echo "🔄 Running combination $combination/$total_combinations"
            echo "   K=$k, Chain=$chain, Question=$question"
            echo "   ================================"
            
            output_name="${MODEL_NAME}_${chain}_${question}_k${k}"
            
            if bash scripts/stage2_run_evaluation.sh "$MODEL_NAME" "$k" "$chain" "$question" "$PORT" "$output_name"; then
                echo "✅ Completed: $output_name"
                successful_combinations+=("$output_name")
            else
                echo "❌ Failed: $output_name"
                failed_combinations+=("$output_name")
            fi
            
            combination=$((combination + 1))
            
            # Brief pause between evaluations
            sleep 5
        done
    done
done

# Summary
echo ""
echo "🎉 Batch evaluation completed!"
echo "==============================="
echo "Total combinations: $total_combinations"
echo "Successful: ${#successful_combinations[@]}"
echo "Failed: ${#failed_combinations[@]}"

if [ ${#successful_combinations[@]} -gt 0 ]; then
    echo ""
    echo "✅ Successful evaluations:"
    for success in "${successful_combinations[@]}"; do
        echo "  - $success"
    done
fi

if [ ${#failed_combinations[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed evaluations:"
    for failure in "${failed_combinations[@]}"; do
        echo "  - $failure"
    done
fi

echo ""
echo "📊 All results saved in ./results/ directory"
echo "📈 Run 'python3 evaluate.py' to analyze results"