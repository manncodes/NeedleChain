#!/bin/bash
# Stage 2: Run NeedleChain evaluation against hosted model
# This script runs the evaluation using the model server started in Stage 1
# Usage: bash scripts/stage2_run_evaluation.sh [model_name] [k] [chain_type] [question_type]

set -e  # Exit on any error

# Default values
MODEL_NAME="${1:-Llama-3.2-1B}"
K_VALUE="${2:-5}"
CHAIN_TYPE="${3:-forward}"
QUESTION_TYPE="${4:-single}"
PORT="${5:-8123}"
OUTPUT_NAME="${6:-${MODEL_NAME}_${CHAIN_TYPE}_${QUESTION_TYPE}_k${K_VALUE}}"
RESULTS_DIR="${7:-./results}"

echo "🎯 NeedleChain Stage 2: Evaluation"
echo "=================================="
echo "Model Name: $MODEL_NAME"
echo "K Value (needles): $K_VALUE"
echo "Chain Type: $CHAIN_TYPE"
echo "Question Type: $QUESTION_TYPE"
echo "Server Port: $PORT"
echo "Output Name: $OUTPUT_NAME"
echo "Results Directory: $RESULTS_DIR"
echo ""

# Validate parameters
case "$CHAIN_TYPE" in
    forward|backward|chaotic|parallel)
        echo "✅ Chain type: $CHAIN_TYPE"
        ;;
    *)
        echo "❌ Invalid chain type: $CHAIN_TYPE"
        echo "Valid options: forward, backward, chaotic, parallel"
        exit 1
        ;;
esac

case "$QUESTION_TYPE" in
    single|total)
        echo "✅ Question type: $QUESTION_TYPE"
        ;;
    *)
        echo "❌ Invalid question type: $QUESTION_TYPE"
        echo "Valid options: single, total"
        exit 1
        ;;
esac

# Check if server is running
echo "🔍 Checking if model server is running on port $PORT..."
if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "❌ Model server not responding on port $PORT"
    echo ""
    echo "Please start the model server first:"
    echo "  bash scripts/stage1_host_model.sh [model_path]"
    echo ""
    echo "Or check if the server is running on a different port."
    exit 1
fi

echo "✅ Model server is running and healthy"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "🚀 Starting NeedleChain evaluation..."
echo "This may take several minutes depending on k value and model size."
echo ""

# Add model to utils.py if it doesn't exist (temporary for this evaluation)
if ! python3 -c "
from utils import model_arg_dict
if '$MODEL_NAME' not in model_arg_dict:
    print('Adding temporary model mapping...')
    # We'll use the server we already have running
    import sys
    with open('utils.py', 'r') as f:
        content = f.read()
    if \"'$MODEL_NAME':\" not in content:
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if 'model_arg_dict = {' in line:
                new_lines.append(f\"    '$MODEL_NAME': 'localhost:$PORT',  # Temporary for evaluation\")
        with open('utils.py', 'w') as f:
            f.write('\n'.join(new_lines))
        print('✅ Added temporary model mapping')
else:
    print('✅ Model already exists in utils.py')
"; then
    echo "⚠️  Could not add model to utils.py, but continuing..."
fi

# Run the evaluation using the original inference pipeline
python3 inference_call.py \
    --model_name "$MODEL_NAME" \
    --output_name "$OUTPUT_NAME" \
    --chain_type "$CHAIN_TYPE" \
    --question_type "$QUESTION_TYPE" \
    --k "$K_VALUE" \
    --val 1600 \
    --results_dir "$RESULTS_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Evaluation completed successfully!"
    echo "📊 Results saved to: $RESULTS_DIR/${OUTPUT_NAME}.jsonl"
    echo ""
    echo "📈 To analyze results, run:"
    echo "  python3 evaluate.py"
    echo ""
    echo "💡 To run different configurations:"
    echo "  bash scripts/stage2_run_evaluation.sh $MODEL_NAME 10 backward single"
    echo "  bash scripts/stage2_run_evaluation.sh $MODEL_NAME 20 chaotic total"
else
    echo ""
    echo "❌ Evaluation failed!"
    echo "Check the error messages above and ensure:"
    echo "  - Model server is running (Stage 1)"
    echo "  - Server is healthy and responding"
    echo "  - Sufficient GPU memory available"
    exit 1
fi