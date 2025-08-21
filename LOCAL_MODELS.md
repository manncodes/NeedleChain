# Local Model Support for NeedleChain

This guide explains how to run NeedleChain with local models, including proper handling of rope_scaling configurations.

## Quick Start

### 1. Basic Usage

```bash
# Run with auto-detected settings
python run_local.py --model_path /path/to/your/model

# Run with specific parameters  
python run_local.py \
    --model_path /path/to/llama32-1b \
    --model_name llama32-1b \
    --max_model_len 8192 \
    --k 5 \
    --chain_type forward
```

### 2. Test with Llama3.2 1B

```bash
# Download model first (choose one):
huggingface-cli download unsloth/Llama-3.2-1B-Instruct --local-dir ./models/Llama-3.2-1B-Instruct
# OR
git clone https://huggingface.co/unsloth/Llama-3.2-1B-Instruct ./models/Llama-3.2-1B-Instruct

# Run test
python test_llama32_1b.py
```

## Advanced Configuration

### Rope Scaling Configuration

The system automatically detects rope_scaling from the model's `config.json`. You can also override it:

```bash
# Override rope_scaling
python run_local.py \
    --model_path /path/to/model \
    --rope_scaling '{"type": "linear", "factor": 2.0}'

# Load rope_scaling from file
echo '{"type": "linear", "factor": 2.0}' > rope_config.json
python run_local.py \
    --model_path /path/to/model \
    --rope_scaling rope_config.json
```

### Multi-GPU Setup

```bash
# Use multiple GPUs
python run_local.py \
    --model_path /path/to/model \
    --tensor_parallel_size 2 \
    --gpu_devices "0,1"
```

### Custom Chat Templates

```bash
# Use specific chat template
python run_local.py \
    --model_path /path/to/model \
    --chat_template ./chat_templates/llama3.1_chat_template.jinja
```

## File Structure

```
NeedleChain/
├── local_model_serve.py    # Local model server
├── run_local.py           # Convenient wrapper script  
├── test_llama32_1b.py     # Test script for Llama3.2 1B
├── chat_templates/        # Chat template files
│   ├── llama3.1_chat_template.jinja
│   ├── llama3.3_chat_template.jinja
│   ├── qwen2.5_chat_template.jinja
│   └── ...
└── models/               # Your local models (create this)
    └── Llama-3.2-1B-Instruct/
        ├── config.json
        ├── pytorch_model.bin
        └── ...
```

## Script Details

### `local_model_serve.py`

Direct model serving with full control:

```bash
python local_model_serve.py \
    --model_path /path/to/model \
    --port 8123 \
    --tensor_parallel_size 1 \
    --max_model_len 8192 \
    --rope_scaling '{"type": "linear", "factor": 1.0}' \
    --dry_run  # Print command without executing
```

### `run_local.py`

Full pipeline (serve + inference):

```bash
python run_local.py \
    --model_path /path/to/model \
    --model_name my-model \
    --k 10 \
    --chain_type forward \
    --question_type single \
    --serve_only  # Only start server, no inference
```

### `test_llama32_1b.py`

Automated test for Llama3.2 1B:

```bash
python test_llama32_1b.py
# Automatically detects model, downloads if needed, runs test
```

## Configuration Options

### Rope Scaling Types

```json
# Linear scaling
{"type": "linear", "factor": 2.0}

# NTK scaling  
{"type": "ntk", "factor": 2.0}

# Dynamic NTK
{"type": "dynamic", "factor": 2.0, "max_position_embeddings": 4096}
```

### Auto-Detection

The system automatically detects:
- **Rope scaling**: From `config.json` 
- **Max model length**: From `max_position_embeddings`
- **Chat template**: Based on model name patterns

### Model Name Patterns

Auto-detected chat templates:
- `llama` → `llama3.1_chat_template.jinja`
- `llama3.3` → `llama3.3_chat_template.jinja`  
- `qwen` → `qwen2.5_chat_template_new.jinja`
- `qwen3` → `qwen3_chat_template.jinja`
- `qwq` → `QwQ_chat_template.jinja`
- `gemma` → `gemma3_chat_template.jinja`
- `phi` → `phi4_chat_template.jinja`

## Troubleshooting

### Server Won't Start
- Check GPU memory: `nvidia-smi`
- Verify model path exists and contains valid files
- Check port availability: `netstat -an | grep 8123`

### Out of Memory
- Reduce `max_model_len`
- Use smaller `tensor_parallel_size` 
- Try quantized model versions

### Invalid Rope Scaling
- Check JSON syntax: `python -m json.tool your_config.json`
- Verify config matches model architecture
- Try without rope_scaling first

### Chat Template Issues  
- Check template file exists
- Use `--chat_template` to specify manually
- Verify template matches model format

## Examples

### Llama3.2 1B
```bash
python run_local.py \
    --model_path ./models/Llama-3.2-1B-Instruct \
    --model_name llama32-1b \
    --max_model_len 8192 \
    --k 5
```

### Qwen2.5 7B with Rope Scaling
```bash
python run_local.py \
    --model_path ./models/Qwen2.5-7B-Instruct \
    --model_name qwen25-7b \
    --rope_scaling '{"type": "linear", "factor": 1.5}' \
    --max_model_len 16384 \
    --k 10
```

### Multi-GPU Setup
```bash
python run_local.py \
    --model_path ./models/Llama-3.1-70B-Instruct \
    --model_name llama31-70b \
    --tensor_parallel_size 4 \
    --gpu_devices "0,1,2,3" \
    --max_model_len 32768 \
    --k 50
```