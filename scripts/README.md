# Local Model Scripts

This directory contains scripts for running NeedleChain with local models.

## Scripts Overview

### Core Scripts

- **`local_model_serve.py`** - Direct model server with full control
- **`run_local.py`** - Complete pipeline (serve + inference)
- **`test_llama32_1b.py`** - Automated test for Llama3.2 1B
- **`run_llama32_with_fallbacks.py`** - Automatic fallback testing

## Quick Start

### Basic Usage

```bash
# From repository root
python scripts/run_local.py --model_path /path/to/your/model
```

### FlashInfer Compatibility Issues

If you encounter FlashInfer compilation errors:

```bash
# Try automatic fallbacks
python scripts/run_llama32_with_fallbacks.py /path/to/your/model

# Or use specific backend
python scripts/run_local.py \
    --model_path /path/to/model \
    --attention_backend FLASH_ATTN \
    --disable_flashinfer_sampling
```

### Test Llama 3.2 1B

```bash
python scripts/test_llama32_1b.py
```

## Script Details

### `local_model_serve.py`

Direct model serving with maximum control:

```bash
python scripts/local_model_serve.py \
    --model_path /path/to/model \
    --port 8123 \
    --rope_scaling '{"type":"linear","factor":2.0}' \
    --attention_backend FLASH_ATTN \
    --disable_flashinfer_sampling
```

**Key Features:**
- Auto-detects rope_scaling from model config.json
- Supports custom rope_scaling overrides  
- Real-time colored log streaming
- Multiple attention backend support
- FlashInfer compatibility options

### `run_local.py`

Complete pipeline from serving to inference:

```bash  
python scripts/run_local.py \
    --model_path /path/to/model \
    --model_name my-model \
    --k 10 \
    --chain_type forward \
    --attention_backend XFORMERS
```

**Key Features:**
- Auto-configuration from model files
- Automatic model registration in utils.py
- Server health checking with colored progress
- Real-time inference streaming
- Comprehensive error handling

### `run_llama32_with_fallbacks.py`

Automatic compatibility testing:

```bash
python scripts/run_llama32_with_fallbacks.py /path/to/Llama-3.2-1B
```

**Tries multiple configurations:**
1. Default (FlashInfer enabled)
2. FlashInfer disabled
3. Flash Attention backend
4. xFormers backend  
5. PyTorch SDPA backend

Reports which configuration works and provides the exact command for future use.

### `test_llama32_1b.py`

Automated Llama 3.2 1B setup and testing:

```bash
python scripts/test_llama32_1b.py
```

**Features:**
- Automatic model detection/download guidance
- Requirements checking
- Interactive setup
- Conservative test parameters

## Common Options

### Attention Backends

- `FLASH_ATTN` - Flash Attention (usually fastest)
- `XFORMERS` - xFormers (most compatible)  
- `TORCH_SDPA` - PyTorch native (fallback)

### FlashInfer Options

- `--disable_flashinfer_sampling` - Disable FlashInfer sampling
- Environment variable: `VLLM_ATTENTION_BACKEND`

### Rope Scaling

```bash
# JSON string
--rope_scaling '{"type":"linear","factor":2.0}'

# From file  
--rope_scaling /path/to/config.json

# Auto-detect from model config.json (default)
```

## Troubleshooting

### FlashInfer Compilation Error

```
nvcc fatal: Unknown option '-static-global-template-stub=false'
```

**Solution:** Use attention backend fallbacks:
```bash
python scripts/run_local.py --attention_backend XFORMERS --disable_flashinfer_sampling
```

### Out of Memory

- Reduce `--max_model_len`
- Use smaller models
- Try `--tensor_parallel_size 1`

### Server Won't Start

- Check GPU availability: `nvidia-smi`
- Verify model path exists
- Check port availability: `netstat -an | grep 8123`

## Examples

### Llama 3.2 1B
```bash
python scripts/run_local.py \
    --model_path /path/to/Llama-3.2-1B \
    --attention_backend FLASH_ATTN \
    --max_model_len 8192
```

### Qwen2.5 7B  
```bash
python scripts/run_local.py \
    --model_path /path/to/Qwen2.5-7B \
    --rope_scaling '{"type":"linear","factor":1.5}' \
    --k 20
```

### Multi-GPU Large Model
```bash
python scripts/run_local.py \
    --model_path /path/to/Llama-3.1-70B \
    --tensor_parallel_size 4 \
    --gpu_devices "0,1,2,3" \
    --max_model_len 32768
```