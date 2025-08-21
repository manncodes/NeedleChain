# NeedleChain Scripts

Convenient shell scripts for running NeedleChain evaluations with different configurations, following the proper 2-stage approach: host model + run evaluation.

## 🎯 Two-Stage Evaluation (Recommended)

### Stage 1: Host Model
```bash
# Start model server (keeps running)
bash scripts/stage1_host_model.sh [model_path] [attention_backend]

# Examples:
bash scripts/stage1_host_model.sh /exp/model/Huggingface/meta-llama/Llama-3.2-1B XFORMERS
bash scripts/stage1_host_model.sh /path/to/model FLASH_ATTN
```

### Stage 2: Run Evaluation  
```bash
# Run evaluation against hosted model (in another terminal)
bash scripts/stage2_run_evaluation.sh [model_name] [k] [chain_type] [question_type]

# Examples:
bash scripts/stage2_run_evaluation.sh Llama-3.2-1B 5 forward single
bash scripts/stage2_run_evaluation.sh Llama-3.2-1B 10 backward total
```

## 🚀 One-Command Solutions

### Full Pipeline
```bash
# Combines both stages automatically
bash scripts/full_evaluation.sh [model_path] [k] [chain_type] [question_type] [backend]

# Examples:
bash scripts/full_evaluation.sh /exp/model/Huggingface/meta-llama/Llama-3.2-1B 5 forward single XFORMERS
bash scripts/full_evaluation.sh /path/to/model 10 chaotic total FLASH_ATTN
```

### Batch Evaluation
```bash
# Run multiple k/chain/question combinations
bash scripts/batch_evaluation.sh [model_path] [attention_backend]

# Example:
bash scripts/batch_evaluation.sh /exp/model/Huggingface/meta-llama/Llama-3.2-1B XFORMERS
```

## 🚀 Legacy Quick Start Scripts

### Basic Evaluation
```bash
# Default settings (may fail with FlashInfer issues)
bash scripts/llama3_2_1b_eval.sh

# With custom model path
bash scripts/llama3_2_1b_eval.sh /path/to/your/Llama-3.2-1B
```

### FlashInfer Compatibility Scripts

**Most Recommended - Auto Fallback**:
```bash
bash scripts/llama3_2_1b_auto_fallback.sh
```
Automatically tries different configurations until one works.

**Flash Attention Backend**:
```bash
bash scripts/llama3_2_1b_flash_attn.sh
```

**xFormers Backend (Most Compatible)**:
```bash
bash scripts/llama3_2_1b_xformers.sh
```

**PyTorch SDPA Backend**:
```bash
bash scripts/llama3_2_1b_torch_sdpa.sh
```

### Server Only Scripts

**Start server without inference**:
```bash
bash scripts/serve_only_flash_attn.sh
```
Server runs on http://localhost:8123, press Ctrl+C to stop.

### Testing Scripts

**Test different needle counts (k values)**:
```bash
bash scripts/test_different_ks.sh
bash scripts/test_different_ks.sh /path/to/model XFORMERS
```

## 📁 File Descriptions

### Two-Stage Scripts (Recommended)
| Script | Purpose | Usage |
|--------|---------|--------|
| `stage1_host_model.sh` | Start model server | `bash scripts/stage1_host_model.sh [model_path] [backend]` |
| `stage2_run_evaluation.sh` | Run evaluation | `bash scripts/stage2_run_evaluation.sh [model] [k] [chain] [question]` |
| `full_evaluation.sh` | Complete pipeline | `bash scripts/full_evaluation.sh [model_path] [k] [chain] [question]` |
| `batch_evaluation.sh` | Multiple configurations | `bash scripts/batch_evaluation.sh [model_path] [backend]` |

### Legacy Scripts (FlashInfer compatibility)
| Script | Purpose | Backend | FlashInfer |
|--------|---------|---------|------------|
| `llama3_2_1b_eval.sh` | Basic evaluation | Default | Enabled |
| `llama3_2_1b_flash_attn.sh` | Flash Attention | FLASH_ATTN | Disabled |
| `llama3_2_1b_xformers.sh` | xFormers (safest) | XFORMERS | Disabled |
| `llama3_2_1b_torch_sdpa.sh` | PyTorch SDPA | TORCH_SDPA | Disabled |
| `llama3_2_1b_auto_fallback.sh` | Auto-retry all | Multiple | Smart |
| `serve_only_flash_attn.sh` | Server only | FLASH_ATTN | Disabled |
| `test_different_ks.sh` | Multi-k testing | Configurable | Disabled |

## 🛠️ Troubleshooting

**FlashInfer compilation errors**:
- Try `llama3_2_1b_auto_fallback.sh` first
- If that fails, try `llama3_2_1b_xformers.sh`

**Out of memory errors**:
- Use smaller k values
- Reduce max_model_len in the scripts

**CUDA errors**:
- Check GPU availability: `nvidia-smi`
- Try different attention backends

## 📝 Customization

All scripts accept model path as first argument:
```bash
bash scripts/llama3_2_1b_flash_attn.sh /your/custom/model/path
```

To modify other parameters, edit the scripts directly or use the Python scripts:
```bash
python3 run_local.py --help
python3 local_model_serve.py --help
```

## 🎯 Results

All results are saved to `./results/` directory with descriptive filenames.