# NeedleChain Scripts

Convenient shell scripts for running NeedleChain with different configurations, especially for FlashInfer compatibility issues.

## 🚀 Quick Start Scripts

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