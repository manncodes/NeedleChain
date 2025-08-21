#!/usr/bin/env python3
"""
Automatic fallback script for Llama 3.2 1B with FlashInfer compatibility issues.
Tries different attention backends until one works.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_with_backend(model_path, backend=None, disable_flashinfer=False, timeout=300):
    """Try to run the model with a specific backend configuration."""
    
    cmd = [
        sys.executable, 'run_local.py',
        '--model_path', model_path,
        '--model_name', 'llama32-1b-fallback',
        '--k', '5',
        '--chain_type', 'forward',
        '--question_type', 'single',
        '--max_model_len', '8192',  # Conservative for compatibility
    ]
    
    if backend:
        cmd.extend(['--attention_backend', backend])
        print(f"🔄 Trying with attention backend: {backend}")
    else:
        print("🔄 Trying with default settings")
    
    if disable_flashinfer:
        cmd.append('--disable_flashinfer_sampling')
        print("🔄 With FlashInfer sampling disabled")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        timeout_min = timeout // 60
        print(f"❌ Timed out after {timeout_min} minutes")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automatic fallback script for Llama 3.2 1B with FlashInfer compatibility issues"
    )
    parser.add_argument('model_path', help='Path to the model directory')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per attempt in seconds')
    
    args = parser.parse_args()
    model_path = args.model_path
    
    if not Path(model_path).exists():
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)
    
    print("🚀 Llama 3.2 1B Automatic Fallback Runner")
    print("="*50)
    print(f"Model: {model_path}")
    print()
    
    # List of fallback configurations to try
    fallbacks = [
        # Try default first (might work if FlashInfer is fixed)
        {"name": "Default (FlashInfer)", "backend": None, "disable_flashinfer": False},
        
        # Try with FlashInfer disabled
        {"name": "FlashInfer Disabled", "backend": None, "disable_flashinfer": True},
        
        # Try different attention backends
        {"name": "Flash Attention", "backend": "FLASH_ATTN", "disable_flashinfer": True},
        {"name": "xFormers", "backend": "XFORMERS", "disable_flashinfer": True},
        {"name": "PyTorch SDPA", "backend": "TORCH_SDPA", "disable_flashinfer": True},
        
        # Last resort - Flash Attention without FlashInfer sampling
        {"name": "Flash Attention (safe mode)", "backend": "FLASH_ATTN", "disable_flashinfer": True},
    ]
    
    for i, config in enumerate(fallbacks, 1):
        print(f"\n📋 Attempt {i}/{len(fallbacks)}: {config['name']}")
        print("-" * 40)
        
        success = run_with_backend(
            model_path, 
            backend=config['backend'],
            disable_flashinfer=config['disable_flashinfer'],
            timeout=args.timeout
        )
        
        if success:
            print(f"\n🎉 SUCCESS! Configuration that worked:")
            print(f"   - Attention Backend: {config['backend'] or 'Default'}")
            print(f"   - FlashInfer Sampling: {'Disabled' if config['disable_flashinfer'] else 'Enabled'}")
            print(f"\n💡 To use this configuration again, run:")
            
            cmd_suggestion = f"python run_local.py --model_path {model_path}"
            if config['backend']:
                cmd_suggestion += f" --attention_backend {config['backend']}"
            if config['disable_flashinfer']:
                cmd_suggestion += " --disable_flashinfer_sampling"
            
            print(f"   {cmd_suggestion}")
            return 0
        else:
            print(f"❌ Failed with {config['name']}")
            time.sleep(2)  # Brief pause between attempts
    
    print("\n💥 All fallback configurations failed!")
    print("This might indicate:")
    print("  - CUDA/driver compatibility issues")
    print("  - Insufficient GPU memory") 
    print("  - Model files are corrupted")
    print("  - vLLM version incompatibility")
    
    return 1

if __name__ == '__main__':
    sys.exit(main())