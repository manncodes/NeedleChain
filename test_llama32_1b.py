#!/usr/bin/env python3
"""
Test script for Llama3.2 1B model with NeedleChain.
This script demonstrates how to use the local model serving functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def download_model_if_needed(model_name="unsloth/Llama-3.2-1B-Instruct"):
    """Download Llama3.2 1B model if not already available."""
    
    # Check common model locations
    possible_paths = [
        f"./models/{model_name.split('/')[-1]}",
        f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots",
        f"/models/{model_name.split('/')[-1]}"
    ]
    
    for path in possible_paths:
        expanded_path = Path(path).expanduser()
        if expanded_path.exists():
            print(f"Found model at: {expanded_path}")
            return str(expanded_path)
    
    print(f"Model not found locally. You can download it using:")
    print(f"  huggingface-cli download {model_name} --local-dir ./models/Llama-3.2-1B-Instruct")
    print(f"  or")
    print(f"  git clone https://huggingface.co/{model_name} ./models/Llama-3.2-1B-Instruct")
    
    return None

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['vllm', 'openai', 'transformers', 'torch']
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def run_test(model_path, use_rope_scaling=True):
    """Run a test with the Llama3.2 1B model."""
    
    # Basic rope scaling configuration for Llama3.2
    rope_scaling_config = {
        "type": "linear",
        "factor": 1.0
    } if use_rope_scaling else None
    
    print("="*60)
    print("TESTING LLAMA3.2 1B WITH NEEDLECHAIN")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Rope scaling: {rope_scaling_config}")
    print()
    
    # Prepare arguments for run_local.py
    cmd = [
        sys.executable, 'run_local.py',
        '--model_path', model_path,
        '--model_name', 'llama32-1b',
        '--tensor_parallel_size', '1',
        '--max_model_len', '8192',  # Conservative length for 1B model
        '--k', '5',  # Start with smaller test
        '--chain_type', 'forward',
        '--question_type', 'single',
        '--results_dir', './test_results'
    ]
    
    if rope_scaling_config:
        import json
        cmd.extend(['--rope_scaling', json.dumps(rope_scaling_config)])
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Test failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
        return False

def main():
    print("Llama3.2 1B Test Setup")
    print("=" * 30)
    
    # Check requirements
    if not check_requirements():
        print("Please install missing packages first.")
        return 1
    
    # Find or prompt for model
    model_path = download_model_if_needed()
    if not model_path:
        print("\nPlease provide model path manually:")
        model_path = input("Enter path to Llama3.2 1B model directory: ").strip()
        if not model_path or not Path(model_path).exists():
            print("Invalid model path provided.")
            return 1
    
    # Confirm test parameters
    print(f"\nModel path: {model_path}")
    response = input("Proceed with test? [Y/n]: ").strip().lower()
    if response and response[0] == 'n':
        print("Test cancelled.")
        return 0
    
    # Run the test
    success = run_test(model_path)
    
    if success:
        print("\n🎉 Test completed! Check ./test_results/ for output files.")
    else:
        print("\n💥 Test failed. Check error messages above.")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())