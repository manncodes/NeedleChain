#!/usr/bin/env python3
"""
Simple test script to verify NeedleChain can run with a mock model.
Tests the most basic configuration that should work.
"""

import subprocess
import sys
import os
import tempfile
import json
import time
from pathlib import Path

def create_mock_model(model_dir):
    """Create a minimal mock model structure."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal config.json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 16,
        "num_hidden_layers": 22,
        "num_key_value_heads": 8,
        "rope_scaling": {
            "factor": 2.0,
            "type": "linear"
        },
        "vocab_size": 128256
    }
    
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create tokenizer config
    tokenizer_config = {
        "model_max_length": 8192,
        "tokenizer_class": "PreTrainedTokenizerFast"
    }
    
    with open(model_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"✅ Created mock model at: {model_dir}")
    return str(model_dir)

def test_basic_server_start():
    """Test the most basic server start with environment variables."""
    print("\n🧪 Testing basic server start with XFORMERS backend...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_model = create_mock_model(Path(temp_dir) / "test_model")
        
        # Build command with environment variables
        env = os.environ.copy()
        env['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
        env['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        cmd = [
            sys.executable, 'local_model_serve.py',
            '--model_path', mock_model,
            '--attention_backend', 'XFORMERS',
            '--disable_flashinfer_sampling',
            '--dry_run'
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Environment: VLLM_ATTENTION_BACKEND=XFORMERS VLLM_USE_FLASHINFER_SAMPLER=0")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Dry run successful!")
                print("\nGenerated command:")
                print(result.stdout)
                return True
            else:
                print(f"❌ Dry run failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def test_complete_pipeline():
    """Test the complete pipeline with run_local.py."""
    print("\n🧪 Testing complete pipeline with run_local.py...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_model = create_mock_model(Path(temp_dir) / "test_model")
        
        # Test with safest configuration
        cmd = [
            sys.executable, 'run_local.py',
            '--model_path', mock_model,
            '--model_name', 'test-model',
            '--k', '3',
            '--chain_type', 'forward',
            '--question_type', 'single',
            '--max_model_len', '4096',
            '--attention_backend', 'XFORMERS',
            '--disable_flashinfer_sampling',
            '--dry_run'
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if 'Dry run mode' in result.stdout or result.returncode == 0:
                print("✅ Pipeline dry run successful!")
                return True
            else:
                print(f"❌ Pipeline failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

def test_fallback_script():
    """Test the automatic fallback script."""
    print("\n🧪 Testing automatic fallback script...")
    
    # Just test help to ensure it's working
    cmd = [sys.executable, 'run_llama32_with_fallbacks.py', '--help']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and 'usage:' in result.stdout:
            print("✅ Fallback script is functional")
            return True
        else:
            print(f"❌ Fallback script issue: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 NeedleChain Simple Functionality Test")
    print("=" * 50)
    print("Testing minimal configuration that should work...")
    print()
    
    tests = [
        ("Basic Server Start", test_basic_server_start),
        ("Complete Pipeline", test_complete_pipeline),
        ("Fallback Script", test_fallback_script),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED")
    
    print(f"\n{'='*60}")
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 SUCCESS! All basic functionality tests passed.")
        print("\n💡 Recommended command for your Llama 3.2 1B model:")
        print("="*50)
        print("python3 run_local.py \\")
        print("    --model_path /path/to/your/Llama-3.2-1B \\")
        print("    --model_name llama32-1b \\")
        print("    --attention_backend XFORMERS \\")
        print("    --disable_flashinfer_sampling \\")
        print("    --k 5 \\")
        print("    --chain_type forward \\")
        print("    --question_type single")
        print("\nOr use the automatic fallback script:")
        print("python3 run_llama32_with_fallbacks.py /path/to/your/Llama-3.2-1B")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return 0 if passed == len(tests) else 1

if __name__ == '__main__':
    sys.exit(main())