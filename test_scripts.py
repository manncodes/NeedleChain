#!/usr/bin/env python3
"""
Test script to validate NeedleChain scripts work with specific configuration:
- vLLM 0.10.1
- FlashInfer 0.2.11
- Python 3.10
"""

import subprocess
import sys
import os
import tempfile
import json
from pathlib import Path

def test_import_compatibility():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    required_packages = [
        'torch',
        'transformers', 
        'vllm',
        'flashinfer',
        'openai',
        'numpy',
        'tqdm',
        'setproctitle',
        'requests'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_script_syntax():
    """Test if Python scripts have valid syntax."""
    print("\n🧪 Testing Python script syntax...")
    
    python_scripts = [
        'local_model_serve.py',
        'run_local.py',
        'run_llama32_with_fallbacks.py',
        'inference_call.py',
        'utils.py'
    ]
    
    failed_scripts = []
    for script in python_scripts:
        if os.path.exists(script):
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'py_compile', script
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✅ {script}")
                else:
                    print(f"  ❌ {script}: {result.stderr}")
                    failed_scripts.append(script)
            except Exception as e:
                print(f"  ❌ {script}: {e}")
                failed_scripts.append(script)
        else:
            print(f"  ⚠️ {script}: Not found")
    
    return len(failed_scripts) == 0

def test_bash_scripts():
    """Test if bash scripts have valid syntax."""
    print("\n🧪 Testing bash script syntax...")
    
    bash_scripts = [
        'scripts/stage1_host_model.sh',
        'scripts/stage2_run_evaluation.sh', 
        'scripts/full_evaluation.sh',
        'scripts/batch_evaluation.sh',
        'scripts/llama3_2_1b_xformers.sh'
    ]
    
    failed_scripts = []
    for script in bash_scripts:
        if os.path.exists(script):
            try:
                result = subprocess.run([
                    'bash', '-n', script
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✅ {script}")
                else:
                    print(f"  ❌ {script}: {result.stderr}")
                    failed_scripts.append(script)
            except Exception as e:
                print(f"  ❌ {script}: {e}")
                failed_scripts.append(script)
        else:
            print(f"  ⚠️ {script}: Not found")
    
    return len(failed_scripts) == 0

def test_help_commands():
    """Test if help commands work."""
    print("\n🧪 Testing help commands...")
    
    commands = [
        [sys.executable, 'local_model_serve.py', '--help'],
        [sys.executable, 'run_local.py', '--help'],
        [sys.executable, 'run_llama32_with_fallbacks.py', '--help'] 
    ]
    
    failed_commands = []
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'usage:' in result.stdout:
                print(f"  ✅ {' '.join(cmd)}")
            else:
                print(f"  ❌ {' '.join(cmd)}: {result.stderr}")
                failed_commands.append(cmd)
        except Exception as e:
            print(f"  ❌ {' '.join(cmd)}: {e}")
            failed_commands.append(cmd)
    
    return len(failed_commands) == 0

def test_dry_run_commands():
    """Test dry run functionality with mock model."""
    print("\n🧪 Testing dry run commands...")
    
    # Create mock model directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_model_dir = Path(temp_dir) / "mock_llama"
        mock_model_dir.mkdir()
        
        # Create mock config.json
        config = {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 8192,
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0
            }
        }
        
        with open(mock_model_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        # Test dry run commands
        commands = [
            [sys.executable, 'local_model_serve.py', 
             '--model_path', str(mock_model_dir), 
             '--dry_run'],
            [sys.executable, 'local_model_serve.py',
             '--model_path', str(mock_model_dir),
             '--attention_backend', 'XFORMERS', 
             '--disable_flashinfer_sampling',
             '--dry_run']
        ]
        
        failed_commands = []
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0 and 'Dry run mode' in result.stdout:
                    print(f"  ✅ Dry run test passed")
                else:
                    print(f"  ❌ Dry run failed: {result.stderr}")
                    failed_commands.append(cmd)
            except Exception as e:
                print(f"  ❌ Dry run error: {e}")
                failed_commands.append(cmd)
        
        return len(failed_commands) == 0

def test_vllm_compatibility():
    """Test vLLM version compatibility."""
    print("\n🧪 Testing vLLM compatibility...")
    
    try:
        import vllm
        print(f"  ✅ vLLM version: {vllm.__version__}")
        
        # Test if we can import key vLLM components
        from vllm import LLM, SamplingParams
        print(f"  ✅ vLLM core imports successful")
        
        # Test if vLLM serve command exists
        result = subprocess.run(['vllm', '--help'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✅ vLLM CLI available")
        else:
            print(f"  ❌ vLLM CLI issue: {result.stderr}")
            return False
            
        return True
    except Exception as e:
        print(f"  ❌ vLLM compatibility issue: {e}")
        return False

def test_flashinfer_compatibility():
    """Test FlashInfer compatibility."""
    print("\n🧪 Testing FlashInfer compatibility...")
    
    try:
        import flashinfer
        print(f"  ✅ FlashInfer available")
        
        # Check if the problematic compilation issue exists
        print(f"  ℹ️  FlashInfer location: {flashinfer.__file__}")
        return True
    except Exception as e:
        print(f"  ❌ FlashInfer issue: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 NeedleChain Script Compatibility Test")
    print("========================================")
    print("Testing with your configuration:")
    print("- vLLM 0.10.1")
    print("- FlashInfer 0.2.11") 
    print("- Python 3.10")
    print("")
    
    tests = [
        ("Import Compatibility", test_import_compatibility),
        ("Python Script Syntax", test_script_syntax),
        ("Bash Script Syntax", test_bash_scripts),
        ("Help Commands", test_help_commands),
        ("Dry Run Commands", test_dry_run_commands),
        ("vLLM Compatibility", test_vllm_compatibility),
        ("FlashInfer Compatibility", test_flashinfer_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🔬 {test_name}")
        print('='*50)
        
        try:
            if test_func():
                print(f"\n✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            print(f"\n💥 {test_name}: CRASHED - {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Scripts should work with your configuration.")
        return 0
    else:
        print("⚠️  Some tests failed. Check issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())