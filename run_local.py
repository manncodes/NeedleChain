#!/usr/bin/env python3
"""
Convenient script for running NeedleChain with local models.
Handles model serving and inference automatically.
"""

import os
import json
import time
import argparse
import subprocess
import signal
import sys
from pathlib import Path
from multiprocessing import Process

import requests
from openai import OpenAI

def check_server_health(port=8123, max_retries=30, retry_delay=10):
    """Check if the model server is healthy and ready."""
    url = f"http://localhost:{port}/health"
    
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ Server is healthy and ready on port {port}")
                return True
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass
        
        print(f"Waiting for server to start... ({i+1}/{max_retries})")
        time.sleep(retry_delay)
    
    print(f"✗ Server failed to start after {max_retries * retry_delay} seconds")
    return False

def start_model_server(model_path, port=8123, rope_scaling=None, max_model_len=None, 
                      tensor_parallel_size=1, gpu_devices="0", chat_template=None):
    """Start the model server in a subprocess."""
    
    cmd = [
        sys.executable, 'local_model_serve.py',
        '--model_path', model_path,
        '--port', str(port),
        '--tensor_parallel_size', str(tensor_parallel_size),
        '--gpu_devices', gpu_devices
    ]
    
    if rope_scaling:
        cmd.extend(['--rope_scaling', json.dumps(rope_scaling)])
    
    if max_model_len:
        cmd.extend(['--max_model_len', str(max_model_len)])
    
    if chat_template:
        cmd.extend(['--chat_template', chat_template])
    
    print(f"Starting model server: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def run_inference(model_name, chain_type='forward', question_type='single', 
                 k=5, val=1600, results_dir='./results', output_name=None):
    """Run the inference using the running model server."""
    
    if not output_name:
        output_name = f"{model_name}_{chain_type}_{question_type}_k{k}"
    
    cmd = [
        sys.executable, 'inference_call.py',
        '--model_name', model_name,
        '--output_name', output_name,
        '--chain_type', chain_type,
        '--question_type', question_type,
        '--k', str(k),
        '--val', str(val),
        '--results_dir', results_dir
    ]
    
    print(f"Running inference: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Inference completed successfully")
        print(f"Results saved to: {results_dir}/{output_name}.jsonl")
    else:
        print("✗ Inference failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def load_model_config(model_path):
    """Load model configuration."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def add_local_model_to_utils(model_name, model_path, chat_template=None):
    """Add the local model to utils.py for compatibility."""
    
    # Read current utils.py
    with open('utils.py', 'r') as f:
        content = f.read()
    
    # Check if model already exists
    if f"'{model_name}'" in content:
        print(f"Model {model_name} already exists in utils.py")
        return
    
    # Add to model_arg_dict
    model_dict_line = f"    '{model_name}': '{model_path}',"
    
    # Find the end of model_arg_dict
    lines = content.split('\n')
    new_lines = []
    in_model_dict = False
    
    for line in lines:
        if 'model_arg_dict = {' in line:
            in_model_dict = True
            new_lines.append(line)
        elif in_model_dict and line.strip() == '}':
            new_lines.append(model_dict_line)
            new_lines.append(line)
            in_model_dict = False
        else:
            new_lines.append(line)
    
    # Add to chat_template_dict if template provided
    if chat_template:
        template_dict_line = f"    '{model_name}': '{chat_template}',"
        in_template_dict = False
        final_lines = []
        
        for line in new_lines:
            if 'chat_template_dict = {' in line:
                in_template_dict = True
                final_lines.append(line)
            elif in_template_dict and line.strip() == '}':
                final_lines.append(template_dict_line)
                final_lines.append(line)
                in_template_dict = False
            else:
                final_lines.append(line)
        new_lines = final_lines
    
    # Write back to utils.py
    with open('utils.py', 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"✓ Added {model_name} to utils.py")

def main():
    parser = argparse.ArgumentParser(description="Run NeedleChain with local models")
    parser.add_argument('--model_path', required=True, help='Path to local model directory')
    parser.add_argument('--model_name', help='Name for the model (defaults to model directory name)')
    parser.add_argument('--port', type=int, default=8123, help='Port to serve on')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, help='Maximum model context length')
    parser.add_argument('--gpu_devices', default='0', help='CUDA device IDs (comma-separated)')
    parser.add_argument('--rope_scaling', help='Rope scaling configuration (JSON string)')
    parser.add_argument('--chat_template', help='Path to chat template file')
    
    # Inference parameters
    parser.add_argument('--chain_type', default='forward', 
                       choices=['forward', 'parallel', 'backward', 'chaotic'])
    parser.add_argument('--question_type', default='single', choices=['single', 'total'])
    parser.add_argument('--k', type=int, default=5, help='Number of needles')
    parser.add_argument('--val', type=int, default=1600, choices=[160, 1600, 16000])
    parser.add_argument('--results_dir', default='./results', help='Results directory')
    parser.add_argument('--output_name', help='Output filename (auto-generated if not provided)')
    
    # Control options
    parser.add_argument('--serve_only', action='store_true', 
                       help='Only start server, do not run inference')
    parser.add_argument('--no_auto_config', action='store_true', 
                       help='Do not auto-detect model configuration')
    
    args = parser.parse_args()
    
    # Determine model name
    if not args.model_name:
        args.model_name = Path(args.model_path).name.replace('/', '_')
    
    print(f"Setting up local model: {args.model_name}")
    print(f"Model path: {args.model_path}")
    
    # Load model config for auto-configuration
    rope_scaling = None
    if not args.no_auto_config:
        config = load_model_config(args.model_path)
        if 'rope_scaling' in config and not args.rope_scaling:
            rope_scaling = config['rope_scaling']
            print(f"Auto-detected rope_scaling from model config: {rope_scaling}")
        
        if 'max_position_embeddings' in config and not args.max_model_len:
            args.max_model_len = config['max_position_embeddings']
            print(f"Auto-detected max_model_len: {args.max_model_len}")
    
    # Parse rope_scaling if provided
    if args.rope_scaling:
        try:
            rope_scaling = json.loads(args.rope_scaling)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in rope_scaling: {args.rope_scaling}")
            return
    
    # Add model to utils.py
    add_local_model_to_utils(args.model_name, args.model_path, args.chat_template)
    
    server_process = None
    try:
        # Start model server
        print("\n" + "="*60)
        print("STARTING MODEL SERVER")
        print("="*60)
        
        server_process = start_model_server(
            model_path=args.model_path,
            port=args.port,
            rope_scaling=rope_scaling,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_devices=args.gpu_devices,
            chat_template=args.chat_template
        )
        
        # Wait for server to be ready
        if not check_server_health(args.port):
            print("Failed to start model server")
            return
        
        if args.serve_only:
            print("Server started. Use Ctrl+C to stop.")
            try:
                server_process.wait()
            except KeyboardInterrupt:
                print("\nStopping server...")
            return
        
        # Run inference
        print("\n" + "="*60)
        print("RUNNING INFERENCE")
        print("="*60)
        
        success = run_inference(
            model_name=args.model_name,
            chain_type=args.chain_type,
            question_type=args.question_type,
            k=args.k,
            val=args.val,
            results_dir=args.results_dir,
            output_name=args.output_name
        )
        
        if success:
            print("\n✓ Complete! Results are ready.")
        else:
            print("\n✗ Inference failed. Check logs above.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        if server_process:
            print("Stopping model server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print("Server stopped.")

if __name__ == '__main__':
    main()