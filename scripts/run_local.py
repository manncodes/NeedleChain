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
import threading
from pathlib import Path
from multiprocessing import Process

try:
    import requests
except ImportError:
    print("Warning: requests not available, server health checking disabled")
    requests = None

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai not available")
    OpenAI = None

# Import color utilities from local_model_serve
try:
    from .local_model_serve import Colors, colorize_vllm_log, stream_process_output
except ImportError:
    try:
        from local_model_serve import Colors, colorize_vllm_log, stream_process_output
    except ImportError:
        # Fallback if local_model_serve not available
        class Colors:
            RESET = '\033[0m'
            BRIGHT_GREEN = '\033[92m'
            BRIGHT_RED = '\033[91m'
            BRIGHT_YELLOW = '\033[93m'
            BRIGHT_BLUE = '\033[94m'
            BG_BLUE = '\033[44m'
            BG_GREEN = '\033[42m'
            BG_RED = '\033[41m'
            WHITE = '\033[97m'
    
        def colorize_vllm_log(line):
            return line
        
        def stream_process_output(process, prefix="[vLLM]"):
            return None, None

def check_server_health(port=8123, max_retries=30, retry_delay=2):
    """Check if the model server is healthy and ready."""
    if not requests:
        print(f"{Colors.BRIGHT_YELLOW}Warning: requests not available, skipping health check{Colors.RESET}")
        time.sleep(10)  # Give some time for server to start
        return True
    
    url = f"http://localhost:{port}/health"
    
    print(f"\n{Colors.BRIGHT_BLUE}Checking server health...{Colors.RESET}")
    
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"{Colors.BRIGHT_GREEN}✓ Server is healthy and ready on port {port}{Colors.RESET}")
                return True
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass
        
        # Show progress with dots
        dots = "." * (i % 4)
        print(f"{Colors.BRIGHT_BLUE}Waiting for server to be ready{dots:<3} ({i+1}/{max_retries}){Colors.RESET}", end='\r')
        time.sleep(retry_delay)
    
    print(f"\n{Colors.BRIGHT_RED}✗ Server failed to start after {max_retries * retry_delay} seconds{Colors.RESET}")
    return False

def start_model_server(model_path, port=8123, rope_scaling=None, max_model_len=None, 
                      tensor_parallel_size=1, gpu_devices="0", chat_template=None,
                      attention_backend=None, disable_flashinfer_sampling=False):
    """Start the model server in a subprocess with colored output streaming."""
    
    cmd = [
        sys.executable, 'scripts/local_model_serve.py',
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
    
    if attention_backend:
        cmd.extend(['--attention_backend', attention_backend])
    
    if disable_flashinfer_sampling:
        cmd.extend(['--disable_flashinfer_sampling'])
    
    print(f"{Colors.BRIGHT_BLUE}Starting model server with command:{Colors.RESET}")
    print(f"{Colors.WHITE}{' '.join(cmd)}{Colors.RESET}\n")
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=False
    )
    
    # Start streaming output in background threads
    if stream_process_output != None:  # Check if function is available
        stream_process_output(process, prefix="[Server]")
    
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
    
    print(f"{Colors.BRIGHT_BLUE}Running inference:{Colors.RESET}")
    print(f"{Colors.WHITE}{' '.join(cmd)}{Colors.RESET}\n")
    
    # Run inference with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=False
    )
    
    # Stream the output with colors
    if stream_process_output != None:
        stdout_thread, stderr_thread = stream_process_output(process, prefix="[Inference]")
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for threads to finish
        if stdout_thread:
            stdout_thread.join(timeout=1)
        if stderr_thread:
            stderr_thread.join(timeout=1)
    else:
        return_code = process.wait()
    
    if return_code == 0:
        print(f"{Colors.BRIGHT_GREEN}✓ Inference completed successfully{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}Results saved to: {results_dir}/{output_name}.jsonl{Colors.RESET}")
    else:
        print(f"{Colors.BRIGHT_RED}✗ Inference failed (exit code: {return_code}){Colors.RESET}")
    
    return return_code == 0

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
    parser.add_argument('--attention_backend', choices=['FLASH_ATTN', 'XFORMERS', 'TORCH_SDPA'], 
                       help='vLLM attention backend (helps with FlashInfer issues)')
    parser.add_argument('--disable_flashinfer_sampling', action='store_true',
                       help='Disable FlashInfer sampling (use for CUDA compatibility issues)')
    
    args = parser.parse_args()
    
    # Determine model name
    if not args.model_name:
        args.model_name = Path(args.model_path).name.replace('/', '_')
    
    print(f"{Colors.BRIGHT_BLUE}Setting up local model: {Colors.BRIGHT_WHITE}{args.model_name}{Colors.RESET}")
    print(f"{Colors.BRIGHT_BLUE}Model path: {Colors.WHITE}{args.model_path}{Colors.RESET}")
    
    # Load model config for auto-configuration
    rope_scaling = None
    if not args.no_auto_config:
        config = load_model_config(args.model_path)
        if 'rope_scaling' in config and not args.rope_scaling:
            rope_scaling = config['rope_scaling']
            print(f"{Colors.BRIGHT_GREEN}Auto-detected rope_scaling from model config: {rope_scaling}{Colors.RESET}")
        
        if 'max_position_embeddings' in config and not args.max_model_len:
            args.max_model_len = config['max_position_embeddings']
            print(f"{Colors.BRIGHT_GREEN}Auto-detected max_model_len: {args.max_model_len}{Colors.RESET}")
    
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
        print(f"\n{Colors.BG_BLUE}{Colors.WHITE} STARTING MODEL SERVER {Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}{'='*60}{Colors.RESET}")
        
        server_process = start_model_server(
            model_path=args.model_path,
            port=args.port,
            rope_scaling=rope_scaling,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_devices=args.gpu_devices,
            chat_template=args.chat_template,
            attention_backend=args.attention_backend,
            disable_flashinfer_sampling=args.disable_flashinfer_sampling
        )
        
        # Wait for server to be ready
        if not check_server_health(args.port):
            print("Failed to start model server")
            return
        
        if args.serve_only:
            print(f"{Colors.BRIGHT_GREEN}Server started. Use Ctrl+C to stop.{Colors.RESET}")
            try:
                server_process.wait()
            except KeyboardInterrupt:
                print(f"\n{Colors.BRIGHT_YELLOW}Stopping server...{Colors.RESET}")
            return
        
        # Run inference
        print(f"\n{Colors.BG_GREEN}{Colors.WHITE} RUNNING INFERENCE {Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{'='*60}{Colors.RESET}")
        
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
            print(f"\n{Colors.BG_GREEN}{Colors.WHITE} COMPLETE! RESULTS ARE READY {Colors.RESET}")
        else:
            print(f"\n{Colors.BG_RED}{Colors.WHITE} INFERENCE FAILED - CHECK LOGS ABOVE {Colors.RESET}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.BRIGHT_YELLOW}Interrupted by user{Colors.RESET}")
    
    finally:
        if server_process:
            print(f"{Colors.BRIGHT_YELLOW}Stopping model server...{Colors.RESET}")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print(f"{Colors.BRIGHT_GREEN}Server stopped.{Colors.RESET}")

if __name__ == '__main__':
    main()