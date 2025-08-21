import os
import json
import argparse
from pathlib import Path

try:
    from setproctitle import setproctitle
    setproctitle("needlechain_local")
except ImportError:
    # setproctitle is optional
    pass

def load_model_config(model_path):
    """Load model configuration to check for rope_scaling settings."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('rope_scaling', None)
    return None

def build_vllm_command(model_path, port=8123, rope_scaling=None, max_model_len=None, 
                      tensor_parallel_size=1, api_key="needlechain", gpu_devices="0"):
    """Build vLLM serving command with proper rope_scaling configuration."""
    
    # Load config from model if rope_scaling not provided
    if rope_scaling is None:
        rope_scaling = load_model_config(model_path)
    
    cmd_parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_devices}",
        "vllm serve", 
        f'"{model_path}"',
        f"--port {port}",
        f"--api-key {api_key}",
        "--dtype auto",
        f"--tensor-parallel-size {tensor_parallel_size}",
        "--max_num_seqs 1"
    ]
    
    # Add rope_scaling if specified
    if rope_scaling:
        if isinstance(rope_scaling, dict):
            rope_scaling_str = json.dumps(rope_scaling).replace('"', '\\"')
            cmd_parts.append(f'--rope-scaling \\"{rope_scaling_str}\\"')
        else:
            cmd_parts.append(f"--rope-scaling {rope_scaling}")
    
    # Add max_model_len if specified
    if max_model_len:
        cmd_parts.append(f"--max_model_len {max_model_len}")
    
    return " \\\n        ".join(cmd_parts)

def get_default_chat_template(model_path):
    """Try to determine appropriate chat template based on model path."""
    model_path_lower = model_path.lower()
    
    if 'llama' in model_path_lower:
        if '3.3' in model_path_lower:
            return './chat_templates/llama3.3_chat_template.jinja'
        elif '3.1' in model_path_lower:
            return './chat_templates/llama3.1_chat_template.jinja'
        else:
            return './chat_templates/llama3.1_chat_template.jinja'  # default to 3.1
    elif 'qwen' in model_path_lower:
        if 'qwen3' in model_path_lower:
            return './chat_templates/qwen3_chat_template.jinja'
        elif 'qwq' in model_path_lower:
            return './chat_templates/QwQ_chat_template.jinja'
        elif 'qwen_long' in model_path_lower or 'qwenlong' in model_path_lower:
            return './chat_templates/qwen_long_chat_template.jinja'
        else:
            return './chat_templates/qwen2.5_chat_template_new.jinja'
    elif 'gemma' in model_path_lower:
        return './chat_templates/gemma3_chat_template.jinja'
    elif 'phi' in model_path_lower:
        return './chat_templates/phi4_chat_template.jinja'
    elif 'mistral' in model_path_lower:
        return './chat_templates/mistral_chat_template.jinja'
    else:
        # Default fallback - try llama format first
        return './chat_templates/llama3.1_chat_template.jinja'

def main():
    parser = argparse.ArgumentParser(description="Serve local models with NeedleChain compatibility")
    parser.add_argument('--model_path', required=True, help='Path to local model directory')
    parser.add_argument('--port', type=int, default=8123, help='Port to serve on')
    parser.add_argument('--api_key', default='needlechain', help='API key for the server')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--max_model_len', type=int, help='Maximum model context length')
    parser.add_argument('--gpu_devices', default='0', help='CUDA device IDs (comma-separated)')
    parser.add_argument('--rope_scaling', help='Rope scaling configuration (JSON string or file path)')
    parser.add_argument('--chat_template', help='Path to chat template file')
    parser.add_argument('--framework', default='vllm', choices=['vllm'], help='Serving framework')
    parser.add_argument('--dry_run', action='store_true', help='Print command without executing')
    
    args = parser.parse_args()
    
    # Handle rope_scaling argument
    rope_scaling = None
    if args.rope_scaling:
        if os.path.isfile(args.rope_scaling):
            # Load from file
            with open(args.rope_scaling, 'r') as f:
                rope_scaling = json.load(f)
        else:
            # Parse as JSON string
            try:
                rope_scaling = json.loads(args.rope_scaling)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in rope_scaling: {args.rope_scaling}")
                return
    
    # Determine chat template
    chat_template = args.chat_template
    if not chat_template:
        chat_template = get_default_chat_template(args.model_path)
        print(f"Auto-detected chat template: {chat_template}")
    
    if args.framework == 'vllm':
        cmd = build_vllm_command(
            model_path=args.model_path,
            port=args.port,
            rope_scaling=rope_scaling,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            api_key=args.api_key,
            gpu_devices=args.gpu_devices
        )
        
        if chat_template and os.path.exists(chat_template):
            cmd += f" \\\n        --chat-template {chat_template}"
        
        print("="*60)
        print("Starting local model server with command:")
        print("="*60)
        print(cmd)
        print("="*60)
        
        if args.dry_run:
            print("Dry run mode - command not executed")
            return
        
        # Execute the command
        os.system(cmd)
    
    else:
        print(f"Framework {args.framework} not yet supported")

if __name__ == '__main__':
    main()