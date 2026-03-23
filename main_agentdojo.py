"""
AgentDojo Benchmark Evaluation with PIArena Defenses

This script evaluates PIArena defenses on the AgentDojo benchmark for
prompt injection attacks in tool-integrated LLM agents.

Usage:
    # Benign utility (no attack) with GPT-4o (OpenAI API)
    python main_agentdojo.py --model gpt-4o-2024-05-13 --attack none

    # With Azure OpenAI (uses configs/azure_configs/gpt-4o.yaml)
    python main_agentdojo.py --model azure/gpt-4o --attack tool_knowledge --defense datafilter

    # With HuggingFace model (starts vLLM server automatically)
    python main_agentdojo.py --model meta-llama/Llama-3.1-8B-Instruct --attack tool_knowledge --defense datafilter

    # Specify tensor parallel size for large models
    python main_agentdojo.py --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --attack none

Setup:
    AgentDojo is included as a git submodule with PIArena modifications pre-applied.
    Initialize with: git submodule update --init --recursive
    Then install: cd agentdojo && pip install -e .
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import yaml
from dataclasses import dataclass
from datetime import datetime


AGENTDOJO_PATH = "agents/agentdojo"
AZURE_CONFIG_DIR = "configs/azure_configs"

# Known API model prefixes (don't need vLLM server)
API_MODEL_PREFIXES = ["gpt", "claude", "gemini", "command-r"]

SECALIGN_BASE_MODELS = {
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    "70b": "meta-llama/Llama-3.3-70B-Instruct",
}
SECALIGN_MAX_LORA_RANK = 64
SECALIGN_MAX_MODEL_LEN = 24576

# Map Azure model names to AgentDojo model enums
# AgentDojo CLI expects specific enum values, not deployment names
AZURE_TO_AGENTDOJO_MODEL = {
    "gpt-4o": "GPT_4O_2024_05_13",
    "gpt-4o-2024-05-13": "GPT_4O_2024_05_13",
    "gpt-4o-mini": "GPT_4O_MINI_2024_07_18",
    "gpt-4o-mini-2024-07-18": "GPT_4O_MINI_2024_07_18",
    "gpt-4-0125-preview": "GPT_4_0125_PREVIEW",
    "gpt-4-turbo": "GPT_4_TURBO_2024_04_09",
    "gpt-4-turbo-2024-04-09": "GPT_4_TURBO_2024_04_09",
    "gpt-3.5-turbo": "GPT_3_5_TURBO_0125",
    "gpt-3.5-turbo-0125": "GPT_3_5_TURBO_0125",
}


@dataclass(frozen=True)
class VLLMServeConfig:
    model: str
    served_model_name: str
    tokenizer: str | None = None
    lora_adapter: str | None = None
    max_model_len: int = 16384

    @property
    def is_secalign(self) -> bool:
        return self.lora_adapter is not None


def is_azure_model(model: str) -> bool:
    """Check if model is an Azure model (azure/<model> format)."""
    return model.lower().startswith("azure/")


def is_api_model(model: str) -> bool:
    """Check if model is an API-based model (OpenAI, Anthropic, Google, Cohere)."""
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in API_MODEL_PREFIXES)


def is_huggingface_model(model: str) -> bool:
    """Check if model is a HuggingFace model (needs vLLM server)."""
    return not is_azure_model(model) and not is_api_model(model)


def is_secalign_model(model: str) -> bool:
    """Check if model refers to a SecAlign adapter/checkpoint."""
    return "secalign" in model.lower()


def infer_secalign_base_model(model: str) -> str | None:
    """Infer SecAlign's base model from a known adapter/checkpoint name."""
    lowered = model.lower()
    if "_SecAlign" in model:
        return model.split("_SecAlign", 1)[0]
    if "meta-secalign-8b" in lowered:
        return SECALIGN_BASE_MODELS["8b"]
    if "meta-secalign-70b" in lowered:
        return SECALIGN_BASE_MODELS["70b"]
    return None


def resolve_vllm_serve_config(model: str) -> VLLMServeConfig:
    """Resolve how a HuggingFace model should be served through vLLM."""
    if not is_secalign_model(model):
        return VLLMServeConfig(
            model=model,
            served_model_name=model,
        )

    base_model = infer_secalign_base_model(model) or os.getenv("SECALIGN_BASE_MODEL")
    if base_model is None:
        raise ValueError(
            "Could not infer the base model for SecAlign checkpoint "
            f"'{model}'. Use a known SecAlign name or set SECALIGN_BASE_MODEL."
        )

    return VLLMServeConfig(
        model=base_model,
        served_model_name=model,
        tokenizer=model,
        lora_adapter=model,
        max_model_len=SECALIGN_MAX_MODEL_LEN,
    )


def load_azure_config(model_name: str) -> dict:
    """Load Azure config from configs/azure_configs/<model_name>.yaml"""
    config_path = os.path.join(AZURE_CONFIG_DIR, f"{model_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Azure config not found: {config_path}\n"
            f"Available configs: {os.listdir(AZURE_CONFIG_DIR)}"
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the first config entry from 'default'
    if 'default' not in config or not config['default']:
        raise ValueError(f"Invalid Azure config format in {config_path}")

    return config['default'][0]


def check_agentdojo_installed():
    """Check if AgentDojo is cloned and installed."""
    if not os.path.exists(AGENTDOJO_PATH):
        raise FileNotFoundError(
            f"AgentDojo not found at '{AGENTDOJO_PATH}'. Please run:\n"
            "  git submodule update --init --recursive\n"
            "  cd agentdojo && pip install -e ."
        )

    try:
        import agentdojo  # noqa: F401
    except ImportError:
        raise ImportError(
            "AgentDojo not installed as package. Please run:\n"
            "  cd agentdojo && pip install -e ."
        )


def start_vllm_server(model: str, tensor_parallel_size: int, port: int = 8000) -> tuple[subprocess.Popen, str]:
    """Start vLLM server for HuggingFace model."""
    serve_config = resolve_vllm_serve_config(model)
    log_dir = "results/agent_evaluations/agentdojo/vllm_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"vllm_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    cmd = [
        "vllm", "serve", serve_config.model,
        "--dtype", "auto",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", str(serve_config.max_model_len),
    ]

    if serve_config.tokenizer:
        cmd.extend(["--tokenizer", serve_config.tokenizer])
    if serve_config.lora_adapter:
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(SECALIGN_MAX_LORA_RANK),
            "--lora-modules", f"{serve_config.served_model_name}={serve_config.lora_adapter}",
        ])

    print(f"[vLLM] Starting server: {' '.join(cmd)}")
    print(f"[vLLM] Log file: {log_file}")
    if serve_config.is_secalign:
        print(f"[vLLM] SecAlign adapter: {serve_config.lora_adapter}")
        print(f"[vLLM] SecAlign base model: {serve_config.model}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )

    # Wait for server to start
    print("[vLLM] Waiting for server to start...")
    max_wait = 300  # 5 minutes max
    waited = 0
    while waited < max_wait:
        time.sleep(10)
        waited += 10

        with open(log_file, 'r') as f:
            log_content = f.read()
            if "Application startup complete" in log_content:
                print(f"[vLLM] Server started successfully after {waited}s")
                return process, log_file
            if "error" in log_content.lower() and "Error" in log_content:
                print(f"[vLLM] Server failed to start. Check log: {log_file}")
                process.terminate()
                raise RuntimeError(f"vLLM server failed to start. Check {log_file}")

        print(f"[vLLM] Still waiting... ({waited}s)")

    print(f"[vLLM] Timeout waiting for server. Check log: {log_file}")
    process.terminate()
    raise RuntimeError(f"vLLM server timeout. Check {log_file}")


def stop_vllm_server(process: subprocess.Popen):
    """Stop vLLM server."""
    if process:
        print("[vLLM] Stopping server...")
        try:
            # Kill the process group
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except Exception as e:
            print(f"[vLLM] Error stopping server: {e}")
            try:
                process.kill()
            except:
                pass


def run_agentdojo_benchmark(args, model_type: str, azure_model_name: str = None):
    """Run AgentDojo benchmark with specified configuration.

    Args:
        args: Command line arguments
        model_type: One of 'azure', 'api', 'huggingface'
        azure_model_name: The actual model name for Azure (e.g., 'gpt-4o')
    """

    # Set environment variables for PIArena defense
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + ":" + env.get("PYTHONPATH", "")
    env["PIARENA_PATH"] = os.path.dirname(os.path.abspath(__file__))

    if args.defense != "none":
        env["PIARENA_DEFENSE"] = args.defense

    # Set Azure environment variables if needed
    if model_type == "azure":
        azure_config = load_azure_config(azure_model_name)
        env["AZURE_API_KEY"] = azure_config.get("api_key", "")
        env["AZURE_API_ENDPOINT"] = azure_config.get("azure_endpoint", "")
        env["AZURE_API_VERSION"] = azure_config.get("api_version", "2024-12-01-preview")
        # Azure deployment name (may differ from model name, e.g., "gpt-4o" vs "gpt-4o-2024-05-13")
        env["AZURE_DEPLOYMENT"] = azure_config.get("azure_deployment", azure_model_name)
        print(f"[Azure] Using endpoint: {azure_config.get('azure_endpoint', '')}")
        print(f"[Azure] Using deployment: {env['AZURE_DEPLOYMENT']}")

    # Build command
    cmd = [sys.executable, "-m", "agentdojo.scripts.benchmark"]

    if model_type == "huggingface":
        serve_config = resolve_vllm_serve_config(args.model)
        # Use local vLLM server (uppercase LOCAL for AgentDojo CLI enum)
        cmd.extend(["--model", "LOCAL"])
        cmd.extend(["--model-id", serve_config.served_model_name])
        cmd.extend(["--tool-delimiter", "input"])
    elif model_type == "azure":
        # Map Azure deployment name to AgentDojo model enum
        agentdojo_model = AZURE_TO_AGENTDOJO_MODEL.get(azure_model_name, azure_model_name)
        if azure_model_name not in AZURE_TO_AGENTDOJO_MODEL:
            print(f"[Warning] Unknown Azure model '{azure_model_name}', passing as-is. "
                  f"Known models: {list(AZURE_TO_AGENTDOJO_MODEL.keys())}")
        cmd.extend(["--model", agentdojo_model])
    else:
        # Use API model directly
        cmd.extend(["--model", args.model])

    # cmd.extend(["--tool-output-format", "json"])

    # Add attack if not benign
    if args.attack != "none":
        cmd.extend(["--attack", args.attack])

    # Add defense
    if args.defense != "none":
        cmd.extend(["--defense", "piarena"])

    # Add suite if specified
    if args.suite:
        cmd.extend(["-s", args.suite])

    # Add user tasks if specified
    if args.user_tasks:
        for ut in args.user_tasks:
            cmd.extend(["-ut", ut])

    # Set custom results directory (use absolute path since cwd changes)
    results_dir = os.path.abspath(f"results/agent_evaluations/agentdojo/{args.name}")
    os.makedirs(results_dir, exist_ok=True)
    cmd.extend(["--logdir", results_dir])

    print(f"\n[Run] Executing: {' '.join(cmd)}")
    print(f"[Run] Working directory: {AGENTDOJO_PATH}/src")
    if args.defense != "none":
        print(f"[Run] PIARENA_DEFENSE={args.defense}")
    print()

    # Run benchmark
    result = subprocess.run(
        cmd,
        cwd=f"{AGENTDOJO_PATH}/src",
        env=env
    )

    return result.returncode


def main(args):
    # Check AgentDojo is installed
    check_agentdojo_installed()

    # Determine model type
    azure_model_name = None
    if is_azure_model(args.model):
        model_type = "azure"
        azure_model_name = args.model.split("/", 1)[1]  # Extract model name after "azure/"
    elif is_api_model(args.model):
        model_type = "api"
    else:
        model_type = "huggingface"

    # Print configuration
    print("\n" + "="*60)
    print("AgentDojo Benchmark")
    print("="*60)
    print(f"  Model:    {args.model}")
    if model_type == "azure":
        print(f"  Type:     Azure OpenAI ({azure_model_name})")
    elif model_type == "huggingface":
        serve_config = resolve_vllm_serve_config(args.model)
        if serve_config.is_secalign:
            print(f"  Type:     HuggingFace (vLLM + SecAlign LoRA)")
            print(f"  Base:     {serve_config.model}")
        else:
            print(f"  Type:     HuggingFace (vLLM)")
    else:
        print(f"  Type:     API")
    print(f"  Attack:   {args.attack}")
    print(f"  Defense:  {args.defense}")
    print(f"  Suite:    {args.suite or 'all'}")
    if model_type == "huggingface":
        print(f"  TP Size:  {args.tensor_parallel_size}")
    print("="*60)

    vllm_process = None
    returncode = 1

    try:
        # Start vLLM server if needed
        if model_type == "huggingface":
            vllm_process, _ = start_vllm_server(
                args.model,
                args.tensor_parallel_size,
                port=8000
            )

        # Run benchmark
        returncode = run_agentdojo_benchmark(args, model_type, azure_model_name)

    except KeyboardInterrupt:
        print("\n[Interrupted] Cleaning up...")
    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        # Stop vLLM server
        if vllm_process:
            stop_vllm_server(vllm_process)

    # Results are saved to PIArena results directory
    print(f"\n[Done] Results saved in results/agent_evaluations/agentdojo/{args.name}/")

    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentDojo Benchmark Evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13",
                        help="Model: HuggingFace path (e.g., meta-llama/Llama-3.1-8B-Instruct), "
                             "API model (e.g., gpt-4o-2024-05-13), "
                             "or Azure (e.g., azure/gpt-4o)")
    parser.add_argument("--attack", type=str, default="tool_knowledge",
                        choices=["none", "direct", "important_instructions", "tool_knowledge", "injecagent"],
                        help="Attack type to evaluate (use 'none' for benign utility)")
    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "datafilter", "pisanitizer", "promptguard",
                                 "datasentinel", "piguard", "attentiontracker",
                                 "promptarmor", "promptlocate"],
                        help="Defense to evaluate")
    parser.add_argument("--suite", "-s", type=str, default=None,
                        choices=["workspace", "slack", "travel", "banking"],
                        help="Specific suite to evaluate (default: all)")
    parser.add_argument("--user_tasks", "-ut", type=str, nargs="*", default=None,
                        help="Specific user tasks to evaluate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM (for HuggingFace models)")
    parser.add_argument("--name", type=str, default="default",
                        help="Experiment name (for reference)")
    args = parser.parse_args()

    sys.exit(main(args))
