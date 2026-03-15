import os
import glob
import subprocess

import torch
print("GPUs available:", torch.cuda.device_count())
if torch.cuda.device_count() == 0:
    mode = "slurm"
else:
    mode = "local"

total_jobs = 0
gpu_count = 0
gpu_processes = {}  # Track running processes for each GPU (list of processes per GPU)

def run(model, attack, defense, suite="all", tensor_parallel_size=1, name="test"):
    """
    Run AgentDojo evaluation with specified parameters.
    
    Args:
        model: Name of the backend LLM to be used
        attack: Type of attack (none, direct, important_instructions, tool_knowledge, injecagent)
        defense: Type of defense (none, datafilter, pisanitizer, promptguard, etc.)
        suite: Specific suite to evaluate (all, workspace, slack, travel, banking)
        tensor_parallel_size: Tensor parallel size for vLLM (HuggingFace models)
        name: Name of the experiment
    """
    global gpu_count, total_jobs, gpus, gpu_processes, processes_per_gpu
    
    # Allocate tensor_parallel_size number of GPUs
    allocated_gpus = []
    for _ in range(tensor_parallel_size):
        allocated_gpus.append(gpus[gpu_count])
        gpu_count = (gpu_count + 1) % len(gpus)
    
    gpu_ids_str = ",".join(str(g) for g in allocated_gpus)
    gpu_id = allocated_gpus[0]  # Use first GPU as the primary for process tracking
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_ids_str}"
    
    # Create log file path
    model_name = model.replace('/', '-')
    suite_suffix = f"-{suite}" if suite != "all" else ""
    attack_suffix = attack if attack != "none" else "benign"

    log_file = f"./logs/agentdojo_logs/{name}/{model_name}-{attack_suffix}-{defense}{suite_suffix}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Build command
    if mode == "local":
        # Initialize process list for this GPU if not exists
        if gpu_id not in gpu_processes:
            gpu_processes[gpu_id] = []
        
        # Remove finished processes from the list
        gpu_processes[gpu_id] = [p for p in gpu_processes[gpu_id] if p.poll() is None]
        
        # Wait if we've reached the max processes per GPU
        while len(gpu_processes[gpu_id]) >= processes_per_gpu:
            print(f"[GPU {gpu_id}] Reached max {processes_per_gpu} processes, waiting for one to finish...")
            # Wait for any process to finish
            for proc in gpu_processes[gpu_id]:
                proc.wait()
                break
            # Remove finished processes
            gpu_processes[gpu_id] = [p for p in gpu_processes[gpu_id] if p.poll() is None]
        
        suite_args = f"--suite {suite}" if suite != "all" else ""
        
        cmd = f"{gpu_cmd} && python3 -u main_agentdojo.py \
                --model {model} \
                --attack {attack} \
                --defense {defense} \
                --tensor_parallel_size {tensor_parallel_size} \
                {suite_args} \
                --name {name}"
        
        print(f"[GPU {gpu_id}] Starting ({len(gpu_processes[gpu_id]) + 1}/{processes_per_gpu}): {model_name} - {attack_suffix} - {defense}")
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
        gpu_processes[gpu_id].append(proc)
    elif mode == "slurm":
        cmd = f"sbatch scripts/main_agentdojo.sh \"{model}\" \"{attack}\" \"{defense}\" \"{suite}\" \"{tensor_parallel_size}\" \"{name}\" \"{log_file}\""
        print(cmd)
        os.system(cmd)
    total_jobs += 1
    return 1



name = "agentdojo"
gpus = [0]
processes_per_gpu = 2  # Number of processes allowed to run on each GPU simultaneously

all_attacks = [
    "none",  # benign utility (no attack)
    # "direct",
    # "important_instructions",
    # "tool_knowledge",
    # "injecagent",
]

all_defenses = [
    "none",
    # "pisanitizer",
    # "datafilter",
    # "promptguard",
    # "datasentinel",
    # "piguard",
    # "attentiontracker",
    # "promptarmor",
    # "promptlocate",
]

all_suites = [
    "all",  # run all suites together
    # "workspace",
    # "slack",
    # "travel",
    # "banking",
]

# Model configurations: (model_name, tensor_parallel_size)
all_models = [
    ("GPT_4O", 1),  # gpt-4o-mini API model, TP size ignored
    # ("azure/gpt-4o", 1),  # Azure model, TP size ignored
    # ("meta-llama/Llama-3.1-8B-Instruct", 1),
    # ("meta-llama/Llama-3.1-70B-Instruct", 4),  # Large model needs more GPUs
    # ("Qwen/Qwen3-4B-Instruct-2507", 1),
]

# Run experiments
for model, tensor_parallel_size in all_models:
    for attack in all_attacks:
        for defense in all_defenses:
            for suite in all_suites:
                run(
                    model=model,
                    attack=attack,
                    defense=defense,
                    suite=suite,
                    tensor_parallel_size=tensor_parallel_size,
                    name=name,
                )
print(f"Started {total_jobs} jobs in total")

# Wait for all remaining processes to finish (local mode only)
if mode == "local":
    print("Waiting for all remaining jobs to finish...")
    for gpu_id, proc_list in gpu_processes.items():
        for proc in proc_list:
            if proc is not None:
                proc.wait()
    print("All jobs completed.")
