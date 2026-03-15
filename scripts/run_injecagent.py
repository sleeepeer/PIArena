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

def run(model, defense, name="test", seed=42):
    """
    Run InjecAgent evaluation with specified parameters.
    
    Args:
        model: Name of the backend LLM to be used
        defense: Type of defense to be used (none, datafilter, pisanitizer, promptguard, etc.)
        name: Name of the experiment
        seed: Seed for the experiment
    """
    global gpu_count, total_jobs, gpus, gpu_processes, processes_per_gpu
    gpu_id = gpus[gpu_count]
    gpu_count = (gpu_count + 1) % len(gpus)
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={str(gpu_id)}"
    
    # Create log file path
    model_name = model.replace('/', '-')

    log_file = f"./logs/injecagent_logs/{name}/{model_name}-{defense}-{seed}.txt"
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
        
        cmd = f"{gpu_cmd} && python3 -u main_injecagent.py \
                --model {model} \
                --defense {defense} \
                --name {name} \
                --seed {seed} \
                --checkpoint_interval 10"
        
        print(f"[GPU {gpu_id}] Starting ({len(gpu_processes[gpu_id]) + 1}/{processes_per_gpu}): {model_name} - {defense}")
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
        gpu_processes[gpu_id].append(proc)
    elif mode == "slurm":
        cmd = f"sbatch scripts/main_injecagent.sh \"{model}\" \"{defense}\" \"{name}\" \"{seed}\" \"{log_file}\""
        print(cmd)
        os.system(cmd)
    total_jobs += 1
    return 1



name = "injecagent"
gpus = [0, 1]
processes_per_gpu = 1  # Number of processes allowed to run on each GPU simultaneously

all_defenses = [
    "none",
    # "datafilter",
    # "pisanitizer",
    # "promptguard",
    # "datasentinel",
    # "piguard",
    # "attentiontracker",
    # "promptarmor",
    # "promptlocate",
    # "secalign",
]

all_models = [
    "Qwen/Qwen3-4B-Instruct-2507",
]

# Run experiments
for model in all_models:
    for defense in all_defenses:
        run(
            model=model,
            defense=defense,
            name=name,
            seed=42,
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
