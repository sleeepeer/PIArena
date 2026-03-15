import os
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

def run(dataset, backend_llm, attack, defense, attack_path=None, name="test", seed=42):
    """
    Run PIArena evaluation with specified parameters.
    
    Args:
        dataset: Path of dataset or subset in https://huggingface.co/datasets/sleeepeer/PIArena
        backend_llm: Name of the backend LLM to be used
        attack: Type of attack to be used
        defense: Type of defense to be used
        attack_path: Existing attack result to reuse (optional)
        name: Name of the experiment
        seed: Seed for the experiment
    """
    global gpu_count, total_jobs, gpus, gpu_processes, processes_per_gpu
    gpu_id = gpus[gpu_count]
    gpu_count = (gpu_count + 1) % len(gpus)
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={str(gpu_id)}"
    
    # Create log file path
    backend_llm_name = backend_llm.replace('/', '-')
    dataset_name = dataset.split('/')[-1].split('.')[0]

    if attack_path is not None:
        attack_name = attack_path.split('/')[-1].split('.')[0]
    else:
        attack_name = attack

    log_file = f"./logs/main_logs/{name}/{dataset_name}-{backend_llm_name}-{attack_name}-{defense}-{seed}.txt"
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
        
        cmd = f"{gpu_cmd} && python3 -u main.py \
                --dataset {dataset} \
                --backend_llm {backend_llm} \
                --attack {attack} \
                --defense {defense} \
                --attack_path {attack_path} \
                --name {name} \
                --seed {seed}"
        
        print(f"[GPU {gpu_id}] Starting ({len(gpu_processes[gpu_id]) + 1}/{processes_per_gpu}): {dataset_name} - {attack} - {defense}")
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
        gpu_processes[gpu_id].append(proc)
    elif mode == "slurm":
        cmd = f"sbatch scripts/main.sh \"{dataset}\" \"{backend_llm}\" \"{attack}\" \"{defense}\" \"{attack_path}\" \"{name}\" \"{seed}\" \"{log_file}\""
        print(cmd)
        os.system(cmd)
    total_jobs += 1
    return 1



# ============================================================================
# Configuration — edit below to set up your experiments
# ============================================================================

name = "main_experiment"          # Experiment name (used for result directory)
gpus = [0, 1]                     # GPU IDs to use
processes_per_gpu = 2             # Processes per GPU (2-3 for heuristic attacks)

all_defenses = [
    "none",                       # No defense (baseline)
    # "pisanitizer",
    # "attentiontracker",
    # "promptguard",
    # "promptlocate",
    # "promptarmor",
    # "datasentinel",
    # "datafilter",
    # "piguard",
    # "secalign",
]

all_datasets = [
    # --- QA ---
    "squad_v2",
    # "dolly_closed_qa",
    # --- Extraction / Summarization ---
    # "dolly_information_extraction",
    # "dolly_summarization",
    # --- RAG ---
    # "nq_rag",
    # "hotpotqa_rag",
    # "msmarco_rag",
    # --- Long-context ---
    # "qasper_long",
    # "multi_news_long",
    # "hotpotqa_long",
]

all_attacks = [
    "none",                       # No attack (utility-only baseline)
    "direct",                     # Direct injection
    "combined",                   # Combined heuristic attack
    # "ignore",
    # "completion",
    # "character",
]

all_llms = [
    "Qwen/Qwen3-4B-Instruct-2507",
    # "meta-llama/Llama-3.1-8B-Instruct",
]

# ============================================================================
# Launch jobs
# ============================================================================

for defense in all_defenses:
    for dataset in all_datasets:
        for attack in all_attacks:
            for backend_llm in all_llms:
                run(
                    dataset=dataset,
                    backend_llm=backend_llm,
                    attack=attack,
                    defense=defense,
                    attack_path=None,
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
