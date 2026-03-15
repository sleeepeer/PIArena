"""
Batch runner for search-based attacks (strategy_search, pair, tap).

Schedules jobs across GPUs with multi-process support.
Supports both local execution and Slurm submission.

Usage:
    python scripts/run_search.py
"""

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
gpu_processes = {}


def run_search(dataset, backend_llm, attack, defense, attacker_llm=None,
               name="search_exp", seed=42, batch_size=8):
    """
    Run a search-based attack evaluation.

    Args:
        dataset: Dataset name or path.
        backend_llm: Target/backend LLM.
        attack: Attack type ("strategy_search", "pair", "tap").
        defense: Defense type.
        attacker_llm: Attacker LLM (defaults to backend_llm).
        name: Experiment name.
        seed: Random seed.
        batch_size: Batch size (for strategy_search).
    """
    global gpu_count, total_jobs, gpus, gpu_processes, processes_per_gpu

    if attacker_llm is None:
        attacker_llm = backend_llm

    gpu_id = gpus[gpu_count]
    gpu_count = (gpu_count + 1) % len(gpus)
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={str(gpu_id)}"

    backend_llm_name = backend_llm.replace('/', '-')
    dataset_name = dataset.split('/')[-1].split('.')[0]
    log_file = f"./logs/search_logs/{name}/{dataset_name}-{backend_llm_name}-{attack}-{defense}-{seed}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if mode == "local":
        if gpu_id not in gpu_processes:
            gpu_processes[gpu_id] = []

        # Remove finished processes
        gpu_processes[gpu_id] = [p for p in gpu_processes[gpu_id] if p.poll() is None]

        # Wait if at capacity
        while len(gpu_processes[gpu_id]) >= processes_per_gpu:
            print(f"[GPU {gpu_id}] Reached max {processes_per_gpu} processes, waiting...")
            for proc in gpu_processes[gpu_id]:
                proc.wait()
                break
            gpu_processes[gpu_id] = [p for p in gpu_processes[gpu_id] if p.poll() is None]

        cmd = (
            f"{gpu_cmd} && python3 -u main_search.py"
            f" --dataset {dataset}"
            f" --backend_llm {backend_llm}"
            f" --attack {attack}"
            f" --defense {defense}"
            f" --attacker_llm {attacker_llm}"
            f" --name {name}"
            f" --seed {seed}"
            f" --batch_size {batch_size}"
        )

        print(f"[GPU {gpu_id}] Starting ({len(gpu_processes[gpu_id]) + 1}/{processes_per_gpu}): "
              f"{dataset_name} - {attack} - {defense}")
        with open(log_file, 'w') as log_f:
            proc = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
        gpu_processes[gpu_id].append(proc)

    elif mode == "slurm":
        cmd = (
            f'sbatch scripts/main_search.sh'
            f' "{dataset}" "{backend_llm}" "{attack}" "{defense}"'
            f' "{name}" "{seed}" "{log_file}" "{attacker_llm}" "{batch_size}"'
        )
        print(cmd)
        os.system(cmd)

    total_jobs += 1
    return 1


# ============================================================================
# Configuration — edit below to set up your experiments
# ============================================================================

name = "search_experiment"               # Experiment name (used for result directory)
gpus = [0, 1]                     # GPU IDs to use
processes_per_gpu = 1             # Search attacks are GPU-intensive; typically 1 per GPU

all_attacks = [
    "pair",                       # PAIR: iterative refinement attack
    "tap",                        # TAP: tree-based attack with pruning
    "strategy_search",          # Evolutionary search with defense feedback
]

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

all_llms = [
    "Qwen/Qwen3-4B-Instruct-2507",
    # "meta-llama/Llama-3.1-8B-Instruct",
]

attacker_llm = "Qwen/Qwen3-4B-Instruct-2507"  # LLM used to generate attacks (can differ from backend_llm)

# ============================================================================
# Launch jobs
# ============================================================================

for attack in all_attacks:
    for defense in all_defenses:
        for dataset in all_datasets:
            for backend_llm in all_llms:
                run_search(
                    dataset=dataset,
                    backend_llm=backend_llm,
                    attack=attack,
                    defense=defense,
                    attacker_llm=attacker_llm,
                    name=name,
                    seed=42,
                    batch_size=8,
                )

print(f"Started {total_jobs} jobs in total")

# Wait for all remaining processes (local mode)
if mode == "local":
    print("Waiting for all remaining jobs to finish...")
    for gpu_id, proc_list in gpu_processes.items():
        for proc in proc_list:
            if proc is not None:
                proc.wait()
    print("All jobs completed.")
