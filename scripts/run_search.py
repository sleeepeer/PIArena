"""
Batch runner for search-based attacks (strategy_search, pair, tap).

Usage:
    python scripts/run_search.py
"""

import os
from piarena.gpu_utils import detect_mode, GPUScheduler

mode = detect_mode()


def run_search(dataset, backend_llm, attack, defense, attacker_llm=None,
               name="search_exp", seed=42, batch_size=8):
    if attacker_llm is None:
        attacker_llm = backend_llm

    backend_llm_name = backend_llm.replace('/', '-')
    dataset_name = dataset.split('/')[-1].split('.')[0]

    log_file = f"./logs/search_logs/{name}/{dataset_name}-{backend_llm_name}-{attack}-{defense}-{seed}.txt"

    cmd = (
        f"python3 -u main_search.py"
        f" --dataset {dataset}"
        f" --backend_llm {backend_llm}"
        f" --attack {attack}"
        f" --defense {defense}"
        f" --attacker_llm {attacker_llm}"
        f" --name {name}"
        f" --seed {seed}"
        f" --batch_size {batch_size}"
    )

    if mode == "local":
        gpu_ids = scheduler.pick_gpus(1)
        scheduler.launch(cmd, log_file, gpu_ids)
    elif mode == "slurm":
        slurm_cmd = (
            f'sbatch scripts/main_search.sh'
            f' "{dataset}" "{backend_llm}" "{attack}" "{defense}"'
            f' "{name}" "{seed}" "{log_file}" "{attacker_llm}" "{batch_size}"'
        )
        print(slurm_cmd)
        os.system(slurm_cmd)


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
    "pisanitizer",
    "attentiontracker",
    "promptguard",
    "promptlocate",
    "promptarmor",
    "datasentinel",
    "datafilter",
    "piguard",
    "secalign",
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

for dataset in all_datasets:
    for attack in all_attacks:
        for defense in all_defenses:
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

print(f"Started {scheduler.total_jobs} jobs in total")

if mode == "local":
    scheduler.wait_all()
