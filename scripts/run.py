import os
from piarena.gpu_utils import detect_mode, GPUScheduler

mode = detect_mode()


def run(dataset, backend_llm, attack, defense, attack_path=None, name="test", seed=42):
    backend_llm_name = backend_llm.replace('/', '-')
    dataset_name = dataset.split('/')[-1].split('.')[0]
    attack_name = attack_path.split('/')[-1].split('.')[0] if attack_path else attack

    log_file = f"./logs/main_logs/{name}/{dataset_name}-{backend_llm_name}-{attack_name}-{defense}-{seed}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    attack_path_arg = f" --attack_path {attack_path}" if attack_path else ""
    cmd = (
        f"python3 -u main.py"
        f" --dataset {dataset}"
        f" --backend_llm {backend_llm}"
        f" --attack {attack}"
        f" --defense {defense}"
        f"{attack_path_arg}"
        f" --name {name}"
        f" --seed {seed}"
    )

    if mode == "local":
        gpu_ids = scheduler.pick_gpus(1)
        scheduler.launch(cmd, log_file, gpu_ids)
    elif mode == "slurm":
        slurm_cmd = f'sbatch scripts/main.sh "{dataset}" "{backend_llm}" "{attack}" "{defense}" "{attack_path}" "{name}" "{seed}" "{log_file}"'
        print(slurm_cmd)
        os.system(slurm_cmd)
        scheduler.total_jobs += 1


# ============================================================================
# Configuration — edit below to set up your experiments
# ============================================================================

name = "main_experiment"          # Experiment name (used for result directory)
gpus = [0, 1]                     # GPU IDs to use
processes_per_gpu = 2             # Processes per GPU (2-3 for heuristic attacks)

scheduler = GPUScheduler(gpus, processes_per_gpu)

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
                    name=name,
                    seed=42,
                )

print(f"Started {scheduler.total_jobs} jobs in total")

if mode == "local":
    scheduler.wait_all()
