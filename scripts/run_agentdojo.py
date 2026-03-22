import os
from piarena.gpu_utils import detect_mode, GPUScheduler

mode = detect_mode()


def run(model, attack, defense, suite="all", tensor_parallel_size=1, name="test"):
    model_name = model.replace('/', '-')
    suite_suffix = f"-{suite}" if suite != "all" else ""
    attack_suffix = attack if attack != "none" else "benign"

    log_file = f"./logs/agentdojo_logs/{name}/{model_name}-{attack_suffix}-{defense}{suite_suffix}.txt"

    suite_args = f" --suite {suite}" if suite != "all" else ""
    cmd = (
        f"python3 -u main_agentdojo.py"
        f" --model {model}"
        f" --attack {attack}"
        f" --defense {defense}"
        f" --tensor_parallel_size {tensor_parallel_size}"
        f"{suite_args}"
        f" --name {name}"
    )

    if mode == "local":
        gpu_ids = scheduler.pick_gpus(tensor_parallel_size)
        scheduler.launch(cmd, log_file, gpu_ids)
    elif mode == "slurm":
        slurm_cmd = f'sbatch scripts/main_agentdojo.sh "{model}" "{attack}" "{defense}" "{suite}" "{tensor_parallel_size}" "{name}" "{log_file}"'
        print(slurm_cmd)
        os.system(slurm_cmd)
        scheduler.total_jobs += 1


# ============================================================================
# Configuration — edit below to set up your experiments
# ============================================================================

name = "agentdojo"
gpus = [0]
processes_per_gpu = 2

scheduler = GPUScheduler(gpus, processes_per_gpu)

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
    # "shopping",
    # "github",
    # "dailylife",
]

# Model configurations: (model_name, tensor_parallel_size)
all_models = [
    ("GPT_4O", 1),  # API model, TP size ignored
    # ("azure/gpt-4o", 1),  # Azure model, TP size ignored
    # ("meta-llama/Llama-3.1-8B-Instruct", 1),
    # ("meta-llama/Llama-3.1-70B-Instruct", 4),  # Large model needs more GPUs
    # ("Qwen/Qwen3-4B-Instruct-2507", 1),
]

# ============================================================================
# Launch jobs
# ============================================================================

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

print(f"Started {scheduler.total_jobs} jobs in total")

if mode == "local":
    scheduler.wait_all()
