import os
from piarena.gpu_utils import detect_mode, GPUScheduler

mode = detect_mode()


def run(model, defense, name="test", seed=42):
    model_name = model.replace('/', '-')

    log_file = f"./logs/injecagent_logs/{name}/{model_name}-{defense}-{seed}.txt"

    cmd = (
        f"python3 -u main_injecagent.py"
        f" --model {model}"
        f" --defense {defense}"
        f" --name {name}"
        f" --seed {seed}"
        f" --checkpoint_interval 10"
    )

    if mode == "local":
        gpu_ids = scheduler.pick_gpus(1)
        scheduler.launch(cmd, log_file, gpu_ids)
    elif mode == "slurm":
        slurm_cmd = f'sbatch scripts/main_injecagent.sh "{model}" "{defense}" "{name}" "{seed}" "{log_file}"'
        print(slurm_cmd)
        os.system(slurm_cmd)


# ============================================================================
# Configuration — edit below to set up your experiments
# ============================================================================

name = "injecagent"
gpus = [0, 1]
processes_per_gpu = 1

scheduler = GPUScheduler(gpus, processes_per_gpu)

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

# ============================================================================
# Launch jobs
# ============================================================================

for model in all_models:
    for defense in all_defenses:
        run(
            model=model,
            defense=defense,
            name=name,
            seed=42,
        )

print(f"Started {scheduler.total_jobs} jobs in total")

if mode == "local":
    scheduler.wait_all()
