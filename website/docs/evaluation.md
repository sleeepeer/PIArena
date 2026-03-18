---
title: Evaluation Pipeline
description: Run systematic evaluations with CLI arguments or YAML config files.
order: 30
---

# Evaluation Pipeline

PIArena provides a comprehensive evaluation pipeline (`main.py`) to systematically evaluate prompt injection attacks and defenses on different benchmark datasets.

## CLI Mode

```bash
python main.py \
  --dataset open_prompt_injection \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attack direct \
  --defense none \
  --name experiment_name \
  --seed 42
```

## YAML Config Mode

```bash
python main.py --config configs/experiments/my_experiment.yaml
```

## Example YAML Config

```yaml
# configs/experiments/my_experiment.yaml
dataset: open_prompt_injection
backend_llm: Qwen/Qwen3-4B-Instruct-2507
attack: combined
defense: pisanitizer
name: my_experiment
seed: 42

# Optional: config passed to attack/defense constructors
attack_config:
  inject_position: end

defense_config:
  smooth_win: 5
  max_gap: 10
  threshold: 0.01
```

## Result Format

Results are saved to: `results/evaluation_results/{name}/{dataset}-{llm}-{attack}-{defense}-{seed}.json`

```json
{
  "0": {
    "target_inst": "...",
    "injected_task": "...",
    "context": "...",
    "injected_context": "...",
    "defense_result": {"response": "...", "detect_flag": true},
    "utility": 0.85,
    "asr": 0.0
  }
}
```


## Batch Experiments

PIArena provides batch runner scripts that schedule experiments across multiple GPUs with multi-process support. Both local execution and Slurm submission are supported (auto-detected).

The GPU scheduler (`piarena/gpu_utils.py`) automatically assigns each job to the least-loaded GPU. When all GPUs are at capacity, it waits for any running job to finish before launching the next one.

Each script contains a `run()` function and a configuration section. Edit the configuration to set up your experiments, then run the script.

### Standard Attacks (`scripts/run.py`)

```bash
python scripts/run.py
```

Key settings:
```python
name = "my_experiment"        # Experiment name
gpus = [0, 1]                 # GPU IDs to use
processes_per_gpu = 2         # Concurrent processes per GPU

scheduler = GPUScheduler(gpus, processes_per_gpu)

all_attacks = ["none", "direct", "combined"]
all_defenses = ["none", "pisanitizer"]
all_datasets = ["squad_v2", "dolly_closed_qa"]
all_llms = ["Qwen/Qwen3-4B-Instruct-2507"]
```

### Search-based Attacks (`scripts/run_search.py`)

For iterative attacks (PAIR, TAP, Strategy Search), use the search runner:

```bash
python scripts/run_search.py
```

Key settings:
```python
name = "search_exp"
gpus = [0, 1]
processes_per_gpu = 1         # Search attacks are GPU-intensive

scheduler = GPUScheduler(gpus, processes_per_gpu)

all_attacks = ["pair", "tap"]
all_defenses = ["none", "pisanitizer"]
all_datasets = ["squad_v2"]
all_llms = ["Qwen/Qwen3-4B-Instruct-2507"]
attacker_llm = "Qwen/Qwen3-4B-Instruct-2507"
```

### Agent Benchmarks

Similarly, `scripts/run_injecagent.py` and `scripts/run_agentdojo.py` use the same GPU scheduler for agent benchmark experiments. `run_agentdojo.py` supports multi-GPU jobs via `tensor_parallel_size`.

### Logs

Logs are saved to `logs/main_logs/`, `logs/search_logs/`, `logs/injecagent_logs/`, and `logs/agentdojo_logs/` respectively.

## (Optional) Reusing Attack Results

If you already have attack results, you can skip the attack phase:

```bash
python main.py \
  --dataset open_prompt_injection \
  --attack_path results/evaluation_results/test/attack_results.json \
  --defense pisanitizer
```

