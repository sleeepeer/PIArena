---
title: Evaluation
slug: evaluation
category: guide
---

# Evaluation

PIArena has two main evaluation paths:

- `main.py` for standard attacks and defenses
- `main_search.py` for search-based attacks such as `pair`, `tap`, and `strategy_search`

## Standard Evaluation

Use `main.py` when the attack is directly constructed from the input sample.

```bash
python main.py \
  --dataset squad_v2 \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attack combined \
  --defense pisanitizer \
  --name demo \
  --seed 42
```

You can also use a config file:

```yaml
dataset: squad_v2
backend_llm: Qwen/Qwen3-4B-Instruct-2507
attack: combined
defense: pisanitizer
name: demo
seed: 42

attack_config:
  inject_position: end

defense_config:
  smooth_win: 5
  max_gap: 10
  threshold: 0.01
```

```bash
python main.py --config configs/experiments/my_experiment.yaml
```

## Search-Based Evaluation

Use `main_search.py` for attacks that iteratively search for stronger injections.

```bash
python main_search.py \
  --dataset squad_v2 \
  --attack strategy_search \
  --defense pisanitizer \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507
```

The search-based attacks supported today are:

- [`pair`](/docs/attacks/pair)
- [`tap`](/docs/attacks/tap)
- [`strategy_search`](/docs/attacks/strategy-search)

## Batch Runs

For large experiments, edit and run the batch scripts:

- `scripts/run.py` for standard attacks
- `scripts/run_search.py` for search-based attacks
- `scripts/run_injecagent.py` and `scripts/run_agentdojo.py` for agent benchmarks

Example:

```bash
python scripts/run.py
python scripts/run_search.py
```

These scripts use `piarena.gpu_utils.GPUScheduler` to distribute work across GPUs.

## Results

Evaluation results are saved under:

```text
results/evaluation_results/{name}/
```

Standard runs also cache attack outputs under:

```text
results/evaluation_results/{name}/tmp_attack_results/
```

Each record stores the sample, defense result, utility, and ASR so interrupted runs can resume.

## Agent Benchmarks

PIArena also includes agent benchmark entry points:

- `main_injecagent.py`
- `main_agentdojo.py`

Minimal setup:

```bash
git submodule update --init --recursive
cd agents/agentdojo && pip install -e . && cd ../..
```

Examples:

```bash
python main_injecagent.py --model meta-llama/Llama-3.1-8B-Instruct --defense none
python main_agentdojo.py --model gpt-5-mini --attack none
```

## How Evaluation Is Chosen

In `main.py`, evaluator selection depends on dataset name:

- `open_prompt_injection` uses `llm_judge` and `open_prompt_injection_utility`
- `sep` uses `llm_judge`
- `knowledge_corruption` uses substring matching
- long-context datasets use `llm_judge` for ASR and task-specific LongBench metrics for utility

This is why the same attack or defense can be evaluated differently depending on the dataset.
