# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the public repo of PIArena — do not leak private info. PIArena is a platform for prompt injection (PI) attack and defense evaluation, providing a plug-and-play toolbox and systematic benchmark for LLM systems.

## Common Commands

```bash
# Setup
conda create -n piarena python=3.10 -y && conda activate piarena
pip install -r requirements.txt
pip install -e .  # Install piarena as editable package (required for batch scripts)
# torch/vllm are commented out in requirements.txt — install separately for your CUDA version
huggingface-cli login

# Run single evaluation (requires GPU)
python main.py --dataset open_prompt_injection --attack combined --defense pisanitizer
python main.py --config configs/experiments/my_experiment.yaml

# Search-based attacks: PAIR, TAP, strategy_search (needs backend + attacker LLMs)
python main_search.py --attack pair --backend_llm <model> --attacker_llm <model> --defense pisanitizer
python main_search.py --attack tap --backend_llm <model> --attacker_llm <model> --defense pisanitizer
python main_search.py --attack strategy_search --backend_llm <model> --attacker_llm <model> --defense pisanitizer --batch_size 8

# Run batch experiments (edit scripts to configure jobs, GPUs, datasets)
python scripts/run.py           # Standard attacks
python scripts/run_search.py    # Search-based attacks

# Agent benchmarks
git submodule update --init --recursive
cd agents/agentdojo && pip install -e . && cd ../..
python main_injecagent.py --model meta-llama/Llama-3.1-8B-Instruct --defense none
python main_agentdojo.py --model gpt-5-mini --attack none

# Website (React 18 + Vite)
cd website && npm install && npm run dev
```

## Architecture

### Pipeline Flow

**Attack → Defense → LLM → Evaluation**

Each dataset item has: `target_inst`, `context`, `injected_task`, `target_task_answer`, `injected_task_answer`.

1. **Attack** transforms `context` into `injected_context` (embedding the injected task)
2. **Defense** processes `(target_inst, injected_context)` and queries the LLM
3. **Evaluation** checks both utility (did LLM answer the original task?) and ASR (did the injection succeed?)

### Registry Pattern (`piarena/registry.py`)

Global `ATTACK_REGISTRY` and `DEFENSE_REGISTRY` use `@registry.register` decorators. Modules auto-register on import via `__init__.py`. Factory functions: `get_attack(name, config)`, `get_defense(name, config)`.

### LLM Backend Selection (`piarena/llm.py`)

The `Model` class selects backend by model name string:
- Contains `azure` → Azure OpenAI (config from `configs/azure_configs/`)
- Contains `google` → Google GenAI (config from `configs/google_configs/`)
- Contains `anthropic` → Anthropic SDK (config from `configs/anthropic_configs/`)
- Everything else → HuggingFace Transformers (loaded via `AutoModelForCausalLM`)

Query interface: `model.query(messages, max_new_tokens=1024, temperature=0.01)` where `messages` is a list of `{"role": str, "content": str}` dicts. Batch interface: `model.batch_query(messages_list, ...)`.

### Evaluator Selection (in `main.py`)

Evaluator is chosen by dataset name pattern:
- `open_prompt_injection` → `llm_judge` + `open_prompt_injection_utility`
- `sep` → `llm_judge` + `llm_judge`
- `knowledge_corruption` → `substring_match` + `substring_match`
- `*_long` → `llm_judge` + LongBench metrics (qa_f1, rouge, retrieval, code_sim)
- Default → `llm_judge` + `llm_judge`

The `llm_judge` uses `Qwen/Qwen3-4B-Instruct-2507` by default (globally cached).

### Adding New Attacks/Defenses

1. Create a new file in `piarena/attacks/` or `piarena/defenses/`
2. Subclass `BaseAttack` or `BaseDefense`, set a `name` class attribute
3. Decorate with `@ATTACK_REGISTRY.register` or `@DEFENSE_REGISTRY.register`
4. Import the module in the package `__init__.py` to trigger registration

**BaseAttack**: implement `execute(context, injected_task, **kwargs) -> str` returning injected context.

**BaseDefense**: implement `execute(target_inst, context) -> dict` and optionally override `get_response(target_inst, context, llm) -> dict`.

Config merging: `DEFAULT_CONFIG` (class-level) is merged with `config` dict passed to constructor.

### Batch Defenses

Batch support now lives on the defense classes themselves through `BaseDefense.execute_batch()` and `BaseDefense.get_response_batch()`. Defenses may override these methods for true batching or rely on the default loop-based fallback. `strategy_search` uses the defense object directly rather than a separate batch registry.

### Checkpointing & Results

- Results saved to `results/evaluation_results/{name}/{dataset}-{llm}-{attack}-{defense}-{seed}.json`
- Attack results cached separately in `tmp_attack_results/` for reuse across defense runs (use `--attack_path` to load pre-computed attacks)
- Re-running skips already-computed sample indices

### Configuration Priority

CLI args > YAML config (`configs/experiments/`) > hardcoded defaults. YAML supports `attack_config` and `defense_config` sub-dicts for component-specific settings.

### Entry Points

- `main.py` — Main evaluation pipeline (GPU required)
- `main_search.py` — Search-based attacks: PAIR, TAP, strategy_search
  - `pair` / `tap` still use an eager attacker model object
  - `strategy_search` accepts an attacker model path and lazily loads `attacker_llm` only if a non-vLLM fallback is needed
- `main_injecagent.py` / `main_agentdojo.py` — Agent benchmarks
- `scripts/run.py` — Batch runner for standard attacks
- `scripts/run_search.py` — Batch runner for search-based attacks
- `scripts/run_injecagent.py` / `scripts/run_agentdojo.py` — Batch runners for agent benchmarks
- All batch scripts use `GPUScheduler` from `piarena/gpu_utils.py` for least-loaded GPU scheduling (auto-detects local vs Slurm)

### Project Docs

- `CHANGELOG.md` — running record of notable repository changes
- `website/docs/` - user manuals hosted on the project page

### Data

Datasets load from local JSON in `datasets/` first, falling back to HuggingFace (`sleeepeer/PIArena`). 17 datasets covering QA, summarization, extraction, long-context, and knowledge corruption tasks.

### Injection Utility (`piarena/utils.py`)

`inject(clean_data, injected_prompt, inject_position, inject_times)` — positions: `"end"`, `"start"`, `"random"` (random sentence insertion). Used by heuristic attacks.
