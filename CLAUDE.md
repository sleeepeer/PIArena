# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the public repo of PIArena — do not leak private info. PIArena is a platform for prompt injection (PI) attack and defense evaluation, providing a plug-and-play toolbox and systematic benchmark for LLM systems.

## Common Commands

```bash
huggingface-cli login

# Generic setup (other machines)
conda create -n piarena python=3.12 -y && conda activate piarena
pip install -r requirements.txt
pip install -e .

# Run single evaluation. main.py uses HF (sample-by-sample). main_vllm.py uses vLLM (batched).
python main.py --dataset squad_v2 --attack combined --defense pisanitizer
python main.py --config configs/experiments/my_experiment.yaml

# Same pipeline on vLLM, batched across attack/defense/judge for higher throughput.
python main_vllm.py --dataset squad_v2 --attack combined --defense pisanitizer \
    --backend_llm Qwen/Qwen3.5-9B --judge_llm Qwen/Qwen3-4B-Instruct-2507 \
    --tp_size 1 --gpu_mem 0.85 --max_model_len 8192

# Search-based attacks: PAIR, TAP, strategy_search (HF backend; needs backend + attacker LLMs)
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
python main_agentdojo.py --model gpt-5-mini --attack none --suite workspace
python main_agentdojo.py --model gpt-4o-2024-08-06 --attack important_instructions --defense datafilter --suite shopping

# Website (React 18 + Vite)
cd website && npm install && npm run dev
```

If editable install fails in a fresh environment, upgrade the packaging toolchain first:

```bash
pip install -U pip setuptools wheel
```

The repository uses `setuptools.build_meta` in `pyproject.toml` for editable installs.

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

The `Model` class picks a backend by:
1. **Provider keyword in model id** (highest priority): `azure` / `openai` / `google` / `anthropic` route to provider SDKs (configs under `configs/<provider>_configs/`).
2. **Explicit `backend=` kwarg** for HF identifiers: `"hf"` (default) or `"vllm"`.

vLLM tunables are constructor kwargs with sensible defaults; `main_vllm.py` exposes them as CLI flags:
- `tp_size` (default 1)
- `gpu_mem` (default 0.85)
- `max_model_len` (default 8192)
- `max_num_seqs` (default 256)
- `enforce_eager` (default False)

Reasoning models (Qwen3+, R1-style): `Model` calls `apply_chat_template(..., enable_thinking=False)` when supported and post-strips any `</think>` block as a safety net. The stripped reasoning text is exposed as `model.last_reasoning` for callers that want it.

Query interface: `model.query(messages, max_new_tokens=1024, temperature=0.01)` where `messages` is a list of `{"role": str, "content": str}` dicts. Batch interface: `model.batch_query(messages_list, ...)`.

Judge LLM: default model is `Qwen/Qwen3-4B-Instruct-2507`. Sample-by-sample callers (`main.py`) lazy-load it via `_ensure_judge()` (HF backend). Batch callers (`main_vllm.py`) explicitly construct a vLLM judge and pass it through the `llm=` kwarg of `llm_judge_utility_batch` / `llm_judge_asr_batch`.

### Evaluator Selection

`main.py` (HF, sample-by-sample): `knowledge_corruption` splits use `substring_match` for both utility and ASR; every other split uses the unified binary `llm_judge_utility` (reference = `target_task_answer`) + `llm_judge_asr` (reference = `injected_task_answer`).

`main_vllm.py` (vLLM, batched): same routing, but uses the batched `llm_judge_utility_batch` / `llm_judge_asr_batch` so the judge runs in one vLLM call per chunk.

`main_search.py` keeps the original per-pattern table (`open_prompt_injection`, `sep`, `*_long`, etc.) since search runs already drive their own legacy evaluators.

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

- `main.py` — Standard evaluation pipeline, HF backend, sample-by-sample
- `main_vllm.py` — Same pipeline on vLLM with batched attack/defense/judge phases (set `--judge_llm same` to share the engine with the backend LLM)
- `main_search.py` — Search-based attacks: PAIR, TAP, strategy_search
  - `pair` / `tap` still use an eager attacker model object
  - `strategy_search` accepts an attacker model path and lazily loads `attacker_llm` only if a non-vLLM fallback is needed
- `main_injecagent.py` / `main_agentdojo.py` — Agent benchmarks
  - `main_agentdojo.py` now covers both classic AgentDojo suites and merged AgentDyn suites in the vendored `agents/agentdojo` tree
- `scripts/run.py` — Batch runner for standard attacks
- `scripts/run_search.py` — Batch runner for search-based attacks
- `scripts/run_injecagent.py` / `scripts/run_agentdojo.py` — Batch runners for agent benchmarks
- All batch scripts use `GPUScheduler` from `piarena/gpu_utils.py` for least-loaded GPU scheduling (auto-detects local vs Slurm)

### Project Docs

- `CHANGELOG.md` — running record of notable repository changes
- `docs/` - user manuals hosted on the project page and consumed by the website

## Workflow Expectations

- When planning implementation work, write the plan as a markdown file under `plans/`.
- When code changes affect behavior, APIs, scripts, workflows, or the website, update the related markdown files and docs in the same change.
- Record notable repository changes in `CHANGELOG.md` as part of the same task.

### Data

Datasets load from local JSON in `datasets/` first, falling back to HuggingFace (`sleeepeer/PIArena`). 17 datasets covering QA, summarization, extraction, long-context, and knowledge corruption tasks.

### Injection Utility (`piarena/utils.py`)

`inject(clean_data, injected_prompt, inject_position, inject_times)` — positions: `"end"`, `"start"`, `"random"` (random sentence insertion). Used by heuristic attacks.
