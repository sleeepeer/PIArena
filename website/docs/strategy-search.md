---
title: Search-based Attacks
description: Iterative and evolutionary attacks (PAIR, TAP, Strategy Search) that adaptively find effective prompt injection payloads.
order: 40
---

# Search-based Attacks

PIArena supports search-based attacks that iteratively refine prompt injections using an attacker LLM. These attacks are more powerful than heuristic attacks as they adapt to the target defense.

Use `main_search.py` (instead of `main.py`) to run search-based attacks.

## Quick Start

```bash
# PAIR attack sample
python main_search.py \
  --dataset squad_v2 \
  --attack pair \
  --defense pisanitizer \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507

# TAP attack sample
python main_search.py \
  --dataset squad_v2 \
  --attack tap \
  --defense secalign \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507

# Strategy Search attack
python main_search.py \
  --dataset squad_v2 \
  --attack strategy_search \
  --defense secalign \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507
```

## How Each Attack Works

### PAIR (Prompt Automatic Iterative Refinement)

PAIR uses an attacker LLM to iteratively refine injection prompts through multi-round conversation:

1. The attacker LLM proposes an injection prompt
2. The target system (with defense) processes the injected context
3. A judge evaluates whether the injection succeeded
4. The attacker receives feedback and refines its prompt
5. Repeat for up to `n_iterations` rounds

### TAP (Tree of Attacks with Pruning)

TAP extends PAIR with a tree search strategy for more efficient exploration:

1. Start with root nodes containing initial attack attempts
2. **Branch**: Each node generates `branching_factor` children via the attacker LLM
3. **Prune off-topic**: Remove candidates that drift from the injection goal
4. **Prune by score**: Keep only the top `width` candidates by score
5. Repeat for up to `depth` levels

### Strategy Search

The strategy search attack uses an evolutionary algorithm with defense feedback:

1. **Initialization**: Try multiple pre-defined injection strategies
2. **Search**: If initial attempts fail, use evolutionary mutations guided by defense responses
3. Uses batch inference (vLLM) for efficient evaluation of candidate payloads

## Batch Experiments

Use `scripts/run_search.py` to run many search-based experiments across GPUs:

```bash
python scripts/run_search.py
```

Edit the configuration section in `scripts/run_search.py` to specify attacks, defenses, datasets, and LLMs. See [Batch Experiments](/docs/evaluation#batch-experiments) for more details.
