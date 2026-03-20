---
title: TAP
slug: attacks/tap
category: attacks
---

# TAP

`tap` is a search-based attack that explores multiple attack branches and prunes weaker candidates over time.

Source:
[piarena/attacks/tap/attack_tap.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/attacks/tap/attack_tap.py)

Related work:
[TAP: A Query-Efficient Method for Jailbreaking Black-Box LLMs](https://github.com/RICommunity/TAP)

## How To Use

```bash
python main_search.py \
  --dataset squad_v2 \
  --attack tap \
  --defense pisanitizer \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507
```

```python
attack = get_attack("tap")
injected_context = attack.execute(
    context=context,
    injected_task=injected_task,
    target_inst=target_inst,
    llm=backend_llm,
    attacker_llm=attacker_llm,
    defense=defense,
)
```

## What It Does

TAP builds a tree of attack candidates. At each depth it expands promising prompts, evaluates them, and prunes weaker ones. This makes it more structured than simple iterative refinement.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `root_nodes`: number of initial roots
- `branching_factor`: children per expansion step
- `width`: how many candidates survive pruning
- `depth`: search depth
- `temperature`: attacker sampling temperature
- `max_tokens`: attacker generation length
- `inject_position`: where the prompt is inserted
- `pre_test`: whether to test the raw injected task first
- `evaluator_temperature`: temperature for evaluator calls
- `verbose`: extra logging

Important runtime inputs:

- `llm`
- `attacker_llm`
- `evaluator_llm`
- `defense`
- `target_inst`
