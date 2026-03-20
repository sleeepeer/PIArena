---
title: PAIR
slug: attacks/pair
category: attacks
---

# PAIR

`pair` is a search-based attack that uses an attacker model to iteratively refine injected prompts.

Source:
[piarena/attacks/pair/attack_pair.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/attacks/pair/attack_pair.py)

Related work:
[PAIR: Jailbreaking black box large language models in twenty queries](https://github.com/patrickrchao/JailbreakingLLMs)

## How To Use

```bash
python main_search.py \
  --dataset squad_v2 \
  --attack pair \
  --defense pisanitizer \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507
```

```python
attack = get_attack("pair")
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

PAIR starts from the current task and context, asks an attacker model to propose new attack prompts, and evaluates those prompts through the defense and backend model. It keeps refining until it finds a better prompt or reaches the iteration limit.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `n_streams`: number of concurrent refinement streams
- `n_iterations`: number of refinement rounds
- `temperature`: attacker sampling temperature
- `max_tokens`: maximum attacker generation length
- `inject_position`: where the prompt is inserted
- `pre_test`: whether to try the raw injected task before refinement
- `verbose`: extra logging

Important runtime inputs:

- `llm`
- `attacker_llm`
- `defense`
- `target_inst`
