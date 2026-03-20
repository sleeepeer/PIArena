---
title: Strategy Search
slug: attacks/strategy-search
category: attacks
---

# Strategy Search

`strategy_search` is PIArena's defense-aware evolutionary search attack. It searches for injected prompts that survive a specific defense and still steer the backend model.

Source:
[piarena/attacks/strategy_search/attack_strategy_search.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/attacks/strategy_search/attack_strategy_search.py)

## How To Use

```bash
python main_search.py \
  --dataset squad_v2 \
  --attack strategy_search \
  --defense promptarmor \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507
```

```python
attack = get_attack("strategy_search")
injected_context = attack.execute(
    context=context,
    injected_task=injected_task,
    target_inst=target_inst,
    target_task_answer=target_task_answer,
    injected_task_answer=injected_task_answer,
    llm=backend_llm,
    attacker_llm=attacker_llm,
    defense=defense,
)
```

## What It Does

It begins with multiple initialization strategies, evaluates them through the chosen defense, and then mutates or combines promising candidates across generations. Compared with `pair` and `tap`, it is more explicitly designed around defense feedback.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `population_size`: number of initial strategies
- `init_attempts_per_strategy`: tries per starting strategy
- `max_generations`: search generations
- `success_threshold`: success cutoff
- `early_stop_generations`: stop after no improvement
- `temperature`: attacker generation temperature
- `top_p`: nucleus sampling
- `top_k`: top-k sampling
- `max_tokens`: maximum generation length
- `defense_specific_init`: whether to use defense-specific initial prompts
- `format_attack_prompt`: whether to format prompts with dedicated tags
- `defense_descriptions_json`: optional defense description file
- `verbose`: extra logging

Important runtime inputs:

- `llm`
- `attacker_llm` or `attacker_model_name_or_path`
- `defense`
- `target_inst`
- `target_task_answer`
- `injected_task_answer`
