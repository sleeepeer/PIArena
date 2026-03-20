---
title: NanoGCG
slug: attacks/nanogcg
category: attacks
---

# NanoGCG

`nanogcg` is an optimization-based attack that searches for a prompt suffix that improves attack success.

Source:
[piarena/attacks/nanogcg/attack_nanogcg.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/attacks/nanogcg/attack_nanogcg.py)

Related work:
[GCG](https://arxiv.org/abs/2307.15043)
[nanoGCG](https://github.com/GraySwanAI/nanoGCG)

## How To Use

```python
attack = get_attack("nanogcg")
injected_context = attack.execute(
    context=context,
    injected_task=injected_task,
    target_inst=target_inst,
    injected_task_answer=injected_task_answer,
    llm=backend_llm,
)
```

## What It Does

It creates an injected prompt template and searches for an optimized string that makes the target model more likely to follow the injected task. Compared with heuristic attacks, this is slower but more targeted.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `num_steps`: optimization steps
- `search_width`: number of candidates explored per step
- `topk`: candidate token pool size
- `n_replace`: how many positions are changed each step
- `batch_size`: optimization batch size
- `seed`: random seed
- `verbosity`: logging level
- `early_stop`: whether to stop once a good candidate is found
- `filter_ids`: whether token ids are filtered during search
- `allow_non_ascii`: whether non-ASCII tokens are allowed

Important runtime inputs:

- `target_inst`
- `injected_task_answer`
- `llm`
- `inject_position`
- `inject_times`
