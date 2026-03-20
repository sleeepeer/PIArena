---
title: PIGuard
slug: defenses/piguard
category: defenses
---

# PIGuard

`piguard` is a fine-tuned classification defense for prompt injection detection.

Source:
[piarena/defenses/piguard/defense_piguard.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/piguard/defense_piguard.py)

Related paper:
[PIGuard](https://arxiv.org/abs/2410.22770)

## How To Use

```python
defense = get_defense("piguard")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It classifies the context as `benign` or `injection`. If the prediction is `injection`, the defense returns a warning instead of forwarding the context to the backend model.

## Parameters

This defense has no user-facing `DEFAULT_CONFIG`. It loads the `leolee99/PIGuard` classifier directly.
