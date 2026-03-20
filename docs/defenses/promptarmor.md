---
title: PromptArmor
slug: defenses/promptarmor
category: defenses
---

# PromptArmor

`promptarmor` is a hybrid defense that uses an auxiliary LLM to decide whether a context contains injected instructions and, if possible, remove them.

Source:
[piarena/defenses/promptarmor/defense_promptarmor.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/promptarmor/defense_promptarmor.py)

Related paper:
[PromptArmor](https://arxiv.org/abs/2507.15219)

## How To Use

```python
defense = get_defense("promptarmor")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It asks an auxiliary model whether the context contains prompt injection. If the model returns an extracted injection span, the defense removes that span and queries the backend model on the cleaned context.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `model_name_or_path`: the auxiliary model used for detection and cleanup
