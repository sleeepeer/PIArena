---
title: PromptLocate
slug: defenses/promptlocate
category: defenses
---

# PromptLocate

`promptlocate` is a hybrid defense that first detects prompt injection and then tries to localize and remove the suspicious text.

Source:
[piarena/defenses/promptlocate/defense_promptlocate.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/promptlocate/defense_promptlocate.py)

Related paper:
[PromptLocate](https://arxiv.org/abs/2510.12252)

## How To Use

```python
defense = get_defense("promptlocate")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It uses a detector to decide whether the input looks injected. If so, it tries to recover a cleaned context and also returns the localized suspicious span as `potential_injection`.

## Parameters

Main config values from `DEFAULT_CONFIG` are nested under:

- `model_info`
- `api_key_info`
- `params`

The most important fields are the detector or locator model settings, device settings, and fine-tuned checkpoint path.
