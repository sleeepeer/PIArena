---
title: DataSentinel
slug: defenses/datasentinel
category: defenses
---

# DataSentinel

`datasentinel` is a detection defense. It decides whether the context contains injected instructions and blocks the request if it does.

Source:
[piarena/defenses/datasentinel/defense_datasentinel.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/datasentinel/defense_datasentinel.py)

Related paper:
[DataSentinel](https://arxiv.org/abs/2504.11358)

## How To Use

```python
defense = get_defense("datasentinel")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It runs a detector on the context. If the detector fires, the defense returns a warning string instead of querying the backend model. Otherwise it forwards the original context.

## Parameters

Main config values from `DEFAULT_CONFIG` are nested under:

- `model_info`
- `api_key_info`
- `params`

In practice, the important fields are the detector model settings, device settings, and fine-tuned checkpoint path.
