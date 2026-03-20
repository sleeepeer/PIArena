---
title: AttentionTracker
slug: defenses/attentiontracker
category: defenses
---

# AttentionTracker

`attentiontracker` is a detection defense that looks at attention patterns from a detector model.

Source:
[piarena/defenses/attentiontracker/defense_attentiontracker.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/attentiontracker/defense_attentiontracker.py)

Related paper:
[Attention Tracker](https://arxiv.org/abs/2411.00348)

## How To Use

```python
defense = get_defense("attentiontracker")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It builds a detector model, analyzes attention behavior on the context, and returns a warning if the context looks injected. Safe inputs are forwarded unchanged.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `model_info`: detector provider, detector name, and detector model id
- `params.temperature`: detector generation temperature
- `params.max_output_tokens`: detector generation length
- `params.important_heads`: attention heads used by the detector
