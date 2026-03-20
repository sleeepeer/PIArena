---
title: PISanitizer
slug: defenses/pisanitizer
category: defenses
---

# PISanitizer

`pisanitizer` is an attention-based sanitization defense for long-context prompt injection.

Source:
[piarena/defenses/pisanitizer/defense_pisanitizer.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/pisanitizer/defense_pisanitizer.py)

Related paper:
[PISanitizer](https://arxiv.org/abs/2511.10720)

## How To Use

```python
defense = get_defense("pisanitizer")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

```bash
python main.py --dataset squad_v2 --attack combined --defense pisanitizer
```

## What It Does

It runs a sanitizer model, looks for suspicious attention peaks over the context, removes tokens that appear to contain injected instructions, and then queries the backend model on the cleaned context.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `smooth_win`: smoothing window size for attention processing
- `max_gap`: maximum gap used when grouping suspicious regions
- `threshold`: cutoff for token removal
