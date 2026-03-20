---
title: No Defense
slug: defenses/none
category: defenses
---

# No Defense

`none` is the baseline defense. It forwards the original context directly to the backend model.

Source:
[piarena/defenses/defense_none.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/defense_none.py)

## How To Use

```python
defense = get_defense("none")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

```bash
python main.py --dataset squad_v2 --attack direct --defense none
```

## What It Does

It does not detect, sanitize, or block anything. It is mainly used as a baseline for both utility and ASR comparisons.

## Parameters

This defense has no configuration.
