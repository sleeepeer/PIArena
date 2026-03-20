---
title: No Attack
slug: attacks/none
category: attacks
---

# No Attack

`none` is the baseline attack. It leaves the context unchanged.

Source:
[piarena/attacks/attack_none.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/attacks/attack_none.py)

## How To Use

```python
from piarena.attacks import get_attack

attack = get_attack("none")
injected_context = attack.execute(
    context=context,
    injected_task=injected_task,
)
```

```bash
python main.py --dataset squad_v2 --attack none --defense none
```

## What It Does

It returns the original context without modification. This is useful when you want to measure utility without any injected prompt.

## Parameters

This attack has no effective configuration. It ignores injection settings because it does not alter the context.
