---
title: Character Attack
slug: attacks/character
category: attacks
---

# Character Attack

`character` is a formatting-based heuristic attack that inserts the injected task with a leading newline.

Source:
[piarena/attacks/heuristic.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/attacks/heuristic.py)

Background:
[Delimiters won’t save you from prompt injection](https://simonwillison.net/2023/May/11/delimiters-wont-save-you/)

## How To Use

```python
attack = get_attack("character")
injected_context = attack.execute(
    context=context,
    injected_task=injected_task,
    inject_position="random",
    inject_times=1,
)
```

## What It Does

It inserts:

```text
\n{injected_task}
```

The idea is simple: even small formatting changes can make injected text look more instruction-like to the model.

## Parameters

- `inject_position`: `start`, `end`, or `random`
- `inject_times`: number of insertions
