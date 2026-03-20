---
title: DataFilter
slug: defenses/datafilter
category: defenses
---

# DataFilter

`datafilter` is a sanitization defense that recursively filters suspicious content before answering.

Source:
[piarena/defenses/datafilter/defense_datafilter.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/datafilter/defense_datafilter.py)

Related paper:
[Defending Against Prompt Injection with DataFilter](https://arxiv.org/abs/2510.19207)

## How To Use

```python
defense = get_defense("datafilter")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It parses the context, recursively filters suspicious parts with a dedicated filtering model, and then runs the task on the filtered context.

## Parameters

This defense has no user-facing `DEFAULT_CONFIG` today. It loads the `JoyYizhu/DataFilter` model internally.
