---
title: PromptGuard
slug: defenses/promptguard
category: defenses
---

# PromptGuard

`promptguard` is a lightweight classifier-based detection defense built on Meta Prompt Guard.

Source:
[piarena/defenses/promptguard/defense_promptguard.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/promptguard/defense_promptguard.py)

Model:
[Meta Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M)

## How To Use

```python
defense = get_defense("promptguard")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It classifies the context as benign or suspicious. Suspicious inputs are blocked with a warning. Benign inputs are passed to the backend model.

## Parameters

This defense has no user-facing configuration in `DEFAULT_CONFIG`. It loads the Prompt Guard classifier directly.
