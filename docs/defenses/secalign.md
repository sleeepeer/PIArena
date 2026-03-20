---
title: SecAlign
slug: defenses/secalign
category: defenses
---

# SecAlign

`secalign` uses a specialized aligned model instead of passing the cleaned context into the normal backend model.

Source:
[piarena/defenses/secalign/defense_secalign.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/secalign/defense_secalign.py)

Related paper:
[SecAlign](https://arxiv.org/abs/2410.05451)

## How To Use

```python
defense = get_defense("secalign")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=None,
)
```

```bash
python main.py --dataset squad_v2 --attack direct --defense secalign
```

## What It Does

Unlike most defenses, `secalign` does not use the passed backend `llm` for generation. It loads its own aligned model and answers with that model directly.

## Parameters

Main config values from `DEFAULT_CONFIG`:

- `model_name_or_path`: the SecAlign model path to load
