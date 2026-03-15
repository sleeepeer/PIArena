---
title: Build Your Own
description: Add your own attacks and defenses to PIArena.
order: 60
---

# Your own attacks & defenses

You can easily integrate your own attack or defense into PIArena's benchmark to systematically assess its performance.

## Adding a New Attack

1. Create your attack class in `piarena/attacks/my_attack.py` with the `@register_attack` decorator:

```python
from .base import BaseAttack, register_attack
from ..utils import inject

@register_attack
class MyAttack(BaseAttack):
    """Description of your attack."""

    name = "my_attack"
    DEFAULT_CONFIG = {
        "param1": "default_value",
    }

    def __init__(self, config=None):
        super().__init__(config)
        # Optional: initialize any models or resources here

    def execute(self, context, injected_task,
                inject_position="random", inject_times=1, **kwargs):
        # Build the injection prompt
        injected_prompt = f"Your custom prefix. {injected_task}"

        # Inject into context and return
        return inject(context, injected_prompt, inject_position, inject_times)
```

2. Add a one-line import in `piarena/attacks/__init__.py` to trigger registration:

```python
from . import my_attack  # noqa: F401
```

3. Use it:

```python
from piarena.attacks import get_attack

attack = get_attack("my_attack", config={"param1": "custom_value"})
injected_context = attack.execute(
    context="some text",
    injected_task="do something"
)
```

## Adding a New Defense

1. Create your defense class in `piarena/defenses/my_defense.py` with the `@register_defense` decorator:

```python
from .base import BaseDefense, register_defense


@register_defense
class MyDefense(BaseDefense):
    """Description of your defense."""

    name = "my_defense"
    DEFAULT_CONFIG = {
        "threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        # Load any models here (eager, not lazy)

    def execute(self, target_inst, context):
        # Core defense logic
        detect_flag = False  # your detection logic
        return {"detect_flag": detect_flag}

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)

        if result.get("detect_flag"):
            result["response"] = "[Warning] Injection detected."
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{target_inst}\n\n{context}"},
            ]
            result["response"] = llm.query(messages)

        return result
```

2. Add a one-line import in `piarena/defenses/__init__.py` to trigger registration:

```python
from . import my_defense  # noqa: F401
```

3. Use it:

```python
from piarena.defenses import get_defense

defense = get_defense("my_defense", config={"threshold": 0.8})
result = defense.execute(target_inst="summarize", context="some text")

# Or with LLM response:
result = defense.get_response(
    target_inst="summarize",
    context="some text",
    llm=my_llm
)
print(result["response"])
```

## Key Interfaces

### Attack Interface

All attacks implement `execute()`:

```python
def execute(self, context, injected_task, **kwargs) -> str:
    """Return the injected context string."""
```

### Defense Interface

All defenses implement `execute()` and optionally `get_response()`:

```python
def execute(self, target_inst, context) -> dict:
    """Return dict with at minimum: {"detect_flag": bool}
    Optional: {"cleaned_context": str, ...}"""

def get_response(self, target_inst, context, llm) -> dict:
    """Return dict with: {"response": str, ...}"""
```
