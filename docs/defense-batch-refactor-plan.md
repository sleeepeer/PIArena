# Defense Batch Refactor Plan

## Implementation Status

The first implementation pass is complete:

- `BaseDefense` now exposes `execute_batch()` and `get_response_batch()`
- `Model` now exposes `batch_query()` with provider-aware fallback
- batch behavior has been moved into the defense classes
- `DEFENSES_BATCH` has been removed
- `strategy_search` now uses the defense object directly

Remaining work, if needed, is performance tuning and adding more specialized batch implementations for defenses that still use the default loop-based fallback.

## Goal

Move batch defense support into the defense classes themselves, without introducing a large new API surface.

The plan is:

- keep the current defense interface
- add `execute_batch()` and `get_response_batch()`
- lazily initialize models and vLLM engines inside defense classes
- remove the separate batch-defense registry
- simplify `main_search.py`, especially `strategy_search`

This should make defenses easier to reuse in other projects while still keeping it easy to add a new defense.

## Main Idea

We do not need a new defense framework.

We keep the current pattern:

- `execute(target_inst, context) -> dict`
- `get_response(target_inst, context, llm) -> dict`

and extend it with:

- `execute_batch(target_insts, contexts) -> list[dict]`
- `get_response_batch(target_insts, contexts, llm) -> list[dict]`

That is enough for the current refactor.

## Desired BaseDefense Shape

The base class should stay simple.

```python
class BaseDefense(ABC):
    name: str
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    @abstractmethod
    def execute(self, target_inst: str, context: str) -> dict:
        ...

    def execute_batch(self, target_insts: list[str], contexts: list[str]) -> list[dict]:
        if len(target_insts) != len(contexts):
            raise ValueError("target_insts and contexts must have the same length")
        return [
            self.execute(target_inst, context)
            for target_inst, context in zip(target_insts, contexts)
        ]

    def get_response(self, target_inst: str, context: str, llm,
                     system_prompt: str = "You are a helpful assistant.") -> dict:
        ...

    def get_response_batch(self, target_insts: list[str], contexts: list[str], llm,
                           system_prompt: str = "You are a helpful assistant.") -> list[dict]:
        if len(target_insts) != len(contexts):
            raise ValueError("target_insts and contexts must have the same length")
        return [
            self.get_response(target_inst, context, llm, system_prompt=system_prompt)
            for target_inst, context in zip(target_insts, contexts)
        ]
```

This gives us:

- one defense object
- one consistent API
- batch support by default
- no need for a separate `DEFENSES_BATCH`

## Initialization Strategy

Defense classes should not fully load models in `__init__()`.

Instead:

- keep model handles as `None`
- initialize them on the first call to `execute()`, `execute_batch()`, `get_response()`, or `get_response_batch()`
- reuse them on later calls

Example pattern:

```python
class PromptGuardDefense(BaseDefense):
    def __init__(self, config=None):
        super().__init__(config)
        self._detector = None
        self._target_vllm = None

    def _ensure_detector(self):
        if self._detector is None:
            self._detector = ...

    def _ensure_target_vllm(self, model_name_or_path):
        if self._target_vllm is None:
            self._target_vllm = ...
```

This is important for:

- startup simplicity
- AgentDojo-style lazy usage
- avoiding unnecessary model loads

## Batch Policy

### Non-batchable defenses

For defenses that do not have a real batch implementation:

- keep `execute()` as the source of truth
- use a for loop in `execute_batch()` to simulate a batch inference
- keep the input and output format identical to other batchable defenses
- always use execute_batch() or get_response_batch() in strategy_search attack

This keeps the API uniform.

### Batchable defenses

For defenses that can batch efficiently:

- override `execute_batch()`
- use the batch-capable implementation there
- use vLLM when appropriate inside `execute_batch()`

For response generation:

- We should add `batch_query()` to the `Model` class.
- For HuggingFace models, `batch_query()` can use transformers batch inference.
- For Azure, Google, Anthropic, and other non-batch backends, `batch_query()` should fall back to a simple loop over `query()`.
- Defense batch methods should rely on this shared `Model.batch_query()` behavior unless they have a special reason not to.

## Defense Behavior Expectations

We should keep the current output style as much as possible.

Examples:

- detection defenses return `{"detect_flag": ...}`
- sanitization defenses return `{"cleaned_context": ...}`
- hybrid defenses may return both
- all `get_response*()` methods return a dict with `"response"`
- all batch version functions return a list of such dict

The goal here is not to redesign the result schema. The goal is to rewrite batch support in the class. Note that the current function-based batch versions are not good, we should rewrite them in a cleaner way.

## Changes to the Defense Package

### What should change

- add `execute_batch()` and `get_response_batch()` to `BaseDefense`
- move logic from `*_batch.py` into the corresponding defense classes, be careful as those legacy batch versions are not always good
- remove `DEFENSES_BATCH` from `piarena/defenses/__init__.py`
- keep `get_defense(name, config)` unchanged

### What should not change

- no large new API layer
- no new request/result wrapper classes
- no separate runtime framework
- no extra burden for adding a simple defense

## Changes to Individual Defenses

### Detection defenses

Examples:

- `datasentinel`
- `promptguard`
- `attentiontracker`
- `piguard`

Plan:

- keep current `execute()` behavior
- lazily load detector in the class
- add `execute_batch()` when detector batching is possible, otherwise also add it but use for loop to simulate
- add `get_response_batch()` calling `execute_batch()` and then `Model.batch_query()`, with provider-specific fallback handled inside `Model`

### Sanitization defenses

Examples:

- `datafilter`
- `pisanitizer`

Plan:

- same as detection-based defenses

### Hybrid defenses

Examples:

- `promptarmor`
- `promptlocate`

Plan:

- merge single and batch logic into the class
- preserve current return keys
- others same as above

### `secalign`

`secalign` is a special case because it owns the response model.

Plan:

- keep its current special response path
- add `get_response_batch()` inside the class
- lazily initialize the vLLM or model backend inside the defense object

## `strategy_search` Refactor

This is the main cleanup target.

### Current problem

`strategy_search` currently:

- creates a defense instance
- separately looks up a batch defense function
- contains defense-specific batch handling logic

### Desired change

`strategy_search` should only use the defense object.

Instead of:

- `get_defense(...)`
- `DEFENSES_BATCH[...]`

it should do:

- `defense = get_defense(...)`
- `defense.get_response_batch(...)`

and fall back automatically because every defense will support the batch API.

If `strategy_search` still needs the defense name for initialization templates or defense-specific attack heuristics, it should read `defense.name` from the defense object instead of receiving a separate `defense_name` argument.

### Expected simplification

This removes:

- the separate batch registry
- defense-batch lookup logic
- part of the defense-specific plumbing in the attack

It also makes it much easier to reuse in other projects, because the attack only depends on the defense object.

## `main_search.py` Simplification

`main_search.py` should stop knowing about batch-defense internals.

### Desired behavior

`main_search.py` should:

1. load attack
2. load defense
3. load backend LLM
4. load attacker LLM if needed
5. call the attack with the defense object
6. call defense methods normally during evaluation

### Specific cleanup for `strategy_search`

- pass `defense`, not a separate batch-defense function
- avoid passing a separate `defense_name`; when needed, read `defense.name` from the object
- let `strategy_search` use `get_response_batch()` as the standard path for candidate evaluation
- let single-item fallback happen automatically through the base defense batch methods
- remove any dependency on `DEFENSES_BATCH`

## AgentDojo Direction

This simpler design still helps AgentDojo.

We do not need a special new API for it.

AgentDojo can continue using the defense object directly, but benefit from:

- lazy initialization
- cleaner imports from installed `piarena`
- optional use of `execute_batch()` later if needed

The important part is that the defense object becomes the only place that knows how single and batch execution work.

## Migration Plan

### Step 1

Update `BaseDefense`:

- add `execute_batch()`
- add `get_response_batch()`
- make the default implementation a simple loop

### Step 2

Update each defense class:

- move lazy init into the class
- set model/vLLM fields to `None` in `__init__()`
- load on first use

### Step 3

For defenses with real batch support:

- move the current `*_batch.py` logic into `execute_batch()` or `get_response_batch()`

### Step 4

Remove:

- `DEFENSES_BATCH`
- direct batch-defense imports from `piarena/defenses/__init__.py`

### Step 5

Refactor `strategy_search` to depend only on the defense object.

### Step 6

Simplify `main_search.py` to pass the defense object and stop doing attack-specific defense wiring.

## Success Criteria

The refactor is successful when:

- every defense is still added as one class
- every defense supports both single and batch calls through the same object
- non-batchable defenses work through the default loop implementation
- batchable defenses override batch methods where useful
- models are loaded lazily, not at defense construction time
- `strategy_search` no longer uses a separate batch-defense registry
- `main_search.py` is simpler and does not special-case batch defense plumbing
