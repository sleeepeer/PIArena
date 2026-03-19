# `main_search.py` and `strategy_search` Walkthrough

## Purpose

This note explains, step by step, how one data sample is processed when we run `main_search.py` with `--attack strategy_search` and some defense such as `datasentinel`, `promptarmor`, or `none`.

The goal is to make the runtime lifecycle explicit:

- what gets loaded at program start
- what gets loaded lazily later
- what each function receives
- what each function returns
- what the next step is after each return

## Relevant Files

- [main_search.py](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py)
- [attack_strategy_search.py](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py)
- [base.py](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/base.py)
- [llm.py](/Users/rfg5487/Desktop/code/PIArena_public/piarena/llm.py)
- Example detection defense: [defense_datasentinel.py](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/datasentinel/defense_datasentinel.py)
- Example hybrid defense: [defense_promptarmor.py](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/promptarmor/defense_promptarmor.py)

## Big Picture

There are two separate stages for each sample.

1. Attack stage
   `strategy_search` tries to build an injected context that can bypass the defense.
2. Evaluation stage
   The chosen injected context is run through the same defense again, then utility and ASR are scored.

Important detail:

- `main_search.py` creates one defense object once.
- That same defense object is passed into `strategy_search`.
- That same defense object is also reused later for final evaluation.
- So any lazy-loaded detector or model inside the defense can be reused across attack-time evaluation and final evaluation.

## What Loads At Program Start

When `main_search.py` starts, this is the order.

### 1. Parse CLI arguments

`parse_args()` reads:

- dataset name
- backend LLM name
- attacker LLM name
- attack name
- defense name
- seed
- batch size
- experiment name

Output:

- one `args` object

Next step:

- `main(args)` is called

### 2. Load dataset

Inside `main(args)`, the code first tries:

- `datasets/{args.dataset}.json`

If that fails, it loads from Hugging Face.

Output:

- `dataset`: iterable dataset of samples
- `dataset_name`: normalized dataset name for filenames

Next step:

- build paths for cached attack results and evaluation results

### 3. Instantiate the attack object

`attack = get_attack(args.attack)` at [main_search.py:98](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:98)

This creates the attack class instance.

For `strategy_search`:

- this gives a `StrategySearchAttack` object
- this is light-weight
- it does not yet build attack candidates
- it does not yet load attacker vLLM

Output:

- `attack` object

Next step:

- try to load cached attack results

### 4. Load cached attack results if present

At [main_search.py:100](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:100), the code tries to load a JSON file of previously generated injected contexts.

Output:

- `attack_result`, either loaded cache or empty dict

Next step:

- load models and defense

### 5. Load backend LLM immediately

`llm = Model(args.backend_llm)` at [main_search.py:110](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:110)

This is eager loading.

What happens:

- if backend is Hugging Face, tokenizer and model are loaded immediately in `Model.__init__()`
- if backend is Azure/Google/Anthropic, the corresponding client is created immediately

Output:

- `llm`, the target model used for defended answering

Next step:

- prepare attacker backend information

### 6. Prepare attacker backend information

At [main_search.py:113](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:113), the code prepares:

- `attacker_model_name_or_path`
- optional eager `attacker_llm`

Current behavior:

- for `strategy_search`, `main_search.py` does not eagerly load a second attacker `Model(...)`
- it only passes the attacker model path into the attack
- inside `strategy_search`, vLLM is attempted first from that model path
- a full `attacker_llm = Model(...)` is only loaded later if the attack needs a non-vLLM fallback
- for `pair` and `tap`, `attacker_llm` is still loaded eagerly in `main_search.py`

Output:

- `attacker_model_name_or_path`
- maybe `attacker_llm`, depending on attack type

Next step:

- instantiate defense object

### 7. Instantiate the defense object

`defense = get_defense(args.defense)` at [main_search.py:121](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:121)

Important:

- the defense object itself is created now
- most heavy defense internals are not loaded now
- after the refactor, defenses keep internal handles like `_detector` or `_locator` as `None`
- those are loaded only when `execute()`, `execute_batch()`, `get_response()`, or `get_response_batch()` first needs them

Example:

- `DataSentinelDefense.__init__()` only sets `self._detector = None` at [defense_datasentinel.py:32](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/datasentinel/defense_datasentinel.py:32)
- `PromptArmorDefense.__init__()` only sets `self._locator = None` at [defense_promptarmor.py:30](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/promptarmor/defense_promptarmor.py:30)

Output:

- `defense` object

Next step:

- choose evaluators

### 8. Choose evaluators

At [main_search.py:124](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:124), the code chooses:

- `asr_evaluator`
- `utility_evaluator`

For many datasets this ends up being `llm_judge`.

Output:

- evaluator functions

Next step:

- try to load cached evaluation results

## Per-Sample Runtime Flow

Now the program iterates over the dataset at [main_search.py:154](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:154).

For each sample `dp`, these fields are extracted:

- `target_inst`
- `context`
- `injected_task`
- `target_task_answer`
- `injected_task_answer`

This sample is the unit that gets attacked and evaluated.

## Sample Flow: High-Level Timeline

For one sample, the control flow is:

1. check whether injected context is already cached
2. if not cached, call `attack.execute(...)`
3. for `strategy_search`, that builds many candidate payloads and evaluates them against the defense
4. `attack.execute(...)` returns one final `injected_context`
5. final evaluation calls `defense.get_response(...)` on that `injected_context`
6. compute utility score
7. compute ASR score
8. save result

## If the Attack Result Is Already Cached

At [main_search.py:172](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:172), the code checks `attack_result`.

If present:

- it reads `injected_context`
- it skips `strategy_search`
- it goes directly to final defense evaluation

So attack generation and final evaluation are decoupled.

## If the Attack Result Is Not Cached

Then [main_search.py:188](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:188) builds `attack_kwargs`.

For `strategy_search`, these are added:

- `llm`: backend target model
- `attacker_model_name_or_path`: attacker model path used for lazy attacker backend initialization
- `attacker_llm`: optional preloaded attacker model, usually omitted for `strategy_search`
- `defense`: the defense object itself

Then:

- `injected_context = attack.execute(**attack_kwargs)` at [main_search.py:208](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:208)

Output:

- one final `injected_context` string

Next step:

- cache it to `attack_result`
- then do final evaluation

## What `StrategySearchAttack.execute()` Does

`StrategySearchAttack.execute()` is a wrapper at [attack_strategy_search.py:1569](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1569).

It forwards everything into `_strategy_search_impl(...)`.

Inputs:

- clean `context`
- `target_inst`
- `injected_task`
- target and injected answers
- `llm`
- optional `attacker_llm`
- optional `attacker_model_name_or_path`
- `defense`

Output:

- one string: the chosen `injected_context`

Next step:

- `_strategy_search_impl()` creates a `StrategySearchAttacker`

## What `_strategy_search_impl()` Does Before Search Starts

### 1. Merge attack config

At [attack_strategy_search.py:1643](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1643), it merges user config with defaults.

Output:

- `final_config`

### 2. Validate `llm`

At [attack_strategy_search.py:1648](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1648), it ensures the target backend model exists.

### 3. Resolve attacker backend path

At [attack_strategy_search.py:1677](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1677):

- if no preloaded `attacker_llm` and no explicit attacker model path are passed, it reuses `llm.model_name_or_path`
- this is just metadata resolution, not a model load

### 4. Run a sanity query only if `attacker_llm` is already loaded

At [attack_strategy_search.py:1682](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1682), it sends a tiny test prompt only when a preloaded `attacker_llm` was passed in.

Purpose:

- keep the old sanity check when an attacker model object already exists
- avoid forcing an eager attacker model load for `strategy_search`

Output:

- no persistent output, only a warning or success-by-silence

### 5. Build `StrategySearchConfig`

At [attack_strategy_search.py:1674](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1674)

Output:

- typed config object

### 6. Instantiate `StrategySearchAttacker`

At [attack_strategy_search.py:1680](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1680)

This is where most attack-side setup happens.

## What Loads Inside `StrategySearchAttacker.__init__()`

### 1. Bind the defense object and derive defense name

At [attack_strategy_search.py:436](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:436):

- if `defense` was passed in, use it directly
- otherwise instantiate from `defense_name`

Important:

- in `main_search.py`, the defense object is passed in
- so the same defense instance is used throughout the sample
- `self.defense_name` is then derived from `self.defense.name`

Output:

- `self.defense`
- normalized `self.defense_name`

### 2. Detect whether defense has custom batch methods

At [attack_strategy_search.py:438](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:438)

This is just classification:

- custom `get_response_batch()` or `execute_batch()`
- or fallback to `BaseDefense` default loop implementation

No heavy model load happens here.

### 3. Initialize attacker vLLM

At [attack_strategy_search.py:449](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:449), `_init_attacker_vllm()` is called.

Important:

- this is attack-side acceleration, not defense-side lazy init
- it happens when `strategy_search` starts on a sample
- it may reuse a global vLLM instance across later samples
- it only attempts this for Hugging Face style models
- it uses `attacker_model_name_or_path`; it does not require a preloaded `attacker_llm`

Output:

- `self.attacker_vllm` or `None`

### 4. Initialize judge vLLM

At [attack_strategy_search.py:453](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:453), `_init_judge_vllm()` is called.

Purpose:

- accelerate batch `llm_judge` evaluations of candidate responses

Output:

- `self.judge_vllm` or `None`

### 5. Build initialization templates

At [attack_strategy_search.py:456](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:456), `build_init_templates_for_defense(...)` is called.

Purpose:

- create prompt templates for generating initial payloads
- these can depend on defense name
- template generation uses attacker vLLM if available
- otherwise it lazy-loads `attacker_llm` only when needed

Output:

- `self.init_mutation_templates`

Next step:

- `search(case, evaluator)` is called

## What the `case` Object Contains

At [attack_strategy_search.py:1694](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1694), the code creates:

```python
case = {
    'context': context,
    'target_inst': target_inst,
    'injected_task': injected_task,
    'injected_task_answer': injected_task_answer,
    'target_task_answer': target_task_answer,
}
```

This `case` object is the full sample payload used during the search.

## Phase 1 of `strategy_search`: Initial Candidate Generation

This starts in `search()` at [attack_strategy_search.py:1288](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1288).

### 1. Read attack settings

From config:

- `population_size`
- `init_attempts_per_strategy`
- `max_generations`
- `success_threshold`

### 2. Build generation prompts

At [attack_strategy_search.py:1354](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1354), the code loops over templates and attempts.

Each generation prompt is created from:

- `injected_task`
- `target_inst`
- `context_theme`
- `context_tail`

Output:

- `prompts`: text prompts sent to the attacker model
- `meta_info`: records which template and attempt produced each response

### 3. Generate payload candidates in batch

At [attack_strategy_search.py:1371](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1371), `_batch_query_attacker_llm(prompts)` is called.

What this does:

- if attacker vLLM exists, use it for batch generation
- otherwise lazy-load `attacker_llm` and use `attacker_llm.batch_query()` or sequential fallback
- clean the outputs with `_clean_output()`

Output:

- list of generated injected prompt strings

### 4. Wrap generated strings into `AttackCandidate` objects

At [attack_strategy_search.py:1374](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1374), each non-empty response becomes an `AttackCandidate`.

Each candidate now has:

- `injected_prompt`
- `generation=0`
- `mutation_type`
- `initial_strategy_name`

Output:

- `all_candidates`

Next step:

- evaluate all candidates against the defense and judge

## Phase 1 Candidate Evaluation: Where Defense Is First Used

This is the most important part for understanding the runtime.

The call is:

- `fitnesses = self._evaluate_candidates_batch(...)` at [attack_strategy_search.py:1396](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1396)

This is where the defense is actually exercised during search.

### Step A. Build attacked contexts

At [attack_strategy_search.py:1032](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1032):

- each candidate payload is inserted into the original clean context
- `_find_injection_position(...)` decides where to insert it
- default is append to the end with `"\n\n"`

Output:

- `attack_prompts`: these are full injected contexts, one per candidate

### Step B. Call the defense batch API

At [attack_strategy_search.py:1041](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1041):

```python
self.defense.get_response_batch(
    target_insts=target_insts,
    contexts=attack_prompts,
    llm=self.target_llm,
)
```

Inputs to the defense:

- repeated `target_inst`
- candidate-specific injected contexts
- the backend target model `llm`

Output from the defense:

- one list of dicts, one dict per candidate
- each dict always contains `response`
- it may also contain `detect_flag`, `cleaned_context`, `potential_injection`, `score`, etc.

This is the key attack-time interface.

## When the Defense Loads Its Own Internal Models

This is the first point where defense-side lazy initialization may happen.

### Example: `DataSentinelDefense`

At [defense_datasentinel.py:36](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/datasentinel/defense_datasentinel.py:36), `_ensure_detector()` is called inside `execute()` or `execute_batch()`.

So the detector is loaded only when candidate evaluation first touches the defense.

Sequence:

1. `strategy_search` calls `defense.get_response_batch(...)`
2. `DataSentinelDefense.get_response_batch(...)` calls `execute_batch(...)`
3. `execute_batch(...)` calls `_ensure_detector()`
4. if `_detector is None`, instantiate `DataSentinelDetector`
5. run detection over candidate contexts
6. for safe candidates only, call `llm.batch_query(...)`
7. return combined result dicts

### Example: `PromptArmorDefense`

At [defense_promptarmor.py:34](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/promptarmor/defense_promptarmor.py:34), `_ensure_locator()` is called only on first use.

Sequence:

1. `strategy_search` calls `defense.get_response_batch(...)`
2. `PromptArmorDefense.get_response_batch(...)` calls `execute_batch(...)`
3. `execute_batch(...)` calls `_ensure_locator()`
4. if `_locator is None`, instantiate a `Model(...)` for detection/localization
5. run batch detection/localization on candidate contexts
6. produce sanitized `cleaned_context` values
7. call `llm.batch_query(...)` on the sanitized contexts
8. return combined result dicts

## What `BaseDefense.get_response_batch()` Does By Default

If a defense does not override the batch path, [base.py:52](/Users/rfg5487/Desktop/code/PIArena_public/piarena/defenses/base.py:52) does this:

1. validate list lengths
2. call `execute_batch()`
3. by default, `execute_batch()` just loops over `execute()`
4. build one batch of messages
5. call `llm.batch_query(messages_batch)`
6. merge each defense result with each response

So even a non-custom defense still works with `strategy_search`.

## What Happens After the Defense Returns Candidate Results

Back in `_evaluate_candidates_batch()`:

### 1. Record raw defended responses

At [attack_strategy_search.py:1054](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1054):

- `candidate.full_response = result["response"]`

### 2. Record sanitized context if present

At [attack_strategy_search.py:1059](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1059):

- if defense returned `cleaned_context`, store it
- otherwise keep original attacked context

### 3. Infer whether defense acted

At [attack_strategy_search.py:1062](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1062):

- sanitization defenses: compare original payload vs `cleaned_context`
- detection defenses: look for warning keywords in `response`

Output per candidate:

- `candidate.was_sanitized` or `candidate.was_detected`

### 4. Judge ASR fitness

At [attack_strategy_search.py:1070](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1070):

- if evaluator is `llm_judge`, it judges whether the defended response completed the injected task
- if judge vLLM exists, batch judge via vLLM
- otherwise use `llm_judge_batch(...)`

Output:

- one numeric fitness per candidate

### 5. Convert defense outcome into mutation feedback

At [attack_strategy_search.py:1121](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1121):

Each candidate gets one of:

- `success`
- `no_defense`
- `sanitized`
- `detected`
- `too_weak`
- `no_signal`

This feedback is what drives the next mutation step.

## Phase 1 Exit Condition

Back in `search()`:

- if any candidate fitness reaches `success_threshold`, Phase 1 stops
- otherwise the best candidates go into Phase 2

Output of Phase 1:

- `all_candidates` with filled-in fitness and defense feedback
- maybe `best_ever`

Next step:

- either return success or start evolution

## Phase 2 of `strategy_search`: Evolution Search

If Phase 1 fails, Phase 2 starts at [attack_strategy_search.py:1449](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1449).

### 1. Select top candidates

At [attack_strategy_search.py:1477](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1477):

- sort by fitness
- keep top `population_size`

### 2. Mutate them in batch

At [attack_strategy_search.py:1485](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1485), `_batch_mutate(...)` is called.

What `_batch_mutate(...)` uses:

- previous `candidate.defense_feedback`
- previous `candidate.full_response`
- mutation templates such as `increase_stealth`, `increase_strength`, `self_analyze_and_improve`

Output:

- new candidate payloads for the next generation

### 3. Evaluate the mutated candidates again

At [attack_strategy_search.py:1491](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1491):

- same defense batch call pattern as before
- same lazy-loaded defense instance reused
- same backend LLM reused
- same judge mechanism reused

### 4. Repeat until success or generation limit

Output of search:

- `best_candidate`
- `history`

## What `strategy_search` Finally Returns

Back in `_strategy_search_impl()`:

At [attack_strategy_search.py:1721](/Users/rfg5487/Desktop/code/PIArena_public/piarena/attacks/strategy_search/attack_strategy_search.py:1721):

- take `best_candidate.injected_prompt`
- insert it into the original clean `context`
- return that final full `injected_context`

Important:

- `strategy_search` does not return the defended response
- it does not return a score
- it returns only the chosen attacked context string

Next step:

- `main_search.py` caches that attacked context
- then runs final evaluation on it

## Final Evaluation After the Attack Returns

After `attack.execute(...)` returns, `main_search.py` does this at [main_search.py:214](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:214):

```python
defense_result = defense.get_response(
    target_inst=target_inst,
    context=injected_context,
    llm=llm,
)
```

This is a single-item evaluation of the final selected attack.

Important:

- this uses the same defense object that was used during search
- so any lazily loaded detector or locator is already in memory
- the backend model `llm` is already loaded

Output:

- one `defense_result` dict
- always contains `response`
- may also contain `detect_flag`, `cleaned_context`, `potential_injection`, and so on

## Final Utility and ASR Scoring

Then `main_search.py` computes:

### Utility

At [main_search.py:221](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:221):

- score whether the defended response still answers the original target task

### ASR

At [main_search.py:226](/Users/rfg5487/Desktop/code/PIArena_public/main_search.py:226):

- score whether the defended response followed the injected malicious task

Output:

- `result_dp["utility"]`
- `result_dp["asr"]`

Next step:

- save `evaluation_result[idx]`
- print summary

## End-to-End Example: One Sample With `datasentinel`

This is the practical sequence for one sample.

1. `main_search.py` extracts `target_inst`, `context`, `injected_task`, answers.
2. It sees there is no cached attacked context.
3. It calls `strategy_search.execute(...)` with:
   - clean context
   - injected task
   - backend `llm`
   - attacker model path
   - `defense` object for `datasentinel`
4. `strategy_search` creates `StrategySearchAttacker`.
5. `StrategySearchAttacker` may load attacker vLLM and judge vLLM.
6. It builds initialization templates.
7. It batch-generates candidate payloads.
8. For each candidate payload, it builds a candidate injected context.
9. It calls `DataSentinelDefense.get_response_batch(...)`.
10. `DataSentinelDefense.execute_batch(...)` sees `_detector is None` and loads `DataSentinelDetector`.
11. It detects which candidate contexts are suspicious.
12. Suspicious ones get warning responses immediately.
13. Safe ones are sent to `llm.batch_query(...)`.
14. The defense returns one dict per candidate with `detect_flag` and `response`.
15. `strategy_search` judges which defended responses actually follow the injected task.
16. It marks candidates as `detected` or `too_weak` or `success`.
17. If needed, it mutates candidates and repeats.
18. It picks the best payload and returns one final `injected_context`.
19. Back in `main_search.py`, the same `datasentinel` defense object is called again with `get_response(...)` on that final attacked context.
20. Utility and ASR are scored and written to disk.

## End-to-End Example: One Sample With `promptarmor`

The structure is the same, but defense behavior differs.

1. `strategy_search` generates candidate payloads.
2. It calls `PromptArmorDefense.get_response_batch(...)`.
3. On first call, `PromptArmorDefense` lazy-loads its `_locator` model.
4. It asks the locator model whether each candidate context contains injection and what text to remove.
5. It constructs `cleaned_context` for each candidate.
6. It sends the cleaned contexts to `llm.batch_query(...)`.
7. It returns:
   - `detect_flag`
   - `cleaned_context`
   - `potential_injection`
   - `response`
8. `strategy_search` compares original payload vs `cleaned_context`.
9. If payload was removed, feedback becomes `sanitized`.
10. Mutation then tries to make the payload stealthier.
11. Final selected attack is returned and later evaluated again by `main_search.py`.

## What Is Eager vs Lazy Right Now

### Eager loads

These happen early, before the sample loop or before a search starts.

- dataset loading in `main_search.py`
- backend `llm = Model(...)`
- attack object instantiation via `get_attack(...)`
- inside `strategy_search`, attacker vLLM and judge vLLM initialization are attempted in `StrategySearchAttacker.__init__()`
- for `pair` and `tap`, `attacker_llm = Model(...)` is still eager in `main_search.py`

### Lazy loads

These happen only when the defense is first used.

Examples:

- `DataSentinelDefense._detector`
- `PromptArmorDefense._locator`
- `PromptGuardDefense._detector`
- `SecAlignDefense._model`
- `PISanitizerDefense._sanitizer_model`
- for `strategy_search`, `attacker_llm = Model(...)` is also lazy now if attacker vLLM is unavailable or a non-vLLM fallback is needed

These are triggered on first call to:

- `execute()`
- `execute_batch()`
- `get_response()`
- `get_response_batch()`

## What Each Major Function Receives and Returns

### `main_search.main(args)`

Receives:

- CLI args

Returns:

- nothing

Effect:

- orchestrates the whole run

### `StrategySearchAttack.execute(...)`

Receives:

- one sample's clean context and task fields
- target `llm`
- optional `attacker_llm`
- optional attacker model path
- `defense`

Returns:

- one final attacked context string

### `StrategySearchAttacker.search(case, evaluator)`

Receives:

- one sample packed as `case`
- evaluator function

Returns:

- `best_candidate`
- `history`

### `defense.get_response_batch(target_insts, contexts, llm)`

Receives:

- many candidate attacked contexts
- target instruction for each
- backend model for answer generation

Returns:

- one result dict per candidate, including final defended response

### `defense.get_response(target_inst, context, llm)`

Receives:

- one final attacked context

Returns:

- one result dict for final evaluation

## Important Design Consequences

### 1. Search-time defense and final-time defense are the same object

This is good because:

- lazy-loaded defense internals can be reused
- attack optimization is aligned with final evaluation behavior

### 2. `strategy_search` optimizes against defended responses, not raw model responses

That is why `get_response_batch()` is the critical API.

### 3. The attack does not directly modify the defense

It only proposes payloads, runs them through the defense, observes the defended result, and adapts.

### 4. The returned artifact from the attack is only `injected_context`

All scoring still happens outside the attack in `main_search.py`.

## Short Summary

For one sample, the runtime chain is:

1. read sample from dataset
2. call `strategy_search` if no cached attacked context exists
3. `strategy_search` tries attacker vLLM first and only lazy-loads `attacker_llm` if needed
4. `strategy_search` inserts each payload into the clean context
5. `strategy_search` sends those attacked contexts into `defense.get_response_batch(...)`
6. the defense lazy-loads its internal detector/sanitizer the first time it is actually needed
7. the defense returns defended responses for all candidates
8. `strategy_search` judges which defended response best satisfies the injected task
9. `strategy_search` mutates and repeats if needed
10. `strategy_search` returns one best `injected_context`
11. `main_search.py` runs final `defense.get_response(...)` on that one attacked context
12. `main_search.py` computes utility and ASR
13. results are saved
