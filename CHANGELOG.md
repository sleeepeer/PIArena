# Changelog

This project does not currently use tagged releases consistently, so this changelog is maintained as a running record of notable repository-level changes.

## Unreleased

### Added

- Added unified binary LLM-judge for non-KC splits: `llm_judge_utility` (uses `target_task_answer`) and `llm_judge_asr` (uses `injected_task_answer`) in [`piarena/evaluations/llm_judge.py`](piarena/evaluations/llm_judge.py); [`main.py`](main.py) now routes all non-KC splits to this judge pair. Validated against synthetic 4-case ground truth on Qwen3.6-27B: 100% utility accuracy (target_only / both / refusal) and 100% ASR accuracy (target_only / injected_only); residual ~10% disagreement is generator drift, not judge error.
- Added batched judge variants `llm_judge_utility_batch` and `llm_judge_asr_batch` in [`piarena/evaluations/llm_judge.py`](piarena/evaluations/llm_judge.py); both call `llm.batch_query` when available and fall back to a per-sample loop otherwise.
- Added [`scripts/generate_injected_task_answers.py`](scripts/generate_injected_task_answers.py) to populate the previously-empty `injected_task_answer` field across all 1700 non-KC samples using Qwen3.6-27B; output goes to `datasets_debug/<split>.json` and is published to the `sleeepeer/PIArena_debug` HF repo (separate from the production `sleeepeer/PIArena`).
- Added vLLM backend in [`piarena/llm.py`](piarena/llm.py); tunables (`tp_size`, `gpu_mem`, `max_model_len`, `max_num_seqs`, `enforce_eager`) are constructor kwargs with defaults, surfaced as CLI flags in [`main_vllm.py`](main_vllm.py).
- Added [`main_vllm.py`](main_vllm.py): a vLLM-only batched runner that mirrors `main.py`'s pipeline but executes attacks per chunk and runs defense + utility/ASR judges in batched calls (`defense.get_response_batch`, `llm_judge_utility_batch`, `llm_judge_asr_batch`). Supports `--judge_llm same` to share the engine with the backend LLM.
- Added [`scripts/setup_piarena_vllm_env.sh`](scripts/setup_piarena_vllm_env.sh) — canonical env build for Delta-AI Grace Hopper (Python 3.12, vllm 0.19.1, torch 2.10).
- Added root-level guidance in [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md) requiring implementation plans to be written under `plans/`.
- Added root docs trees for supported attacks and defenses under [`docs/attacks/`](docs/attacks/) and [`docs/defenses/`](docs/defenses/).
- Added a compact public docs page at [`docs/extending.md`](docs/extending.md) covering how to add new attacks and defenses.
- Added a docs migration and standardization plan at [`plans/docs-root-migration-and-standardization.md`](plans/docs-root-migration-and-standardization.md).
- Added merged AgentDyn benchmark assets to the vendored [`agents/agentdojo/`](agents/agentdojo/) tree, including new `shopping`, `github`, and `dailylife` suites plus the dynamic tool implementations they require.

### Changed

- Migrated the website to consume markdown directly from root [`docs/`](docs/) instead of maintaining a duplicate `website/docs/` tree.
- Reorganized public docs into a smaller structure centered on:
  - [`docs/getting-started.md`](docs/getting-started.md)
  - [`docs/evaluation.md`](docs/evaluation.md)
  - [`docs/attacks/`](docs/attacks/)
  - [`docs/defenses/`](docs/defenses/)
  - [`docs/extending.md`](docs/extending.md)
- Standardized attack and defense docs so each method page focuses on a brief introduction, source links, usage, behavior, and parameters.
- Updated the website docs sidebar in [`website/app.jsx`](website/app.jsx) to discover pages automatically from root docs and render a cleaner docs tree.
- Fixed inline docs link rendering in [`website/app.jsx`](website/app.jsx) so markdown links with code-formatted labels render correctly.
- Updated [`website/vite.config.js`](website/vite.config.js) to allow loading markdown from the repository root during website builds.
- Updated repository guidance in [`README.md`](README.md), [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md), and [`website/AGENTS.md`](website/AGENTS.md) to reflect the root-docs workflow.
- Varied docs and README examples so they do not repeatedly use `pisanitizer` as the default example defense.
- Expanded the vendored [`agents/agentdojo/`](agents/agentdojo/) integration so the existing PIArena defense adapter works for both classic AgentDojo suites and the merged AgentDyn suites.
- Updated [`main_agentdojo.py`](main_agentdojo.py) and [`scripts/run_agentdojo.py`](scripts/run_agentdojo.py) so one runner can execute classic AgentDojo suites, merged AgentDyn suites, PIArena defenses, and benchmark-native defenses from the same vendored benchmark tree.

### Changed

- Refactored [`piarena/llm.py`](piarena/llm.py) to share chat-template handling between HF and vLLM paths, force `enable_thinking=False` for Qwen3+ models when supported, and post-strip any leaked `</think>` block (reasoning text now exposed via `model.last_reasoning`).
- Pinned working dependency set in [`requirements.txt`](requirements.txt) for the vllm 0.19.1 / torch 2.10 / transformers 4.57 stack on aarch64 Grace Hopper; documented module loads.

### Removed

- Removed dead legacy batch defense modules (`defense_none_batch.py`, `defense_promptguard_batch.py`, `defense_piguard_batch.py`, `defense_datasentinel_batch.py`) — these were never imported anywhere; class-based `BaseDefense.execute_batch` / `get_response_batch` is the supported batch path used by `strategy_search` and standard evals.
- Removed the duplicate public docs copies from `website/docs/`.
- Removed the older flat public docs pages that were replaced by the new grouped attack and defense trees.
