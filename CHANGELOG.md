# Changelog

This project does not currently use tagged releases consistently, so this changelog is maintained as a running record of notable repository-level changes.

## Unreleased

### Added

- Added a defense batch refactor plan in [docs/defense-batch-refactor-plan.md](docs/defense-batch-refactor-plan.md).
- Added a runtime walkthrough for `main_search.py` and `strategy_search` in [docs/main-search-strategy-search-walkthrough.md](docs/main-search-strategy-search-walkthrough.md).
- Added this repository-level `CHANGELOG.md`.

### Changed

- Fixed editable package installation by switching `pyproject.toml` to the standard `setuptools.build_meta` backend and adding `wheel` as a build requirement.
- Moved batch defense support into defense classes through `BaseDefense.execute_batch()` and `BaseDefense.get_response_batch()`.
- Added `Model.batch_query()` with provider-aware fallback behavior.
- Removed the separate `DEFENSES_BATCH` registry from `piarena.defenses`.
- Refactored `strategy_search` to evaluate candidates through the defense object rather than separate batch defense functions.
- Simplified `main_search.py` so `strategy_search` receives the defense object directly.
- Simplified `strategy_search` so `defense_name` is optional when a defense object is provided.
- Simplified `strategy_search` so attacker model loading is lazy when possible:
  - attacker vLLM is attempted from the attacker model path first
  - a full `attacker_llm = Model(...)` is loaded only when a non-vLLM fallback is needed

### Documentation

- Updated `README.md`, `AGENTS.md`, and `CLAUDE.md` to reflect the class-owned batch defense API, the simplified `strategy_search` path, and the editable install guidance.
