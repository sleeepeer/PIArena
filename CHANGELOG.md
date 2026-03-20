# Changelog

This project does not currently use tagged releases consistently, so this changelog is maintained as a running record of notable repository-level changes.

## Unreleased

### Added

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

### Removed

- Removed the duplicate public docs copies from `website/docs/`.
- Removed the older flat public docs pages that were replaced by the new grouped attack and defense trees.
