# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact React webpage snapshot, not a full app scaffold. The main UI lives in `app.jsx` (single entry component plus helpers and page sections). Static benchmark inputs are stored in `data/` (`metadata.json`, `results.json`). Markdown content rendered in the docs tab lives in `docs/` (for example `docs/getting-started.md`, `docs/tools.md`).

Keep data and docs changes isolated when possible so UI regressions are easier to review.

## Build, Test, and Development Commands

No `package.json` or local build/test scripts are committed in this directory. Contributors typically edit this folder as source content for a host React/Vite app.

Useful local checks:

- `rg --files` — list tracked files quickly
- `node -e "JSON.parse(require('fs').readFileSync('data/metadata.json','utf8'))"` — validate JSON syntax
- `git diff -- app.jsx docs/ data/` — review changes before commit

If you run this in a host app, ensure it supports raw Markdown imports (e.g., `?raw`) and has `react` plus `lucide-react` installed.

## Coding Style & Naming Conventions

Use 2-space indentation and preserve the existing React style in `app.jsx`: functional components, small utility helpers, and semicolons. Prefer clear camelCase names for functions/variables (`computeDefenseLeaderboard`) and PascalCase for components (`DocsPage`, `Footer`).

For content files, use lowercase kebab-case names in `docs/` and stable JSON keys in `data/`. Update import paths in `app.jsx` whenever docs filenames change.

## Testing Guidelines

There is no automated test suite in this directory. Validate changes by rendering the page in the host app and checking:

- leaderboard tables load from `data/results.json`
- docs sections render expected Markdown content
- tab navigation works on desktop and mobile widths

For data edits, confirm JSON parses and key names match lookups used in `app.jsx`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative commit messages (for example: `remove Next.js page`, `fix code highlight`). Follow that pattern and keep commits scoped.

PRs should include a brief summary, impacted paths (for example `app.jsx`, `docs/`, `data/`), and screenshots/GIFs for UI changes. Call out any renamed docs files or data schema changes explicitly.
