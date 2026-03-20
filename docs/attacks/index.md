---
title: Attacks
slug: attacks
category: attacks
---

# Attacks

PIArena supports simple heuristic attacks, optimization-based attacks, and search-based attacks.

## Supported Attacks

- [`none`](/docs/attacks/none): no attack, useful as a utility baseline
- [`direct`](/docs/attacks/direct): directly inserts the injected task into the context
- [`ignore`](/docs/attacks/ignore): prefixes the injected task with "Ignore previous instructions"
- [`completion`](/docs/attacks/completion): adds a completion-style prefix before the injected task
- [`character`](/docs/attacks/character): inserts the injected task with a newline-style formatting change
- [`combined`](/docs/attacks/combined): combines multiple simple prompt-injection cues
- [`nanogcg`](/docs/attacks/nanogcg): gradient-based prompt optimization
- [`pair`](/docs/attacks/pair): iterative refinement with an attacker model
- [`tap`](/docs/attacks/tap): tree-based search and pruning
- [`strategy_search`](/docs/attacks/strategy-search): defense-aware evolutionary search

## How To Choose

- Start with `direct` or `combined` when you want a quick baseline.
- Use `pair`, `tap`, or `strategy_search` when you want stronger adaptive attacks.
- Use `nanogcg` when you specifically want optimization-based prompt search.

If you are pairing an attack with a specific defense, read the matching method pages under [Defenses](/docs/defenses) too.
