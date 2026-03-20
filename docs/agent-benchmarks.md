---
title: Agent Benchmarks
description: Evaluate prompt injection defenses on tool-integrated LLM agent benchmarks.
order: 50
---

# Agent Benchmarks

PIArena supports evaluation on agent benchmarks for tool-integrated LLM agents: **InjecAgent** and **AgentDojo**.

## Setup

```bash
# AgentDojo
cd agents/agentdojo && pip install -e . && cd ../..
```

## InjecAgent

[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) evaluates indirect prompt injection in API-calling agents with 1,054 test cases across Direct Harm (DH) and Data Stealing (DS) attack types.

```bash
python main_injecagent.py --model meta-llama/Llama-3.1-8B-Instruct --defense none
```

**Metrics:**
- `Valid Rate` — % of outputs with valid ReAct format
- `ASR-valid` — Attack success rate among valid outputs
- `ASR-all` — Attack success rate across all cases (invalid = failure)

Results saved to: `results/agent_evaluations/injecagent/`

## AgentDojo

[AgentDojo](https://github.com/ethz-spylab/agentdojo) evaluates prompt injection with 97 user tasks across multiple attack types and task suites.

```bash
# Benign utility (no attack)
python main_agentdojo.py --model gpt-4o-2024-05-13 --no_attack

# With attack + defense
python main_agentdojo.py --model gpt-4o-2024-05-13 --attack tool_knowledge --defense datafilter

# Specific suite
python main_agentdojo.py --model gpt-4o-2024-05-13 --attack tool_knowledge --suite workspace
```

**Attack Types:** `direct`, `important_instructions`, `tool_knowledge`, `injecagent`

**Metrics:**
- `Benign Utility` — Task completion % without attacks
- `Utility Under Attack` — Task completion % with attacks
- `ASR` — Attack success rate per attack type

Results saved to: `results/agent_evaluations/agentdojo/{name}/`
