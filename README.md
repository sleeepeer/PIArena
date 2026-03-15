<p align="center">
  <h1 align="center">
    <img width="420px" src="assets/icon.png" alt="PIArena"/>
  </h1>
</p>

<p align="center">
  <h3 align="center">A Platform for Prompt Injection Evaluation</h3>
</p>

<p align="center">
   <a href="https://piarena.vercel.app" target="_blank">
    <img src="https://img.shields.io/badge/🌐-Project Page-green?style=flat-rounded" alt="ProjectPage">
  </a>
  <a href="https://huggingface.co/datasets/sleeepeer/PIArena" target="_blank">
    <img src="https://img.shields.io/badge/🤗-HuggingFace Dataset-yellow?style=flat-rounded" alt="HuggingFace">
  </a>
  <a href="https://piarena.vercel.app/#/leaderboard" target="_blank">
    <img src="https://img.shields.io/badge/📊-LeaderBoard-blue?style=flat-rounded" alt="LeaderBoard">
  </a>
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/📄-Paper (Coming soon)-red?style=flat-rounded" alt="Paper">
  </a>
  <a href="https://github.com/sleeepeer/PIArena">
    <img src="https://img.shields.io/badge/⭐-Give PIArena a Star-gold?style=flat&logo=github" alt="Star">
  </a>
</p>

**PIArena** is an **easy-to-use toolbox** and also a **comprehensive benchmark** for researching prompt injection attacks and defenses. It provides:

* **Plug-and-play Attacks & Defenses** – Easily integrate state-of-the-art defenses into your workflow to protect your LLM system against prompt injection attacks. You can also play with existing attack strategies to perform a better research.
* **Systematic Evaluation Benchmark** – End-to-end evaluation pipeline enables you to easily evaluate attacks / defenses on various datasets.
* **Add Your Own** – You can also easily integrate your own attack or defense into our benchmark to systematically assess how well it perform.


## Table of Contents

* [📝 Quick Start](#-quick-start)
	* [⚙️ Installation](#-installation)
	* [📌 Ready-to-use Tools](#-ready-to-use-tools)
	* [📈 Run Evaluation](#-run-evaluation)
	* [🔍 Search-based Attacks](#-search-based-attacks)
	* [🤖 Agent Benchmarks](#-agent-benchmarks)
* [🙋🏻‍♀️ Add your own attacks / defenses](#-add-your-own-attacks--defenses)
<!-- * [🔍 Attack & Defense Details](#-attack--defense-details)
* [🤝 Contribution Guide](#-contribution-guide) -->

## 📝 Quick Start

### ⚙️ Installation

Clone the project and setup python environment:

```bash
git clone git@github.com:sleeepeer/PIArena.git
cd PIArena
conda create -n piarena python=3.10 -y
conda activate piarena
pip install -r requirements.txt
```

**Login to HuggingFace** 🤗 with your HuggingFace Access Token, you can find it at [this link](https://huggingface.co/settings/tokens):

```bash
huggingface-cli login
```

### 📌 Ready-to-use Tools

You can simply import attacks and defenses and integrate them into your own code. Please see details in [Document](https://piarena.vercel.app/#/docs/attacks-and-defenses).

```python
from piarena.attacks import get_attack
from piarena.defenses import get_defense
from piarena.llm import Model

llm = Model("Qwen/Qwen3-4B-Instruct-2507")
defense = get_defense("pisanitizer")
attack = get_attack("combined")
```

### 📈 Run Evaluation

Use `main.py` to run the benchmark:

```bash
# Using CLI arguments
python main.py --dataset squad_v2 --attack direct --defense none

# Using a YAML config file
python main.py --config configs/experiments/my_experiment.yaml

# Run many experiments in parallel across GPUs (edit scripts/run.py to configure)
python scripts/run.py
```
**Available Datasets:** Please see [HuggingFace/PIArena](https://huggingface.co/datasets/sleeepeer/PIArena).

**Available Attacks:**
- `none` - No attack (baseline)
- `direct` - Directly attack using injected prompt (default)
- `combined` - [Formalizing and Benchmarking Prompt Injection Attacks and Defenses](https://www.usenix.org/system/files/usenixsecurity24-liu-yupei.pdf)
- `ignore` - [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527)
- `completion` - [Prompt injection attacks against GPT-3](https://simonwillison.net/2022/Sep/12/prompt-injection/)
- `character` - [Delimiters won’t save you from prompt injection
](https://simonwillison.net/2023/May/11/delimiters-wont-save-you/)
- `nanogcg` - [GCG](https://arxiv.org/abs/2307.15043) and [nanoGCG](https://github.com/GraySwanAI/nanoGCG)
- `tap` - [TAP: A Query-Efficient Method for Jailbreaking Black-Box LLMs](https://github.com/RICommunity/TAP)
- `pair` - [PAIR: Jailbreaking black box large language models in twenty queries](https://github.com/patrickrchao/JailbreakingLLMs)
- `strategy_search` - Strategy search attack based on defense feedback introduced in [PIArena]().

**Available Defenses:**
- `none` - No defense (baseline, default)
- `pisanitizer` - [PISanitizer: Preventing Prompt Injection to Long-Context LLMs via Prompt Sanitization](https://arxiv.org/abs/2511.10720)
- `attentiontracker` - [Attention Tracker: Detecting Prompt Injection Attacks in LLMs](https://arxiv.org/abs/2411.00348)
- `promptguard` - [Meta Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M)
- `promptlocate` - [PromptLocate: Localizing Prompt Injection Attacks](https://arxiv.org/abs/2510.12252)
- `promptarmor` - [PromptArmor: Simple yet Effective Prompt Injection Defenses](https://arxiv.org/abs/2507.15219)
- `datasentinel` - [DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks](https://arxiv.org/abs/2504.11358)
- `datafilter` - [Defending Against Prompt Injection with DataFilter](https://arxiv.org/abs/2510.19207)
- `piguard` - [PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free](https://arxiv.org/abs/2410.22770)
- `secalign` - [SecAlign: Defending Against Prompt Injection with Preference Optimization](https://arxiv.org/abs/2410.05451) (uses [Meta-SecAlign](https://arxiv.org/abs/2507.02735) model)

<!-- You can go to [this link]() to see all supported datasets, attacks and defenses. -->

### 🔍 Search-based Attacks

PIArena supports search-based attacks (PAIR, TAP, Strategy Search) that iteratively refine prompt injections using an attacker LLM. Use `main_search.py` for these attacks:

```bash
# --attack can be tap, pair, strategy_search
python main_search.py --dataset squad_v2 --attack strategy_search --defense pisanitizer \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 --attacker_llm Qwen/Qwen3-4B-Instruct-2507

# Run many search experiments in parallel (edit scripts/run_search.py to configure)
python scripts/run_search.py
```

See [Search-based Attacks](https://piarena.vercel.app/#/docs/strategy-search) for details.

### 🤖 Agent Benchmarks

PIArena also supports agentic benchmarks: [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) and [AgentDojo](https://github.com/ethz-spylab/agentdojo).

#### Setup Agent Benchmarks
```bash
# AgentDojo
cd agents/agentdojo && pip install -e . && cd ../..
```

#### InjecAgent Evaluation

```bash
python main_injecagent.py --model meta-llama/Llama-3.1-8B-Instruct --defense none
```


#### AgentDojo Evaluation

```bash
# With OpenAI API
export OPENAI_API_KEY="Your API Key Here"
python main_agentdojo.py --model gpt-5-mini --attack none

# With HuggingFace model (vLLM server started automatically)
python main_agentdojo.py --model meta-llama/Llama-3.1-8B-Instruct --attack tool_knowledge --defense datafilter
```


## 🙋🏻‍♀️ Add your own attacks / defenses
Please see [Document](https://piarena.vercel.app/#/docs/your-own) for full details.



<!-- ## 🔍 Attack & Defense Details

You can go to `piarena/attacks` or `piarena/defenses` to see the attack & defense details and change their configs.

##  🤝 Contribution Guide

Please see [Contributing](). -->


<!-- ## Citation

If you find our paper or the code useful, please kindly cite the following paper:

```bib
@article{geng2026piarena,
  title={PIArena: A Platform for Prompt Injection Evaluation},
  author={Geng, Runpeng and Yin, Chenlong and Wang, Yanting and Chen, Ying and Jia, Jinyuan},
  year={2026}
}
``` -->
