---
title: Getting Started
description: Install PIArena and run your first evaluation.
order: 10
---

# Quick Start

Install PIArena and run evaluation.

## Installation

Clone the project and setup python environment:

```bash
git clone git@github.com:sleeepeer/PIArena.git
cd PIArena
conda create -n piarena python=3.10 -y
conda activate piarena
pip install -r requirements.txt
pip install -e .   # Install piarena as an editable package
```

**Login to HuggingFace** 🤗 with your HuggingFace Access Token, you can find it at [this link](https://huggingface.co/settings/tokens):

```bash
huggingface-cli login
```

### Ready-to-use Tools

You can simply import attacks and defenses and integrate them into your own code. Please see details in [Attacks & Defenses](https://piarena.vercel.app/#/docs/attacks-and-defenses).

```python
from piarena.attacks import get_attack
from piarena.defenses import get_defense
from piarena.llm import Model

llm = Model("Qwen/Qwen3-4B-Instruct-2507")
defense = get_defense("promptguard")
attack = get_attack("combined")
```

### Run Evaluation

Use `main.py` to run the benchmark. You can also use `scripts/run.py` to run many experiments in parallel.

```bash
# Using CLI arguments
python main.py --dataset squad_v2 --attack direct --defense none

# Using a YAML config file
python main.py --config configs/experiments/my_experiment.yaml

# Easily start large-scale experiments
python scripts/run.py
```
