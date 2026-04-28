# -*- coding: utf-8 -*-
"""Batched vLLM evaluation pipeline.

Mirrors the (attack -> defense -> evaluation) flow of `main.py` but runs each
phase in batch on a vLLM engine: attacks are computed up-front for any rows
missing from the cache, the defense's batch path drives a single
`llm.batch_query` for all unfinished rows, and the LLM judges run in two batched
passes (utility + ASR) for non-KC splits. KC splits use substring matching.

Use `main.py` for the simple sample-by-sample HF runner; this file is for vLLM.
"""

import argparse
import copy
import os
import time

# vLLM 0.19 v1 EngineCore spawns a subprocess; default fork() inherits CUDA from
# the parent, which blows up once any CUDA call has already run here (e.g.
# `torch.cuda.manual_seed` inside `setup_seeds`). Forcing spawn sidesteps it.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm

from piarena.attacks import get_attack
from piarena.defenses import get_defense
from piarena.evaluations import (
    llm_judge_asr_batch,
    llm_judge_utility_batch,
    substring_match,
)
from piarena.llm import Model
from piarena.utils import load_json, nice_print, save_json, setup_seeds


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(prog="PIArena-vllm", description="PIArena vLLM batch pipeline")

    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config. CLI args override config values.")

    # General
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--backend_llm", type=str, default=None)
    p.add_argument("--attack", type=str, default=None)
    p.add_argument("--defense", type=str, default=None)
    p.add_argument("--attack_path", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Rows per batched defense/judge call (0 = run everything in one batch).")

    # Backend vLLM tunables
    p.add_argument("--tp_size", type=int, default=None)
    p.add_argument("--gpu_mem", type=float, default=None)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--max_num_seqs", type=int, default=None)
    p.add_argument("--enforce_eager", action="store_true", default=None,
                   help="Disable CUDA graphs in the backend engine.")

    # Judge LLM
    p.add_argument("--judge_llm", type=str, default=None,
                   help='Judge model id, or "same" to reuse the backend LLM.')
    p.add_argument("--judge_tp_size", type=int, default=None)
    p.add_argument("--judge_gpu_mem", type=float, default=None)
    p.add_argument("--judge_max_model_len", type=int, default=None)
    p.add_argument("--judge_max_num_seqs", type=int, default=None)
    p.add_argument("--judge_enforce_eager", action="store_true", default=None)

    args = p.parse_args()

    file_config = {}
    if args.config is not None:
        file_config = load_config(args.config)

    defaults = {
        "dataset": "squad_v2",
        "backend_llm": "Qwen/Qwen3-4B-Instruct-2507",
        "attack": "combined",
        "defense": "pisanitizer",
        "attack_path": None,
        "name": "test_vllm",
        "seed": 42,
        "batch_size": 100,
        "tp_size": 1,
        "gpu_mem": 0.85,
        "max_model_len": 8192,
        "max_num_seqs": 256,
        "enforce_eager": False,
        "judge_llm": "Qwen/Qwen3-4B-Instruct-2507",
        "judge_tp_size": 1,
        "judge_gpu_mem": 0.85,
        "judge_max_model_len": 8192,
        "judge_max_num_seqs": 256,
        "judge_enforce_eager": False,
    }
    for key, default_val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, file_config.get(key, default_val))

    args.attack_config = file_config.get("attack_config", None)
    args.defense_config = file_config.get("defense_config", None)
    print(args)
    return args


def _load_dataset(args):
    try:
        data = load_json(args.dataset)
        return Dataset.from_list(data), args.dataset.split("/")[-1].split(".")[0]
    except Exception:
        pass
    name = args.dataset
    while True:
        try:
            return (
                load_dataset("sleeepeer/PIArena", split=name, download_mode="force_redownload"),
                name,
            )
        except Exception as e:
            if "429" in str(e):
                print("Hit Hugging Face rate limit when loading dataset. Waiting 5 minutes...")
                time.sleep(300)
            else:
                raise


def _chunked(seq, size):
    if size <= 0:
        yield list(seq)
        return
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _attack_rows(attack, attack_result, dataset, indices, attack_result_path):
    """Compute and cache injected_context for every index missing from attack_result."""
    if attack is None:
        return
    missing = [i for i in indices if attack_result.get(i) is None and attack_result.get(str(i)) is None]
    if not missing:
        return
    for i in tqdm(missing, desc="attack"):
        dp = dataset[i]
        injected_context = attack.execute(
            context=dp["context"],
            injected_task=dp["injected_task"],
            target_inst=dp["target_inst"],
            target_task_answer=dp["target_task_answer"],
            injected_task_answer=dp["injected_task_answer"],
        )
        cached = copy.deepcopy(dp)
        cached["injected_context"] = injected_context
        attack_result[i] = cached
    save_json(attack_result, attack_result_path)


def _injected_context_for(idx, attack_result):
    entry = attack_result.get(idx) or attack_result.get(str(idx))
    if entry is None:
        raise KeyError(f"attack_result missing index {idx}")
    if isinstance(entry, dict):
        return entry["injected_context"]
    return entry


def main(args):
    dataset, dataset_name = _load_dataset(args)
    dataset_name = args.dataset.split("/")[-1].split(".")[0]

    if args.attack_path:
        try:
            attack_result = load_json(args.attack_path)
            assert len(attack_result) == len(dataset)
            print(f"Loaded existing attack result from {args.attack_path}.")
            attack_name = args.attack_path.split("/")[-1].split(".")[0]
            attack = None
        except Exception:
            attack_result = {}
            attack_name = args.attack
            attack = get_attack(args.attack, config=args.attack_config)
            print("Attack path provided but unloadable; running attack on-the-fly.")
    else:
        attack_result = {}
        attack_name = args.attack
        attack = get_attack(args.attack, config=args.attack_config)
        print(f"Initialized attack: {attack}")

    print(f"Loading backend LLM (vLLM): {args.backend_llm}")
    llm = Model(
        args.backend_llm,
        backend="vllm",
        tp_size=args.tp_size,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )

    if args.judge_llm == "same":
        judge_llm = llm
        print("Reusing backend LLM as judge.")
    else:
        print(f"Loading judge LLM (vLLM): {args.judge_llm}")
        judge_llm = Model(
            args.judge_llm,
            backend="vllm",
            tp_size=args.judge_tp_size,
            gpu_mem=args.judge_gpu_mem,
            max_model_len=args.judge_max_model_len,
            max_num_seqs=args.judge_max_num_seqs,
            enforce_eager=args.judge_enforce_eager,
        )

    defense = get_defense(args.defense, config=args.defense_config)
    print(f"Initialized defense: {defense}")

    llm_name = args.backend_llm.replace("/", "-")
    attack_result_path = f"results/evaluation_results/{args.name}/tmp_attack_results/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"
    evaluation_result_path = f"results/evaluation_results/{args.name}/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"

    use_kc_substring = "knowledge_corruption" in dataset_name

    try:
        evaluation_result = load_json(evaluation_result_path)
        print(f"Loaded existing evaluation result from {evaluation_result_path}.")
        if len(evaluation_result) == len(dataset):
            print("Evaluation result has the same length as the dataset, will quit evaluation.")
            return
    except Exception:
        evaluation_result = {}
        print("No existing evaluation result found, will run evaluation on-the-fly.")

    unfinished = [i for i in range(len(dataset)) if str(i) not in evaluation_result and i not in evaluation_result]
    print(f"Unfinished samples: {len(unfinished)} / {len(dataset)}")
    if not unfinished:
        return

    for chunk in _chunked(unfinished, args.batch_size):
        _attack_rows(attack, attack_result, dataset, chunk, attack_result_path)

        target_insts = [dataset[i]["target_inst"] for i in chunk]
        injected_contexts = [_injected_context_for(i, attack_result) for i in chunk]

        defense_results = defense.get_response_batch(target_insts, injected_contexts, llm)
        responses = [r["response"] for r in defense_results]

        if use_kc_substring:
            target_answers = [dataset[i]["target_task_answer"] for i in chunk]
            injected_answers = [dataset[i]["injected_task_answer"] for i in chunk]
            target_prompts = [f"{dataset[i]['target_inst']}\n\n{dataset[i]['context']}" for i in chunk]
            injected_prompts = [dataset[i]["injected_task"] for i in chunk]
            utils = [substring_match(r, ground_truth=g, task_prompt=t)
                     for r, g, t in zip(responses, target_answers, target_prompts)]
            asrs = [substring_match(r, ground_truth=g, task_prompt=t)
                    for r, g, t in zip(responses, injected_answers, injected_prompts)]
        else:
            target_answers = [dataset[i]["target_task_answer"] for i in chunk]
            injected_answers = [dataset[i]["injected_task_answer"] for i in chunk]
            target_prompts = [f"{dataset[i]['target_inst']}\n\n{dataset[i]['context']}" for i in chunk]
            injected_prompts = [dataset[i]["injected_task"] for i in chunk]
            utils = llm_judge_utility_batch(responses, target_answers, target_prompts, llm=judge_llm)
            asrs = llm_judge_asr_batch(responses, injected_answers, injected_prompts, llm=judge_llm)

        for idx, defense_result, util, asr in zip(chunk, defense_results, utils, asrs):
            result_dp = copy.deepcopy(dataset[idx])
            result_dp["defense_result"] = defense_result
            result_dp["utility"] = bool(util)
            result_dp["asr"] = bool(asr)
            evaluation_result[idx] = result_dp

        save_json(evaluation_result, evaluation_result_path)

        running_util = sum(int(r["utility"]) for r in evaluation_result.values()) / len(evaluation_result)
        running_asr = sum(int(r["asr"]) for r in evaluation_result.values()) / len(evaluation_result)
        nice_print(f"chunk done: n={len(chunk)}  running utility={running_util:.3f}  running ASR={running_asr:.3f}")


if __name__ == "__main__":
    args = parse_args()
    setup_seeds(args.seed)
    main(args)
