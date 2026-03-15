# -*- coding: utf-8 -*-

import torch
print("GPUs available:", torch.cuda.device_count())
assert torch.cuda.device_count() > 0, "No GPUs available"
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import argparse
import copy
import time
import yaml
from tqdm import tqdm
from datasets import load_dataset, Dataset
from piarena.utils import save_json, load_json, setup_seeds, nice_print
from piarena.llm import Model
from piarena.attacks import get_attack
from piarena.defenses import get_defense
from piarena.evaluations import (
    llm_judge, open_prompt_injection_utility,
    substring_match, longbench_metric_dict,
)


def load_config(config_path: str) -> dict:
    """Load experiment configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(prog='PIArena', description="PIArena evaluation pipeline")

    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML config file. CLI args override config values. "
                             "Supports attack_config and defense_config keys for component config.")

    # General args
    parser.add_argument('--dataset', type=str, default=None,
                        help="Path of dataset or subset in https://huggingface.co/datasets/sleeepeer/PIArena.")
    parser.add_argument('--backend_llm', type=str, default=None,
                        help="Name of the backend LLM to be used.")
    parser.add_argument('--attack', type=str, default=None,
                        help="Type of attack to be used.")
    parser.add_argument('--defense', type=str, default=None,
                        help="Type of defense to be used.")
    parser.add_argument('--attack_path', type=str, default=None,
                        help="Existing attack result to reuse.")
    parser.add_argument('--name', type=str, default=None,
                        help="Name of the experiment.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed for the experiment.")

    args = parser.parse_args()

    # Merge YAML config with CLI args (CLI takes priority)
    file_config = {}
    if args.config is not None:
        file_config = load_config(args.config)

    defaults = {
        'dataset': 'squad_v2',
        'backend_llm': 'Qwen/Qwen3-4B-Instruct-2507',
        'attack': 'combined',
        'defense': 'pisanitizer',
        'attack_path': None,
        'name': 'test',
        'seed': 42,
    }
    for key, default_val in defaults.items():
        cli_val = getattr(args, key)
        if cli_val is None:
            setattr(args, key, file_config.get(key, default_val))

    # Store component configs from YAML (attack_config, defense_config)
    args.attack_config = file_config.get("attack_config", None)
    args.defense_config = file_config.get("defense_config", None)

    print(args)
    return args


def main(args):
    # Load dataset
    try:
        dataset = load_json(args.dataset)
        dataset = Dataset.from_list(dataset)
        dataset_name = args.dataset.split('/')[-1].split('.')[0]
    except:
        dataset_name = args.dataset
        while True:
            try:
                dataset = load_dataset(
                    "sleeepeer/PIArena",
                    split=dataset_name,
                    download_mode="force_redownload"
                )
                break
            except Exception as e:
                if "429" in str(e):
                    print("Hit Hugging Face rate limit when loading dataset. Waiting 5 minutes...")
                    time.sleep(300)
                else:
                    raise e

    # Load cached attack results or instantiate attack
    try:
        attack_result = load_json(args.attack_path)
        assert len(attack_result) == len(dataset)
        print(f"Loaded existing attack result from {args.attack_path}.")
        attack_name = args.attack_path.split('/')[-1].split('.')[0]
        attack = None
    except:
        attack_result = {}
        attack_name = args.attack
        attack = get_attack(args.attack, config=args.attack_config)
        print(f"Initialized attack: {attack}")
        print("No existing attack result found, will run attack on-the-fly.")

    # Initialize backend LLM
    print(f"Loading backend LLM: {args.backend_llm}")
    llm = Model(args.backend_llm)

    # Instantiate defense
    defense = get_defense(args.defense, config=args.defense_config)
    print(f"Initialized defense: {defense}")

    # Result paths
    llm_name = args.backend_llm.replace('/', '-')
    attack_result_path = f"results/evaluation_results/{args.name}/tmp_attack_results/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"
    evaluation_result_path = f"results/evaluation_results/{args.name}/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"

    # Select evaluators
    if "open_prompt_injection" in dataset_name:
        asr_evaluator = llm_judge
        utility_evaluator = open_prompt_injection_utility
    elif "sep" in dataset_name:
        asr_evaluator = llm_judge
        utility_evaluator = llm_judge
    elif "knowledge_corruption" in dataset_name:
        asr_evaluator = substring_match
        utility_evaluator = substring_match
    elif "_long" in dataset_name:
        asr_evaluator = llm_judge
        for metric in longbench_metric_dict.keys():
            if metric in dataset_name:
                utility_evaluator = longbench_metric_dict[metric]
                break
    else:
        asr_evaluator = llm_judge
        utility_evaluator = llm_judge

    # Load existing evaluation results
    try:
        evaluation_result = load_json(evaluation_result_path)
        print(f"Loaded existing evaluation result from {evaluation_result_path}.")
        if len(evaluation_result) == len(dataset):
            print("Evaluation result has the same length as the dataset, will quit evaluation.")
            return
    except:
        evaluation_result = {}
        print("No existing evaluation result found, will run evaluation on-the-fly.")

    # Main evaluation loop
    for idx, dp in tqdm(enumerate(dataset)):
        print(f"=========={idx+1} / {len(dataset)}==========")
        if f"{idx}" in evaluation_result:
            print(f"Skipping index {idx} because it already exists in the evaluation result.")
            continue

        result_dp = copy.deepcopy(dp)
        target_inst = dp["target_inst"]
        context = dp["context"]
        injected_task = dp["injected_task"]
        target_task_answer = dp["target_task_answer"]
        injected_task_answer = dp["injected_task_answer"]

        # Attack phase
        try:
            injected_context = attack_result[idx]["injected_context"]
        except:
            injected_context = attack.execute(
                context=context,
                injected_task=injected_task,
                target_inst=target_inst,
                target_task_answer=target_task_answer,
                injected_task_answer=injected_task_answer,
            )
            attack_dp = copy.deepcopy(dp)
            attack_dp["injected_context"] = injected_context
            attack_result[idx] = attack_dp
            save_json(attack_result, attack_result_path)

        # Defense phase
        defense_result = defense.get_response(
            target_inst=target_inst,
            context=injected_context,
            llm=llm,
        )

        response = defense_result["response"]
        result_dp["defense_result"] = defense_result
        result_dp["utility"] = utility_evaluator(
            response,
            ground_truth=target_task_answer,
            task_prompt=f"{target_inst}\n\n{context}",
        )
        result_dp["asr"] = asr_evaluator(
            response,
            ground_truth=injected_task_answer,
            task_prompt=injected_task,
        )

        evaluation_result[idx] = result_dp
        save_json(evaluation_result, evaluation_result_path)

        nice_print(f"Target Instruction: {target_inst}")
        print("\n")
        nice_print(f"Injected Task: {injected_task}")
        print("\n")
        nice_print(f"Response: {response}")
        print("\n")
        nice_print(f"Utility: {result_dp['utility']}, {round(sum([r['utility'] for r in evaluation_result.values()]) / len(evaluation_result), 2)}")
        nice_print(f"ASR: {result_dp['asr']}, {round(sum([r['asr'] for r in evaluation_result.values()]) / len(evaluation_result), 2)}")

if __name__ == '__main__':
    args = parse_args()
    setup_seeds(args.seed)
    torch.cuda.empty_cache()
    main(args)
