# -*- coding: utf-8 -*-

import os
import torch

# Only check GPU in main process, not in vLLM subprocesses
# vLLM subprocesses will handle GPU access via CUDA_VISIBLE_DEVICES
# Check if we're in the main process by checking if we're being imported as a module
# vLLM subprocesses will import this file but won't be __main__
is_main_process = __name__ == "__main__" or (
    hasattr(os, 'getppid') and 
    os.getppid() != 1  # Not a daemon process
)

if is_main_process:
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() == 0:
        # Check if CUDA_VISIBLE_DEVICES is set (vLLM subprocesses may not see GPUs directly)
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            raise RuntimeError("No GPUs available and CUDA_VISIBLE_DEVICES not set")
        print(f"⚠️ torch.cuda.device_count()=0 but CUDA_VISIBLE_DEVICES={cuda_visible} is set (may be vLLM subprocess)")
    else:
        assert torch.cuda.device_count() > 0, "No GPUs available"
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    # In vLLM subprocess - skip GPU check, vLLM will handle it
    pass

import argparse
import time
from tqdm import tqdm
from datasets import load_dataset, Dataset
from piarena.utils import *
from piarena.llm import Model
from piarena.attacks import *
from piarena.defenses import *
from piarena.evaluations import *

def parse_args():
    parser = argparse.ArgumentParser(prog='PIArena', description="test")

    # General args
    parser.add_argument('--dataset', type=str, default="squad_v2",
                        help="Path of dataset or subset in https://huggingface.co/datasets/sleeepeer/PIArena.")
    parser.add_argument('--backend_llm', type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Name of the backend LLM to be used.")
    parser.add_argument('--attack', type=str, default="strategy_search",
                        choices=["strategy_search", "pair", "tap"],
                        help="Type of attack to be used.")
    parser.add_argument('--defense', type=str, default="datasentinel",
                        help="Type of defense to be used.")
    parser.add_argument('--attack_path', type=str, default=None,
                        help="Existing attack result to reuse.")
    parser.add_argument('--name', type=str, default='search',
                        help="Name of the experiment.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for the experiment.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for the experiment.")
    parser.add_argument('--attacker_llm', type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Attacker LLM for strategy_search (defaults to backend_llm if not provided).")
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    import os
    
    # Try to load from local JSON file first
    try:
        dataset = load_json(f"datasets/{args.dataset}.json")
        dataset = Dataset.from_list(dataset)
        dataset_name = args.dataset
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
    dataset_name = args.dataset.split('/')[-1].split('.')[0]

    llm_name = args.backend_llm.replace('/', '-')
    attack_result_path = f"results/evaluation_results/{args.name}/tmp_attack_results/{dataset_name}-{llm_name}-{args.attack}-{args.defense}-{args.seed}.json"
    
    # Initialize attack function (always needed, even if attack_result is loaded)
    attack = get_attack(args.attack)
    
    try:
        attack_result = load_json(attack_result_path)
        assert len(attack_result) == len(dataset)
        print(f"Loaded existing attack result from {attack_result_path}.")
    except Exception as e:
        # Fallback: create new attack_result
        attack_result = {}
        print(f"No existing attack result found, will run attack on-the-fly.")

    print(f"Loading backend LLM: {args.backend_llm}")
    llm = Model(args.backend_llm)
    
    attacker_llm = None
    attacker_model_name_or_path = args.attacker_llm
    if args.attack in ["pair", "tap"] and attacker_model_name_or_path:
        print(f"Loading attacker LLM: {args.attacker_llm}")
        attacker_llm = Model(attacker_model_name_or_path)
    elif args.attack in ["pair", "tap"]:
        # For iterative attacks without a dedicated attacker model object, use backend_llm if needed later.
        attacker_llm = llm
    
    defense = get_defense(args.defense)
    evaluation_result_path = f"results/evaluation_results/{args.name}/{dataset_name}-{llm_name}-{args.attack}-{args.defense}-{args.seed}.json"

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


    try:
        evaluation_result = load_json(evaluation_result_path)
        print(f"Loaded existing evaluation result from {evaluation_result_path}.")
        if len(evaluation_result) == len(dataset):
            print("Evaluation result has the same length as the dataset, will quit evaluation.")
            return
    except:
        evaluation_result = {}
        print("No existing evaluation result found, will run evaluation on-the-fly.")

    for idx, dp in tqdm(enumerate(dataset)):
        print(f"=========={idx+1} / {len(dataset)}==========")
        # Only skip if the result exists AND matches current attack+defense configuration
        # (evaluation_result_path already includes attack_name and args.defense, so this check is safe)
        if f"{idx}" in evaluation_result:
            # Additional verification: check if the result has the expected defense
            result_entry = evaluation_result.get(f"{idx}", {})
            # Skip only if entry exists (path already ensures attack+defense match)
            print(f"Skipping index {idx} because it already exists in the evaluation result (attack={args.attack}, defense={args.defense}).")
            continue

        result_dp = copy.deepcopy(dp)
        target_inst = dp["target_inst"]
        context = dp["context"]
        injected_task = dp["injected_task"]
        target_task_answer = dp["target_task_answer"]
        injected_task_answer = dp["injected_task_answer"]
        
        try:
            # Try both string and integer keys (JSON may store keys as strings)
            result_entry = attack_result.get(idx) or attack_result.get(str(idx))
            if result_entry:
                if isinstance(result_entry, dict):
                    injected_context = result_entry.get("injected_context")
                else:
                    injected_context = result_entry
                # Only skip if we have a valid injected_context
                if injected_context is not None and injected_context != "":
                    print(f"✅ Found existing attack result for index {idx}, skipping attack generation")
                else:
                    raise KeyError(f"Key {idx} found but injected_context is empty")
            else:
                raise KeyError(f"Key {idx} not found in attack_result")
        except (KeyError, TypeError):
            # Prepare attack arguments
            attack_kwargs = {
                "target_inst": target_inst,
                "context": context,
                "injected_task": injected_task,
                "target_task_answer": target_task_answer,
                "injected_task_answer": injected_task_answer,
            }
            
            # Add llm and attacker_llm for attacks that need them
            if args.attack in ["strategy_search", "nanogcg", "pair", "tap"]:
                attack_kwargs["llm"] = llm
                if attacker_llm is not None:
                    attack_kwargs["attacker_llm"] = attacker_llm
                # Pass defense for defense-aware attacks
                if args.attack in ["pair", "tap", "strategy_search"]:
                    attack_kwargs["defense"] = defense
                if args.attack == "strategy_search":
                    attack_kwargs["attacker_model_name_or_path"] = attacker_model_name_or_path
            
            injected_context = attack.execute(**attack_kwargs)
            attack_dp = copy.deepcopy(dp)
            attack_dp["injected_context"] = injected_context
            attack_result[idx] = attack_dp
            save_json(attack_result, attack_result_path)

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
