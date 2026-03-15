"""
Batch version of promptlocate defense - sequential detection/location, batch LLM query.
Only used by strategy_search attack.
"""
from typing import List
from ...llm import Model
from .OpenPromptInjection.apps import PromptLocate, DataSentinelDetector

DEFAULT_CONFIG = {
    "model_info":{
        "provider":"mistral",
        "name":"mistralai/Mistral-7B-v0.1"
    },
    "api_key_info":{
        "api_keys":[0],
        "api_key_use": 0
    },
    "params":{
        "temperature":0.1,
        "seed":100,
        "gpus":["5","6"],
        "device":"cuda",
        "max_output_tokens":128,
        "ft_path":"sleeepeer/PromptLocate",
        "decoding_method":"greedy"
    }
}

LOCATOR = None
def get_locator(model_config=DEFAULT_CONFIG):
    locator = PromptLocate(model_config)
    return locator

DETECTOR = None
def get_detector(model_config=DEFAULT_CONFIG):
    detector = DataSentinelDetector(model_config)
    return detector


def promptlocate_batch(
    target_insts: List[str],
    contexts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of promptlocate defense
    
    Args:
        target_insts: List of user instructions
        contexts: List of contexts
        system_prompt: System prompt
        llm: Target LLM (should support batch_query)
        config: Configuration
    
    Returns:
        list of dict: Each containing response, detect_flag, cleaned_context, potential_injection
    """
    
    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")
    
    if LOCATOR is None:
        LOCATOR = get_locator(model_config=config)
    if DETECTOR is None:
        DETECTOR = get_detector(model_config=config)
    
    locator = LOCATOR
    detector = DETECTOR
    
    # Process one by one (detection and location)
    cleaned_contexts = []
    detect_flags = []
    localized_injections = []
    
    for ctx, inst in zip(contexts, target_insts):
        detect_flag = detector.detect(ctx)
        detect_flags.append(detect_flag)
        
        if detect_flag:
            new_context, localized_injected_prompt = locator.locate_and_recover(ctx, inst)
            cleaned_contexts.append(new_context)
            localized_injections.append(localized_injected_prompt)
        else:
            cleaned_contexts.append(ctx)
            localized_injections.append("")
    
    # Batch query LLM
    results = []
    if llm is not None:
        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{target_insts[i]}\n\n{cleaned_contexts[i]}"},
            ]
            for i in range(len(target_insts))
        ]
        
        if hasattr(llm, "batch_query"):
            responses = llm.batch_query(messages_batch)
        else:
            responses = [llm.query(msgs) for msgs in messages_batch]
        
        for resp, flag, cleaned_ctx, loc_inj in zip(responses, detect_flags, cleaned_contexts, localized_injections):
            results.append({
                "response": resp,
                "detect_flag": flag,
                "cleaned_context": cleaned_ctx,
                "potential_injection": loc_inj,
            })
    else:
        for flag, cleaned_ctx, loc_inj in zip(detect_flags, cleaned_contexts, localized_injections):
            results.append({
                "response": "[PIArena] No LLM provided, response is not available.",
                "detect_flag": flag,
                "cleaned_context": cleaned_ctx,
                "potential_injection": loc_inj,
            })
    
    return results
