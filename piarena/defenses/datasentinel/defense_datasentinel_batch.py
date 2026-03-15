"""
Batch version of datasentinel defense - uses transformer batch inference.
Only used by strategy_search attack.
"""
from typing import List
from ...llm import Model
from .OpenPromptInjection.apps import DataSentinelDetector

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
        "ft_path":"sleeepeer/DataSentinel",
        "decoding_method":"greedy"
    }
}

DETECTOR = None
def get_detector(model_config=DEFAULT_CONFIG):
    detector = DataSentinelDetector(model_config)
    return detector


def datasentinel_batch(
    target_insts: List[str],
    contexts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of datasentinel defense - uses transformer batch inference.
    
    Args:
        target_insts: List of user instructions
        contexts: List of contexts
        system_prompt: System prompt
        llm: Model instance (uses batch_query if available)
        config: Configuration
    
    Returns:
        list of dict: Each containing response and detect_flag
    """
    
    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")
    
    # Initialize detector (uses transformer model for detection)
    if DETECTOR is None:
        print("⏳ Initializing DataSentinel detector (transformer model for detection)...")
        DETECTOR = get_detector(model_config=config)
        print("✅ DataSentinel detector initialized")
    
    detector = DETECTOR
    # Use batch detection if available
    print(f"🔍 Running DataSentinel batch detection on {len(contexts)} contexts...")
    if hasattr(detector, 'batch_detect'):
        detect_flags = detector.batch_detect(contexts)
    else:
        # Fallback to sequential detection
        detect_flags = [detector.detect(ctx) for ctx in contexts]
    print(f"✅ Detection complete: {sum(detect_flags)} detected, {len(contexts) - sum(detect_flags)} not detected")
    
    results = []
    detected_indices = [i for i, flag in enumerate(detect_flags) if flag]
    not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

    responses = [""] * len(target_insts)

    for i in detected_indices:
        responses[i] = "[Warning] DataSentinel detected injected prompt in the context."

    if not_detected_indices and llm is not None:
        # Build batch messages for response generation
        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{target_insts[i]}\n\n{contexts[i]}"},
            ]
            for i in not_detected_indices
        ]
        
        # Use batch_query if available, otherwise fallback to sequential
        print(f"⚡ Using transformer batch inference for {len(messages_batch)} prompts")
        if hasattr(llm, "batch_query"):
            batch_responses = llm.batch_query(messages_batch)
        else:
            batch_responses = [llm.query(msgs) for msgs in messages_batch]
        print(f"✅ Batch generation complete: {len(batch_responses)} responses")
        
        for idx, resp in zip(not_detected_indices, batch_responses):
            responses[idx] = resp
    elif not_detected_indices:
        # No LLM provided
        for i in not_detected_indices:
            responses[i] = "[PIArena] No LLM provided, response is not available."

    for resp, flag in zip(responses, detect_flags):
        results.append({"response": resp, "detect_flag": flag})
    
    return results
