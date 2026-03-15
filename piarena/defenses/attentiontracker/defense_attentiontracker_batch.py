"""
Batch version of attentiontracker defense - sequential detection, batch LLM query.
Only used by strategy_search attack.
"""
from typing import List
from ...llm import Model
from .AttentionTracker.utils import create_model
from .AttentionTracker.detector.attn import AttentionDetector

DEFAULT_CONFIG = {
    "model_info": {
        "provider": "attn-hf",
        "name": "qwen-attn",
        "model_id": "Qwen/Qwen2-1.5B-Instruct"
    },
    "params": {
        "temperature": 0.1,
        "max_output_tokens": 32,
        "important_heads": [[10, 6], [11, 0], [11, 2], [11, 8], [11, 9], [11, 11], [12, 8], [13, 10], [14, 8], [15, 7], [15, 11], [17, 0], [18, 9], [19, 7]]
    }
}

DETECTOR = None
def get_detector(model_config=DEFAULT_CONFIG):
    model = create_model(config=model_config)
    model.print_model_info()
    detector = AttentionDetector(model)
    print("===================")
    print(f"Using detector: {detector.name}")
    return detector


def _clear_cuda_cache():
    """Clear CUDA cache to prevent OOM"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def attentiontracker_batch(
    target_insts: List[str],
    contexts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of attentiontracker defense
    
    Args:
        target_insts: List of user instructions
        contexts: List of contexts
        system_prompt: System prompt
        llm: Target LLM (should support batch_query)
        config: Model configuration
    
    Returns:
        list of dict: Each containing response and detect_flag
    """

    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")

    if DETECTOR is None:
        DETECTOR = get_detector(model_config=config)
    
    detector = DETECTOR

    # AttentionDetector only supports single detect, call one by one
    detect_flags = []
    for i, ctx in enumerate(contexts):
        flag, _ = detector.detect(ctx)
        detect_flags.append(flag)
        # Clear CUDA cache every 10 samples
        if (i + 1) % 10 == 0:
            _clear_cuda_cache()

    results = []
    if llm is not None:
        detected_indices = [i for i, flag in enumerate(detect_flags) if flag]
        not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

        responses = [""] * len(target_insts)

        for i in detected_indices:
            responses[i] = "[Warning] AttentionTracker detected injected prompt in the context."

        if not_detected_indices:
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_insts[i]}\n\n{contexts[i]}"},
                ]
                for i in not_detected_indices
            ]

            if hasattr(llm, "batch_query"):
                batch_responses = llm.batch_query(messages_batch)
            else:
                batch_responses = [llm.query(msgs) for msgs in messages_batch]

            for idx, resp in zip(not_detected_indices, batch_responses):
                responses[idx] = resp

        for resp, flag in zip(responses, detect_flags):
            results.append({"response": resp, "detect_flag": flag})
    else:
        for flag in detect_flags:
            results.append(
                {
                    "response": "[PIArena] No LLM provided, response is not available.",
                    "detect_flag": flag,
                }
            )

    return results
