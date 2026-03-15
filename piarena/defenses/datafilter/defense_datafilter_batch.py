"""
Batch version of datafilter defense - uses vLLM+batch for efficient inference.
Only used by strategy_search attack.
"""
import os
from typing import List
from ...llm import Model
from .inference_utils import recursive_filter, parse, apply_filter_in_batch

# Default configuration
DEFAULT_CONFIG = {
    "use_vllm": True,  # Whether to use vLLM, False uses transformers
    "gpu_memory_utilization": 0.3,  # vLLM GPU memory fraction
    "max_model_len": 20480,  # Max sequence length
}

# Can override via environment variables:
# DATAFILTER_USE_VLLM=0 use transformers
# DATAFILTER_GPU_MEMORY=0.2 set GPU memory fraction
# DATAFILTER_MAX_MODEL_LEN=20480 set max sequence length

FILTER_MODEL = None
FILTER_MODEL_TYPE = None  # "vllm" or "transformers"
FILTER_MAX_MODEL_LEN = None  # Track current vLLM max_model_len for truncation


def get_filter_model_vllm(gpu_memory_utilization=0.3, max_model_len=20480):
    """Load filter model with vLLM (high throughput)"""
    global FILTER_MAX_MODEL_LEN
    from vllm import LLM
    
    # Check if CUDA graph should be disabled
    enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
    
    print(f"⏳ Loading DataFilter model with vLLM (gpu_memory={gpu_memory_utilization:.0%}, max_model_len={max_model_len})...")
    filter_model = LLM(
        model="JoyYizhu/DataFilter",
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )
    FILTER_MAX_MODEL_LEN = max_model_len
    print(f"✅ DataFilter model loaded (vLLM)")
    return filter_model


def get_filter_model_transformers():
    """Load filter model with transformers (better compatibility)"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    model_name = "JoyYizhu/DataFilter"
    print(f"⏳ Loading DataFilter model with transformers...")
    
    hf_cache_dir = os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=hf_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=hf_cache_dir
    )
    model.eval()
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ DataFilter model loaded (transformers)")
    return {"model": model, "tokenizer": tokenizer}


def recursive_filter_transformers(data, model_dict, instruction):
    """Filter using transformers model"""
    from .inference_utils import apply_filter_for_single as _apply_filter_for_single
    return _apply_filter_for_single(model_dict, instruction, data)


def datafilter_batch(
    target_insts: List[str],
    contexts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=None,
):
    """
    Batch version of datafilter defense
    
    Args:
        target_insts: List of user instructions
        contexts: List of contexts (corresponding to target_insts)
        system_prompt: System prompt
        llm: Target LLM
        config: Configuration
    
    Returns:
        list of dict: Each containing response and cleaned_context
    """
    global FILTER_MODEL, FILTER_MODEL_TYPE
    
    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")
    
    # Merge config
    effective_config = DEFAULT_CONFIG.copy()
    if config:
        effective_config.update(config)
    
    # Environment variable overrides
    if os.getenv("DATAFILTER_USE_VLLM") is not None:
        effective_config["use_vllm"] = os.getenv("DATAFILTER_USE_VLLM").lower() in ("1", "true", "yes")
    if os.getenv("DATAFILTER_GPU_MEMORY") is not None:
        effective_config["gpu_memory_utilization"] = float(os.getenv("DATAFILTER_GPU_MEMORY"))
    if os.getenv("DATAFILTER_MAX_MODEL_LEN") is not None:
        effective_config["max_model_len"] = int(os.getenv("DATAFILTER_MAX_MODEL_LEN"))
    
    use_vllm = effective_config["use_vllm"]
    gpu_memory = effective_config["gpu_memory_utilization"]
    max_model_len = effective_config["max_model_len"]
    
    # Load model
    expected_type = "vllm" if use_vllm else "transformers"
    if FILTER_MODEL is None or FILTER_MODEL_TYPE != expected_type:
        if use_vllm:
            FILTER_MODEL = get_filter_model_vllm(gpu_memory, max_model_len)
            FILTER_MODEL_TYPE = "vllm"
        else:
            FILTER_MODEL = get_filter_model_transformers()
            FILTER_MODEL_TYPE = "transformers"
    
    # Parse all contexts
    parsed_contexts = []
    context_types = []  # Track each context type: 'str', 'dict', 'list'
    
    for ctx in contexts:
        parsed = parse(ctx)
        if isinstance(parsed, str):
            parsed_contexts.append(parsed)
            context_types.append('str')
        elif isinstance(parsed, dict):
            parsed_contexts.append(parsed)
            context_types.append('dict')
        elif isinstance(parsed, list):
            parsed_contexts.append(parsed)
            context_types.append('list')
        else:
            parsed_contexts.append(str(parsed))
            context_types.append('str')
    
    # Process based on model type
    if FILTER_MODEL_TYPE == "vllm":
        # Collect all strings to filter
        str_indices = []
        str_instructions = []
        str_datas = []
        
        for i, (pctx, ctype, inst) in enumerate(zip(parsed_contexts, context_types, target_insts)):
            if ctype == 'str':
                str_indices.append(i)
                str_instructions.append(inst)
                str_datas.append(pctx)
            elif ctype == 'dict':
                for k, v in pctx.items():
                    if isinstance(v, str):
                        str_indices.append((i, k))
                        str_instructions.append(inst)
                        str_datas.append(v)
            elif ctype == 'list':
                for j, v in enumerate(pctx):
                    if isinstance(v, str):
                        str_indices.append((i, j))
                        str_instructions.append(inst)
                        str_datas.append(v)
        
        # Batch filter
        if str_datas:
            filtered_strs = apply_filter_in_batch(FILTER_MODEL, str_instructions, str_datas)
        else:
            filtered_strs = []
        
        # Rebuild filtered_contexts
        filtered_contexts = [None] * len(contexts)
        str_result_idx = 0
        
        for idx in str_indices:
            if isinstance(idx, int):
                filtered_contexts[idx] = filtered_strs[str_result_idx]
            else:
                i, key = idx
                if filtered_contexts[i] is None:
                    filtered_contexts[i] = parsed_contexts[i].copy() if isinstance(parsed_contexts[i], dict) else list(parsed_contexts[i])
                filtered_contexts[i][key] = filtered_strs[str_result_idx]
            str_result_idx += 1
        
        # Handle unmodified contexts
        for i in range(len(filtered_contexts)):
            if filtered_contexts[i] is None:
                filtered_contexts[i] = parsed_contexts[i]
    else:
        # transformers version, process one by one
        filtered_contexts = [
            recursive_filter_transformers(pctx, FILTER_MODEL, inst)
            for pctx, inst in zip(parsed_contexts, target_insts)
        ]
    
    # Batch query LLM
    results = []
    if llm is not None:
        messages_list = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{inst}\n\n{fctx}"},
            ]
            for inst, fctx in zip(target_insts, filtered_contexts)
        ]
        
        if hasattr(llm, 'batch_query'):
            responses = llm.batch_query(messages_list)
        else:
            responses = [llm.query(msgs) for msgs in messages_list]
        
        for resp, fctx in zip(responses, filtered_contexts):
            results.append({
                "response": resp,
                "cleaned_context": fctx,
            })
    else:
        for fctx in filtered_contexts:
            results.append({
                "response": "[PIArena] No LLM provided, response is not available.",
                "cleaned_context": fctx,
            })
    
    return results
