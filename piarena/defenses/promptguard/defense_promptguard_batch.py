"""
Batch version of promptguard defense - uses independent vLLM instance for efficient batch inference.
Only used by strategy_search attack.
"""
import os
from typing import List, Union, Dict
from ...llm import Model
from transformers import pipeline
DEFAULT_CONFIG = {}

DETECTOR = None
def get_detector():
    classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M", device_map="auto")
    return classifier


# Prompt-Guard model max input length (avoid OOM)
MAX_CONTEXT_LENGTH = 20480

# Global vLLM model instance (independent from llm.py)
PROMPTGUARD_VLLM_MODEL = None


class PromptGuardVLLMModel:
    """
    PromptGuard vLLM model wrapper.
    Uses vLLM for efficient batch inference after detection.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = None,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
        max_num_seqs: int = 256,
    ):
        from vllm import LLM, SamplingParams
        
        self.model_name_or_path = model_name_or_path
        self.max_model_len = max_model_len
        self.SamplingParams = SamplingParams
        
        # Get available GPU count
        if tensor_parallel_size is None:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                tensor_parallel_size = len([g.strip() for g in cuda_visible.split(",") if g.strip()])
            else:
                import torch
                tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Get HF cache directory
        hf_cache_dir = os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None
        
        print(f"⏳ Loading PromptGuard target LLM with vLLM (batch)...")
        print(f"   Model: {model_name_or_path}")
        print(f"   Tensor parallel size: {tensor_parallel_size} GPU(s)")
        print(f"   GPU memory utilization: {gpu_memory_utilization:.0%}")
        print(f"   Max model length: {max_model_len}")
        print(f"   Max number of sequences: {max_num_seqs}")
        
        # Check if CUDA graph should be disabled
        enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
        
        # Load vLLM model
        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            download_dir=hf_cache_dir,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        
        print(f"✅ PromptGuard target LLM loaded (vLLM, supports batch inference)")
    
    def _format_messages(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """Format messages to prompt string"""
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
        
        return prompt
    
    def batch_query(
        self,
        messages_list: List[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Batch query"""
        if not messages_list:
            return []
        
        # Format all messages
        prompts = [self._format_messages(m) for m in messages_list]
        
        # Truncate long prompts
        max_input_tokens = self.max_model_len - max_new_tokens - 100  # Safety margin
        truncated_prompts = []
        for prompt in prompts:
            prompt_tokens = self.tokenizer.encode(prompt)
            if len(prompt_tokens) > max_input_tokens:
                print(f"⚠️ PromptGuard (batch): Truncating input from {len(prompt_tokens)} to {max_input_tokens} tokens")
                prompt_tokens = prompt_tokens[:max_input_tokens]
                prompt = self.tokenizer.decode(prompt_tokens)
            truncated_prompts.append(prompt)
        
        # Create sampling parameters
        effective_temp = temperature if do_sample else 0.0
        sampling_params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=effective_temp,
            top_p=top_p if do_sample else 1.0,
        )
        
        # Generate
        outputs = self.model.generate(
            truncated_prompts,
            sampling_params,
        )
        
        # Extract generated text
        results = [output.outputs[0].text for output in outputs]
        return results


def promptguard_batch(
    target_insts: List[str],
    contexts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,  # Used to get model path, but creates independent vLLM instance
    config=DEFAULT_CONFIG,
):
    """
    Batch version of promptguard defense - uses independent vLLM instance for efficient batch inference.
    
    Args:
        target_insts: List of user instructions
        contexts: List of contexts
        system_prompt: System prompt
        llm: Model instance (used to get model path, but creates independent vLLM)
        config: Configuration
    
    Returns:
        list of dict: Each containing response and detect_flag
    """
    global DETECTOR, PROMPTGUARD_VLLM_MODEL

    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")

    if DETECTOR is None:
        DETECTOR = get_detector()

    # Replace None contexts and truncate long texts
    raw_contexts = []
    original_contexts = []
    for ctx in contexts:
        original_contexts.append(ctx)
        if ctx is None:
            raw_contexts.append("")
        elif len(ctx) > MAX_CONTEXT_LENGTH:
            raw_contexts.append(ctx[:MAX_CONTEXT_LENGTH])
        else:
            raw_contexts.append(ctx)

    # Batch prediction (using transformers pipeline)
    try:
        batch_preds = DETECTOR(raw_contexts)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            # OOM: fallback to one by one with shorter truncation
            batch_preds = []
            for ctx in raw_contexts:
                if len(ctx) > MAX_CONTEXT_LENGTH // 2:
                    ctx = ctx[:MAX_CONTEXT_LENGTH // 2]
                try:
                    pred = DETECTOR(ctx)
                    batch_preds.append(pred[0] if isinstance(pred, list) else pred)
                except RuntimeError:
                    batch_preds.append({"label": "injected"})
        else:
            raise

    detect_flags = []
    for pred in batch_preds:
        if isinstance(pred, list):
            label = pred[0]["label"]
        else:
            label = pred["label"]
        detect_flags.append("benign" not in label.lower())

    # Generate responses using independent vLLM
    results = []
    detected_indices = [i for i, flag in enumerate(detect_flags) if flag]
    not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

    responses = [""] * len(target_insts)

    for i in detected_indices:
        responses[i] = "[Warning] PromptGuard detected injected prompt in the context."

    if not_detected_indices and llm is not None:
        # Determine model path
        model_name_or_path = config.get("model_name_or_path")
        if model_name_or_path is None:
            if hasattr(llm, 'model_name_or_path'):
                model_name_or_path = llm.model_name_or_path
            else:
                # Fallback to sequential query
                for i in not_detected_indices:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{target_insts[i]}\n\n{original_contexts[i]}"},
                    ]
                    responses[i] = llm.query(messages)
                for resp, flag in zip(responses, detect_flags):
                    results.append({"response": resp, "detect_flag": flag})
                return results
        
        # Skip vLLM for non-HuggingFace models
        if any(x in model_name_or_path.lower() for x in ["azure", "google", "anthropic"]):
            # Fallback to using llm directly
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_insts[i]}\n\n{original_contexts[i]}"},
                ]
                for i in not_detected_indices
            ]
            if hasattr(llm, "batch_query"):
                batch_responses = llm.batch_query(messages_batch)
            else:
                batch_responses = [llm.query(msgs) for msgs in messages_batch]
            for idx, resp in zip(not_detected_indices, batch_responses):
                responses[idx] = resp
        else:
            # Use independent vLLM instance
            if PROMPTGUARD_VLLM_MODEL is None or PROMPTGUARD_VLLM_MODEL.model_name_or_path != model_name_or_path:
                # Get vLLM configuration
                gpu_memory = float(os.getenv("PROMPTGUARD_VLLM_GPU_MEMORY", "0.4"))
                max_model_len = int(os.getenv("PROMPTGUARD_VLLM_MAX_MODEL_LEN", "8192"))
                tensor_parallel_size = os.getenv("PROMPTGUARD_VLLM_TP_SIZE")
                tensor_parallel_size = int(tensor_parallel_size) if tensor_parallel_size else None
                
                max_num_seqs_env = os.getenv("PROMPTGUARD_VLLM_MAX_NUM_SEQS", "256")
                try:
                    max_num_seqs = int(max_num_seqs_env)
                except ValueError:
                    print(f"⚠️ Cannot parse PROMPTGUARD_VLLM_MAX_NUM_SEQS={max_num_seqs_env!r}, using default 256")
                    max_num_seqs = 256
                
                try:
                    PROMPTGUARD_VLLM_MODEL = PromptGuardVLLMModel(
                        model_name_or_path=model_name_or_path,
                        tensor_parallel_size=tensor_parallel_size,
                        gpu_memory_utilization=gpu_memory,
                        max_model_len=max_model_len,
                        max_num_seqs=max_num_seqs,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to load PromptGuard target LLM with vLLM (batch): {e}")
                    print(f"   Falling back to sequential query with provided llm...")
                    # Fallback to sequential
                    for i in not_detected_indices:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"{target_insts[i]}\n\n{original_contexts[i]}"},
                        ]
                        responses[i] = llm.query(messages)
                    for resp, flag in zip(responses, detect_flags):
                        results.append({"response": resp, "detect_flag": flag})
                    return results
            
            # Build batch messages
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_insts[i]}\n\n{original_contexts[i]}"},
                ]
                for i in not_detected_indices
            ]
            
            # Use batch query with vLLM
            try:
                batch_responses = PROMPTGUARD_VLLM_MODEL.batch_query(messages_batch)
                for idx, resp in zip(not_detected_indices, batch_responses):
                    responses[idx] = resp
            except Exception as e:
                error_msg = str(e).lower()
                if any(kw in error_msg for kw in ['cuda', 'enginecore', 'nccl', 'illegal instruction', 'worker', 'cancelled']):
                    print(f"⚠️ PromptGuard vLLM fatal error: {e}")
                    print(f"   Falling back to sequential query with provided llm...")
                    # Fallback to sequential
                    for i in not_detected_indices:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"{target_insts[i]}\n\n{original_contexts[i]}"},
                        ]
                        responses[i] = llm.query(messages)
                else:
                    raise
    elif not_detected_indices:
        # No LLM provided
        for i in not_detected_indices:
            responses[i] = "[PIArena] No LLM provided, response is not available."

    for resp, flag in zip(responses, detect_flags):
        results.append({"response": resp, "detect_flag": flag})
    
    return results
