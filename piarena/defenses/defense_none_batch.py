"""
Batch version of no_defense - uses independent vLLM instance for efficient batch inference.
Only used by strategy_search attack.
"""
import os
from typing import List, Union, Dict
from ..llm import Model

DEFAULT_CONFIG = {
    "model_name_or_path": None,  # Will use llm's model if provided
}

# Global vLLM model instance (independent from llm.py)
NONE_VLLM_MODEL = None


class NoneDefenseVLLMModel:
    """
    No defense vLLM model wrapper.
    Uses vLLM for efficient batch inference without any defense.
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
        
        print(f"⏳ Loading NoDefense model with vLLM (batch)...")
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
        
        print(f"✅ NoDefense model loaded (vLLM, supports batch inference)")
    
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
                print(f"⚠️ NoDefense (batch): Truncating input from {len(prompt_tokens)} to {max_input_tokens} tokens")
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


def no_defense_batch(
    target_insts: List[str],
    contexts: List[str],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,  # Used to get model path, but creates independent vLLM instance
    config=DEFAULT_CONFIG,
    attacker_vllm_model=None,  # Optional: reuse attacker vLLM instance if same model
):
    """
    Batch version of no_defense - uses independent vLLM instance for efficient batch inference.
    
    Args:
        target_insts: List of user instructions
        contexts: List of contexts (may contain injected content)
        system_prompt: System prompt
        llm: Model instance (used to get model path, but creates independent vLLM)
        config: Configuration dict
    
    Returns:
        list of dict: Each containing 'response'
    """
    global NONE_VLLM_MODEL
    
    if len(target_insts) != len(contexts):
        raise ValueError("target_insts and contexts must have the same length")
    
    # Determine model path
    model_name_or_path = config.get("model_name_or_path")
    if model_name_or_path is None:
        if llm is not None and hasattr(llm, 'model_name_or_path'):
            model_name_or_path = llm.model_name_or_path
        else:
            raise ValueError("No model path provided. Either set config['model_name_or_path'] or provide llm with model_name_or_path attribute.")
    
    # Skip vLLM for non-HuggingFace models (Azure, Google, Anthropic)
    if any(x in model_name_or_path.lower() for x in ["azure", "google", "anthropic"]):
        # Fallback to using llm directly (if it supports batch_query)
        if llm is not None and hasattr(llm, "batch_query"):
            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                for target_inst, context in zip(target_insts, contexts)
            ]
            batch_responses = llm.batch_query(messages_batch)
            return [{"response": resp} for resp in batch_responses]
        else:
            # Fallback to sequential
            results = []
            for target_inst, context in zip(target_insts, contexts):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                if llm is not None:
                    response = llm.query(messages)
                else:
                    response = "[PIArena] No LLM provided, response is not available."
                results.append({"response": response})
            return results
    
    # Get or initialize vLLM model
    # First, try to reuse existing vLLM instance from attacker_llm if same model
    current_model_path = None
    if NONE_VLLM_MODEL is not None and hasattr(NONE_VLLM_MODEL, 'model_name_or_path'):
        current_model_path = NONE_VLLM_MODEL.model_name_or_path
    
    if NONE_VLLM_MODEL is None or current_model_path != model_name_or_path:
        # Try to reuse attacker_vllm_model if provided and same model
        if attacker_vllm_model is not None:
            if hasattr(attacker_vllm_model, 'model_name_or_path') and attacker_vllm_model.model_name_or_path == model_name_or_path:
                print(f"✅ Reusing existing Attacker LLM vLLM instance for NoDefense (same model: {model_name_or_path})")
                NONE_VLLM_MODEL = attacker_vllm_model
        
        # If still None, create new instance
        if NONE_VLLM_MODEL is None:
            # Get vLLM configuration
            gpu_memory = float(os.getenv("NONE_VLLM_GPU_MEMORY", "0.4"))
            max_model_len = int(os.getenv("NONE_VLLM_MAX_MODEL_LEN", "8192"))
            tensor_parallel_size = os.getenv("NONE_VLLM_TP_SIZE")
            tensor_parallel_size = int(tensor_parallel_size) if tensor_parallel_size else None
            
            # Control vLLM concurrency
            max_num_seqs_env = os.getenv("NONE_VLLM_MAX_NUM_SEQS", "256")
            try:
                max_num_seqs = int(max_num_seqs_env)
            except ValueError:
                print(f"⚠️ Cannot parse NONE_VLLM_MAX_NUM_SEQS={max_num_seqs_env!r}, using default 256")
                max_num_seqs = 256
            
            try:
                NONE_VLLM_MODEL = NoneDefenseVLLMModel(
                    model_name_or_path=model_name_or_path,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory,
                    max_model_len=max_model_len,
                    max_num_seqs=max_num_seqs,
                )
            except Exception as e:
                print(f"⚠️ Failed to load NoDefense model with vLLM (batch): {e}")
                print(f"   Falling back to sequential query with provided llm...")
                # Fallback to sequential with provided llm
                results = []
                for target_inst, context in zip(target_insts, contexts):
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{target_inst}\n\n{context}"},
                    ]
                    if llm is not None:
                        response = llm.query(messages)
                    else:
                        response = "[PIArena] No LLM provided, response is not available."
                    results.append({"response": response})
                return results
    
    # Build batch messages
    messages_batch = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_inst}\n\n{context}"},
        ]
        for target_inst, context in zip(target_insts, contexts)
    ]
    
    # Use batch query with vLLM
    # Use default generation parameters (temperature=0.01, do_sample=False) to match no_defense behavior
    try:
        batch_responses = NONE_VLLM_MODEL.batch_query(
            messages_batch,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            top_p=0.95,
        )
        return [{"response": resp} for resp in batch_responses]
    except Exception as e:
        error_msg = str(e).lower()
        # Detect vLLM fatal errors
        if any(kw in error_msg for kw in ['cuda', 'enginecore', 'nccl', 'illegal instruction', 'worker', 'cancelled']):
            print(f"⚠️ NoDefense vLLM fatal error: {e}")
            print(f"   Falling back to sequential query with provided llm...")
            # Fallback to sequential
            results = []
            for target_inst, context in zip(target_insts, contexts):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                if llm is not None:
                    response = llm.query(messages)
                else:
                    response = "[PIArena] No LLM provided, response is not available."
                results.append({"response": response})
            return results
        else:
            raise
