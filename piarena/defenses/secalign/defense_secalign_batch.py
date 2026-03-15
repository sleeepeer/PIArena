"""
Batch version of secalign defense - uses independent vLLM instance for parallel inference.
Only used by strategy_search attack.
"""
import os
from typing import List, Optional, Union
from ...llm import Model

DEFAULT_CONFIG_SECALIGN = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",  # Base model
    "lora_adapter": "facebook/Meta-SecAlign-8B",       # LoRA adapter
}

# Global SecAlign vLLM model instance (independent from llm.py)
SECALIGN_MODEL = None


class SecAlignVLLMModel:
    """
    SecAlign vLLM model wrapper.
    Uses vLLM to load base model + LoRA adapter for inference.
    """
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        lora_adapter: str = "facebook/Meta-SecAlign-8B",
        tensor_parallel_size: int = None,
        gpu_memory_utilization: float = 0.35,
        max_model_len: int = 20480,
        max_num_seqs: Optional[int] = None,
    ):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        
        self.base_model = base_model
        self.lora_adapter = lora_adapter
        self.max_model_len = max_model_len
        self.SamplingParams = SamplingParams
        self.LoRARequest = LoRARequest
        
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
        
        print(f"⏳ Loading SecAlign with vLLM (batch)...")
        print(f"   Base model: {base_model}")
        print(f"   LoRA adapter: {lora_adapter}")
        print(f"   Tensor parallel size: {tensor_parallel_size} GPU(s)")
        print(f"   GPU memory utilization: {gpu_memory_utilization:.0%}")
        print(f"   Max model length: {max_model_len}")
        if max_num_seqs is not None:
            print(f"   Max number of sequences: {max_num_seqs}")
        
        # Check if CUDA graph should be disabled
        enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
        
        # Load vLLM model with LoRA support
        self.model = LLM(
            model=base_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_lora=True,
            max_lora_rank=64,  # SecAlign uses LoRA rank 64
            download_dir=hf_cache_dir,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
        )
        
        self.tokenizer = self.model.get_tokenizer()
        
        # Create LoRA request (for each inference)
        self.lora_request = LoRARequest(
            lora_name="secalign",
            lora_int_id=1,
            lora_path=lora_adapter,
        )
        
        print(f"✅ SecAlign loaded (vLLM + LoRA, supports batch inference)")
    
    def _format_messages(self, messages: Union[str, List[dict]]) -> str:
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
                elif role == "input":
                    prompt += f"Input: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
        
        return prompt
    
    def batch_query(
        self,
        messages_list: List[Union[str, List[dict]]],
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
                print(f"⚠️ SecAlign (batch): Truncating input from {len(prompt_tokens)} to {max_input_tokens} tokens")
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
        
        # Use LoRA for inference
        outputs = self.model.generate(
            truncated_prompts,
            sampling_params,
            lora_request=self.lora_request,
        )
        
        # Extract generated text
        results = [output.outputs[0].text for output in outputs]
        return results


def _build_secalign_messages(target_inst: str, context: Optional[str], system_prompt: str) -> List[dict]:
    """Build SecAlign message format"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": target_inst},
    ]
    if context is not None:
        messages.append({"role": "input", "content": context})
    return messages


def secalign_batch(
    target_insts: List[str],
    contexts: List[Optional[str]],
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,  # Unused, SecAlign uses its own vLLM instance
    config=DEFAULT_CONFIG_SECALIGN,
) -> List[dict]:
    """
    Batch version of SecAlign defense: uses independent vLLM instance for parallel inference.
    
    Args:
        target_insts: List of target instructions
        contexts: List of contexts (may contain injected content)
        system_prompt: System prompt
        llm: Unused, SecAlign uses its own vLLM instance
        config: Configuration dict
    
    Returns:
        List[dict]: [{"response": response1}, {"response": response2}, ...]
    """
    global SECALIGN_MODEL
    
    if not target_insts:
        return []
    
    # Ensure contexts length matches target_insts
    if contexts is None:
        contexts = [None] * len(target_insts)
    elif len(contexts) != len(target_insts):
        raise ValueError(f"Length mismatch: target_insts={len(target_insts)}, contexts={len(contexts)}")
    
    # Get or initialize SecAlign vLLM model
    if SECALIGN_MODEL is None:
        base_model = config.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
        lora_adapter = config.get("lora_adapter", "facebook/Meta-SecAlign-8B")
        
        # Compatibility with old config format
        if "model_name_or_path" in config and "lora_adapter" not in config:
            lora_adapter = config["model_name_or_path"]
        
        # Get vLLM configuration
        gpu_memory = float(os.getenv("SECALIGN_VLLM_GPU_MEMORY", "0.35"))
        max_model_len = int(os.getenv("SECALIGN_VLLM_MAX_MODEL_LEN", "20480"))
        tensor_parallel_size = os.getenv("SECALIGN_VLLM_TP_SIZE")
        tensor_parallel_size = int(tensor_parallel_size) if tensor_parallel_size else None
        
        # Control vLLM concurrency to avoid OOM
        max_num_seqs_env = os.getenv("SECALIGN_VLLM_MAX_NUM_SEQS", "256")
        try:
            max_num_seqs = int(max_num_seqs_env)
        except ValueError:
            print(f"⚠️ Cannot parse SECALIGN_VLLM_MAX_NUM_SEQS={max_num_seqs_env!r}, using default 256")
            max_num_seqs = 256
        
        try:
            SECALIGN_MODEL = SecAlignVLLMModel(
                base_model=base_model,
                lora_adapter=lora_adapter,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
            )
        except Exception as e:
            print(f"⚠️ Failed to load SecAlign with vLLM (batch): {e}")
            print(f"   Please ensure vLLM is installed and configured correctly.")
            raise
    
    # Build batch messages
    messages_list = [
        _build_secalign_messages(inst, ctx, system_prompt)
        for inst, ctx in zip(target_insts, contexts)
    ]
    
    # Use batch query (vLLM supports batch inference)
    try:
        responses = SECALIGN_MODEL.batch_query(messages_list)
    except Exception as e:
        error_msg = str(e).lower()
        # Detect vLLM fatal errors
        if any(kw in error_msg for kw in ['cuda', 'enginecore', 'nccl', 'illegal instruction', 'worker', 'cancelled']):
            print(f"⚠️ SecAlign vLLM fatal error: {e}")
            print(f"   This may require restarting the process.")
            raise
        else:
            raise
    
    return [{"response": resp} for resp in responses]
