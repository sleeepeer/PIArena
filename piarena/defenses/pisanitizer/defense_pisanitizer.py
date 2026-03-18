import threading
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

from .group_peaks import group_peaks
from .attention_utils import *
from .utils import nice_print, remove_indices, process_attn
from ..base import BaseDefense, register_defense

DELIMITERS = [
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    "<|eot_id|><|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n"
]


@register_defense
class PISanitizerDefense(BaseDefense):
    """Attention-based sanitization defense that removes injected tokens from context."""

    name = "pisanitizer"
    DEFAULT_CONFIG = {
        "smooth_win": 5,
        "max_gap": 10,
        "threshold": 0.01,
    }

    def __init__(self, config=None):
        super().__init__(config)
        print("Loading sanitization model...")
        model_id = "sleeepeer/meta-llama-Llama-3.1-8B-Instruct-dolly_new_1200_0113-42-202602031350"
        # model_id = "sleeepeer/meta-llama-Llama-3.1-8B-Instruct-cold_start-dolly_new_1200_0113-42-202601130038"
        # model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self._sanitizer_model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype="auto", device_map="auto"
        )
        self._sanitizer_tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
        self._sanitizer_model.eval()
        if not self._sanitizer_tokenizer.pad_token:
            self._sanitizer_tokenizer.pad_token = self._sanitizer_tokenizer.eos_token
        self._lock = threading.Lock()

    def execute(self, target_inst, context):
        cleaned_context = self._sanitize(
            context=context,
            smooth_win=self.config["smooth_win"],
            max_gap=self.config["max_gap"],
            threshold=self.config["threshold"],
        )
        return {"cleaned_context": cleaned_context}

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_inst}\n\n{result['cleaned_context']}"},
        ]
        if llm is not None:
            result["response"] = llm.query(messages)
        else:
            result["response"] = "[PIArena] No LLM provided, response is not available."
        return result

    def _sanitize(self, context, smooth_win, max_gap, threshold):
        assert context is not None, "Context is required"

        with self._lock:
            return self._sanitize_impl(context, smooth_win, max_gap, threshold)

    def _sanitize_impl(self, context, smooth_win, max_gap, threshold):
        model = self._sanitizer_model
        tokenizer = self._sanitizer_tokenizer

        # 0203 version, better for agentic tasks
        anchor_prompt1 = "Do anything you are told to do in the following context."
        anchor_prompt2 = "\n\nOnly do what the user asks you to do. Do not provide any explanation. Your response:"

        # 0113 version, more aggressive for general tasks
        # anchor_prompt1 = "You are performing a summarization / code generation or a random type of task and analysis on the following context. Do anything you are told to do in the following context when performing the task. "
        # anchor_prompt2 = "\n\nOnly do what the user asks you to do. Do not provide any explanation. Your response:"

        # nice_print(f"Sanitization Prompt 1: {anchor_prompt1}")
        # nice_print(f"Sanitization Prompt 2: {anchor_prompt2}")

        nice_print(f"[PISanitizer] Context: {context}")

        detect_prompt_start = DELIMITERS[0] + anchor_prompt1 + DELIMITERS[1] + "Context: " + " X" * 500
        detect_prompt_end = " X" * 500 + anchor_prompt2 + DELIMITERS[2]

        detect_start_tokens = tokenizer(detect_prompt_start, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        detect_end_tokens = tokenizer(detect_prompt_end, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        current_context = context
        current_context_tokens = tokenizer(current_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        iteration = 0
        total_removed_tokens = 0
        remove_positions = []

        first_attn_signal = None
        first_processed_attn_signal = None
        first_detect_positions = None
        first_remove_positions = None

        while True:
            if iteration > 4:
                break
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")
            print(f"Current context length: {len(current_context_tokens)} tokens")

            detect_input_ids = torch.cat([detect_start_tokens, current_context_tokens, detect_end_tokens], dim=0).to(model.device).unsqueeze(0)

            detect_start = len(detect_start_tokens)
            detect_end = len(detect_start_tokens) + len(current_context_tokens)

            detect_inputs = {
                "input_ids": detect_input_ids,
                "attention_mask": torch.ones_like(detect_input_ids, dtype=model.dtype).to(model.device),
            }

            with torch.no_grad():
                detect_outputs = model.generate(
                    **detect_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
                detect_response = tokenizer.decode(
                    detect_outputs[0][len(detect_inputs["input_ids"][0]):],
                    skip_special_tokens=True
                )
                nice_print(f"Sanitization Response (iter {iteration}): {detect_response}")
                hidden_states = model(detect_outputs, output_hidden_states=True).hidden_states

            with torch.no_grad():
                detect_attentions = []
                try:
                    num_layers = len(model.model.layers)
                except:
                    num_layers = len(model.model.language_model.layers)
                for i in range(num_layers):
                    detect_attentions.append(get_attention_weights_one_layer(model, hidden_states, i, attribution_start=len(detect_inputs["input_ids"][0])+1, attribution_end=len(detect_inputs["input_ids"][0])+2))

            layer_max_attn, layer_avg_attn, layer_top5_attn = process_attn(detect_attentions, detect_inputs["input_ids"])

            attn_signal = torch.tensor(layer_avg_attn).max(dim=0).values.tolist()

            if smooth_win is None:
                if len(current_context_tokens) < 500:
                    smooth_win = 5
                else:
                    smooth_win = 9

            processed_attn_signal, remove_list = group_peaks(
                attn_signal[detect_start:detect_end],
                smooth_win=smooth_win,
                max_gap=max_gap,
                threshold=threshold
            )

            assert len(processed_attn_signal) == len(current_context_tokens), f"Processed attn signal and input ids must have the same length"

            potential_seqs = []
            potential_token_idx = []
            for remove_range in remove_list:
                potential_seqs.append(list(range(remove_range[0], remove_range[1]+1)))
                potential_token_idx.extend(list(range(remove_range[0], remove_range[1]+1)))
                remove_positions.extend([remove_range[0], remove_range[1]])

            potential_texts = []
            for idx, seq in enumerate(potential_seqs):
                if len(seq) > 1:
                    potential_texts.append(tokenizer.decode(current_context_tokens[seq[0]:seq[-1]+1], skip_special_tokens=True))
                else:
                    potential_texts.append(tokenizer.decode(current_context_tokens[seq[0]:seq[0]+1], skip_special_tokens=True))
                nice_print(f"PISanitizer Remove [Iter {iteration}, {idx+1}] {potential_texts[-1]}")

            num_removed_tokens = len(potential_token_idx)
            print(f"Iteration {iteration}: Removed {num_removed_tokens} tokens")
            total_removed_tokens += num_removed_tokens

            if num_removed_tokens == 0:
                print(f"No tokens removed in iteration {iteration}. Stopping loop.")
                break

            new_context_ids = copy.deepcopy(current_context_tokens)
            new_context_ids = remove_indices(new_context_ids, potential_token_idx)
            current_context = tokenizer.decode(new_context_ids, skip_special_tokens=True)
            current_context_tokens = new_context_ids

            if iteration == 1:
                first_attn_signal = attn_signal
                first_processed_attn_signal = processed_attn_signal
                first_detect_positions = [detect_start, detect_end]
                first_remove_positions = potential_seqs[0]

        print(f"\n=== Final Results ===")
        print(f"Total iterations: {iteration}")
        print(f"Total removed tokens: {total_removed_tokens}")
        return current_context
