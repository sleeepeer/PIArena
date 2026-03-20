from typing import List

from .OpenPromptInjection.apps import DataSentinelDetector
from ..base import BaseDefense, register_defense


@register_defense
class DataSentinelDefense(BaseDefense):
    """Detection defense using a fine-tuned DataSentinel model."""

    name = "datasentinel"
    DEFAULT_CONFIG = {
        "model_info": {
            "provider": "mistral",
            "name": "mistralai/Mistral-7B-v0.1"
        },
        "api_key_info": {
            "api_keys": [0],
            "api_key_use": 0
        },
        "params": {
            "temperature": 0.1,
            "seed": 100,
            "gpus": ["5", "6"],
            "device": "cuda",
            "max_output_tokens": 128,
            "ft_path": "sleeepeer/DataSentinel",
            "decoding_method": "greedy"
        }
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._detector = None

    def _ensure_detector(self):
        if self._detector is None:
            self._detector = DataSentinelDetector(self.config)

    def execute(self, target_inst, context):
        self._ensure_detector()
        detect_flag = self._detector.detect(context)
        return {"detect_flag": detect_flag}

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_detector()
        detector = self._detector
        if hasattr(detector, "batch_detect"):
            detect_flags = detector.batch_detect(contexts)
        else:
            detect_flags = [detector.detect(context) for context in contexts]
        return [{"detect_flag": detect_flag} for detect_flag in detect_flags]

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)
        if llm is not None:
            if result["detect_flag"]:
                result["response"] = "[Warning] DataSentinel detected injected prompt in the context."
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                result["response"] = llm.query(messages)
        else:
            result["response"] = "[PIArena] No LLM provided, response is not available."
        return result

    def get_response_batch(self, target_insts, contexts, llm,
                           system_prompt="You are a helpful assistant."):
        defense_results = self.execute_batch(target_insts, contexts)
        responses = ["[Warning] DataSentinel detected injected prompt in the context." if result["detect_flag"] else "" for result in defense_results]
        if llm is not None:
            safe_indices = [i for i, result in enumerate(defense_results) if not result["detect_flag"]]
            if safe_indices:
                messages_batch = [
                    self._build_messages(target_insts[i], contexts[i], system_prompt)
                    for i in safe_indices
                ]
                safe_responses = llm.batch_query(messages_batch)
                for idx, response in zip(safe_indices, safe_responses):
                    responses[idx] = response
        else:
            responses = ["[PIArena] No LLM provided, response is not available."] * len(target_insts)
        return [{**result, "response": response} for result, response in zip(defense_results, responses)]
