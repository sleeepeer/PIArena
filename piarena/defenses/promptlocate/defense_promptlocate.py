from typing import List

from .OpenPromptInjection.apps import PromptLocate, DataSentinelDetector
from ..base import BaseDefense, register_defense


@register_defense
class PromptLocateDefense(BaseDefense):
    """Hybrid defense: detection via DataSentinel + localization and removal via PromptLocate."""

    name = "promptlocate"
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
            "ft_path": "sleeepeer/PromptLocate",
            "decoding_method": "greedy"
        }
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._locator = None
        self._detector = None

    def _ensure_models(self):
        if self._locator is None:
            self._locator = PromptLocate(self.config)
        if self._detector is None:
            self._detector = DataSentinelDetector(self.config)

    def execute(self, target_inst, context):
        self._ensure_models()
        detect_flag = self._detector.detect(context)
        if detect_flag:
            new_context, localized_injected_prompt = self._locator.locate_and_recover(context, target_inst)
        else:
            new_context = context
            localized_injected_prompt = ""
        return {
            "detect_flag": detect_flag,
            "cleaned_context": new_context,
            "potential_injection": localized_injected_prompt,
        }

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_models()
        results = []
        for target_inst, context in zip(target_insts, contexts):
            detect_flag = self._detector.detect(context)
            if detect_flag:
                new_context, localized_injected_prompt = self._locator.locate_and_recover(context, target_inst)
            else:
                new_context = context
                localized_injected_prompt = ""
            results.append({
                "detect_flag": detect_flag,
                "cleaned_context": new_context,
                "potential_injection": localized_injected_prompt,
            })
        return results

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

    def get_response_batch(self, target_insts, contexts, llm,
                           system_prompt="You are a helpful assistant."):
        defense_results = self.execute_batch(target_insts, contexts)
        if llm is None:
            return [
                {**result, "response": "[PIArena] No LLM provided, response is not available."}
                for result in defense_results
            ]
        messages_batch = [
            self._build_messages(target_inst, result["cleaned_context"], system_prompt)
            for target_inst, result in zip(target_insts, defense_results)
        ]
        responses = llm.batch_query(messages_batch)
        return [{**result, "response": response} for result, response in zip(defense_results, responses)]
