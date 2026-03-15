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
        self._locator = PromptLocate(self.config)
        self._detector = DataSentinelDetector(self.config)

    def execute(self, target_inst, context):
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
