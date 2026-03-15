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
        self._detector = DataSentinelDetector(self.config)

    def execute(self, target_inst, context):
        detect_flag = self._detector.detect(context)
        return {"detect_flag": detect_flag}

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
