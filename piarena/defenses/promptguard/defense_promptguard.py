from transformers import pipeline
from ..base import BaseDefense, register_defense


@register_defense
class PromptGuardDefense(BaseDefense):
    """Detection defense using Meta's Prompt-Guard-86M classifier."""

    name = "promptguard"
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._detector = pipeline(
            "text-classification",
            model="meta-llama/Prompt-Guard-86M",
            device_map="auto",
        )

    def execute(self, target_inst, context):
        prediction = self._detector(context)[0]["label"]
        detect_flag = "benign" not in prediction.lower()
        return {"detect_flag": detect_flag}

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)
        if llm is not None:
            if result["detect_flag"]:
                result["response"] = "[Warning] PromptGuard detected injected prompt in the context."
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                result["response"] = llm.query(messages)
        else:
            result["response"] = "[PIArena] No LLM provided, response is not available."
        return result
