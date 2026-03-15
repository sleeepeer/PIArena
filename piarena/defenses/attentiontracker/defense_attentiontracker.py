from .AttentionTracker.utils import create_model
from .AttentionTracker.detector.attn import AttentionDetector
from ..base import BaseDefense, register_defense


@register_defense
class AttentionTrackerDefense(BaseDefense):
    """Detection defense using attention pattern analysis."""

    name = "attentiontracker"
    DEFAULT_CONFIG = {
        "model_info": {
            "provider": "attn-hf",
            "name": "qwen-attn",
            "model_id": "Qwen/Qwen2-1.5B-Instruct"
        },
        "params": {
            "temperature": 0.1,
            "max_output_tokens": 32,
            "important_heads": [
                [10, 6], [11, 0], [11, 2], [11, 8], [11, 9], [11, 11],
                [12, 8], [13, 10], [14, 8], [15, 7], [15, 11],
                [17, 0], [18, 9], [19, 7]
            ]
        }
    }

    def __init__(self, config=None):
        super().__init__(config)
        model = create_model(config=self.config)
        model.print_model_info()
        self._detector = AttentionDetector(model)
        print("===================")
        print(f"Using detector: {self._detector.name}")

    def execute(self, target_inst, context):
        detect_flag, _ = self._detector.detect(context)
        return {"detect_flag": detect_flag}

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)
        if llm is not None:
            if result["detect_flag"]:
                result["response"] = "[Warning] AttentionTracker detected injected prompt in the context."
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{target_inst}\n\n{context}"},
                ]
                result["response"] = llm.query(messages)
        else:
            result["response"] = "[PIArena] No LLM provided, response is not available."
        return result
