from typing import List

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
        self._detector = None

    def _ensure_detector(self):
        if self._detector is None:
            model = create_model(config=self.config)
            model.print_model_info()
            self._detector = AttentionDetector(model)
            print("===================")
            print(f"Using detector: {self._detector.name}")

    def execute(self, target_inst, context):
        self._ensure_detector()
        detect_flag, _ = self._detector.detect(context)
        return {"detect_flag": detect_flag}

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_detector()
        return [{"detect_flag": self._detector.detect(context)[0]} for context in contexts]

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

    def get_response_batch(self, target_insts, contexts, llm,
                           system_prompt="You are a helpful assistant."):
        defense_results = self.execute_batch(target_insts, contexts)
        responses = ["[Warning] AttentionTracker detected injected prompt in the context." if result["detect_flag"] else "" for result in defense_results]
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
