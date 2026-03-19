from typing import List

from transformers import pipeline
from ..base import BaseDefense, register_defense


@register_defense
class PromptGuardDefense(BaseDefense):
    """Detection defense using Meta's Prompt-Guard-86M classifier."""

    name = "promptguard"
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._detector = None

    def _ensure_detector(self):
        if self._detector is None:
            self._detector = pipeline(
                "text-classification",
                model="meta-llama/Prompt-Guard-86M",
                device_map="auto",
            )

    def execute(self, target_inst, context):
        self._ensure_detector()
        prediction = self._detector(context)[0]["label"]
        detect_flag = "benign" not in prediction.lower()
        return {"detect_flag": detect_flag}

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_detector()
        raw_contexts = [context if context is not None else "" for context in contexts]
        batch_preds = self._detector(raw_contexts)
        results = []
        for pred in batch_preds:
            label = pred[0]["label"] if isinstance(pred, list) else pred["label"]
            results.append({"detect_flag": "benign" not in label.lower()})
        return results

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

    def get_response_batch(self, target_insts, contexts, llm,
                           system_prompt="You are a helpful assistant."):
        defense_results = self.execute_batch(target_insts, contexts)
        responses = ["[Warning] PromptGuard detected injected prompt in the context." if result["detect_flag"] else "" for result in defense_results]
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
