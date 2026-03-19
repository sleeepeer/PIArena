import os
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ..base import BaseDefense, register_defense


@register_defense
class PIGuardDefense(BaseDefense):
    """Detection defense using the PIGuard fine-tuned classifier."""

    name = "piguard"
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._detector = None

    def _ensure_detector(self):
        if self._detector is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "leolee99/PIGuard", cache_dir=os.getenv("HF_HOME")
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "leolee99/PIGuard",
                cache_dir=os.getenv("HF_HOME"),
                trust_remote_code=True,
                device_map="auto",
            )
            self._detector = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
            )

    def execute(self, target_inst, context):
        self._ensure_detector()
        result = self._detector(context)[0]
        prediction = result["label"]
        score = result["score"]

        if prediction == "benign":
            detect_flag = False
        elif prediction == "injection":
            detect_flag = True
        else:
            raise ValueError(f"Unknown prediction: {prediction}")

        return {"detect_flag": detect_flag, "score": score}

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_detector()
        raw_contexts = [context if context is not None else "" for context in contexts]
        batch_preds = self._detector(raw_contexts)
        results = []
        for pred in batch_preds:
            result = pred[0] if isinstance(pred, list) else pred
            prediction = result["label"]
            score = result["score"]
            if prediction == "benign":
                detect_flag = False
            elif prediction == "injection":
                detect_flag = True
            else:
                raise ValueError(f"Unknown prediction: {prediction}")
            results.append({"detect_flag": detect_flag, "score": score})
        return results

    def get_response(self, target_inst, context, llm,
                     system_prompt="You are a helpful assistant."):
        result = self.execute(target_inst, context)
        if llm is not None:
            if result["detect_flag"]:
                result["response"] = "[Warning] PIGuard detected injected prompt in the context."
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
        responses = ["[Warning] PIGuard detected injected prompt in the context." if result["detect_flag"] else "" for result in defense_results]
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
