from typing import List

from ...llm import Model
from ..base import BaseDefense, register_defense
from .inference_utils import recursive_filter, parse


@register_defense
class DataFilterDefense(BaseDefense):
    """Sanitization defense using recursive content filtering."""

    name = "datafilter"
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._filter_model = None

    def _ensure_filter_model(self):
        if self._filter_model is None:
            self._filter_model = Model("JoyYizhu/DataFilter")

    def execute(self, target_inst, context):
        self._ensure_filter_model()
        filtered_context = recursive_filter(parse(context), self._filter_model, target_inst)
        return {"cleaned_context": filtered_context}

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        return super().execute_batch(target_insts, contexts)

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
