from ...llm import Model
from ..base import BaseDefense, register_defense


@register_defense
class DataFilterDefense(BaseDefense):
    """Sanitization defense using recursive content filtering."""

    name = "datafilter"
    DEFAULT_CONFIG = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._filter_model = Model("JoyYizhu/DataFilter")

    def execute(self, target_inst, context):
        from .inference_utils import recursive_filter, parse
        filtered_context = recursive_filter(parse(context), self._filter_model, target_inst)
        return {"cleaned_context": filtered_context}

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
