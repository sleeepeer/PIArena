from ...llm import Model
from ..base import BaseDefense, register_defense


@register_defense
class SecAlignDefense(BaseDefense):
    """Specialized alignment-based defense using the SecAlign model."""

    name = "secalign"
    DEFAULT_CONFIG = {
        "model_name_or_path": "facebook/Meta-SecAlign-8B",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = Model(self.config["model_name_or_path"])

    def execute(self, target_inst, context):
        return {}

    def get_response(self, target_inst, context, llm=None,
                     system_prompt="You are a helpful assistant."):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": target_inst},
        ]
        if context is not None:
            messages.append({"role": "input", "content": context})
        return {"response": self._model.query(messages)}
