from typing import List

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
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = Model(self.config["model_name_or_path"])

    def execute(self, target_inst, context):
        return {}

    def get_response(self, target_inst, context, llm=None,
                     system_prompt="You are a helpful assistant."):
        self._ensure_model()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": target_inst},
        ]
        if context is not None:
            messages.append({"role": "input", "content": context})
        return {"response": self._model.query(messages)}

    def get_response_batch(self, target_insts: List[str], contexts: List[str], llm=None,
                           system_prompt="You are a helpful assistant.") -> List[dict]:
        self._validate_batch_inputs(target_insts, contexts)
        self._ensure_model()
        messages_batch = []
        for target_inst, context in zip(target_insts, contexts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": target_inst},
            ]
            if context is not None:
                messages.append({"role": "input", "content": context})
            messages_batch.append(messages)
        responses = self._model.batch_query(messages_batch)
        return [{"response": response} for response in responses]
