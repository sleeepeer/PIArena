from abc import ABC, abstractmethod
from typing import List
from ..registry import DEFENSE_REGISTRY

# Expose decorator at module level for convenience
register_defense = DEFENSE_REGISTRY.register


class BaseDefense(ABC):
    """Base class for all defenses in PIArena."""

    name: str
    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    @abstractmethod
    def execute(self, target_inst: str, context: str) -> dict:
        """Core defense algorithm. Return dict with defense-specific keys."""
        ...

    def execute_batch(self, target_insts: List[str], contexts: List[str]) -> List[dict]:
        """Default batch execution: fall back to single-item execution."""
        self._validate_batch_inputs(target_insts, contexts)
        return [
            self.execute(target_inst, context)
            for target_inst, context in zip(target_insts, contexts)
        ]

    def _validate_batch_inputs(self, target_insts: List[str], contexts: List[str]) -> None:
        if len(target_insts) != len(contexts):
            raise ValueError("target_insts and contexts must have the same length")

    def _build_messages(self, target_inst: str, context: str, system_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_inst}\n\n{context}"},
        ]

    def get_response(self, target_inst: str, context: str, llm,
                     system_prompt: str = "You are a helpful assistant.") -> dict:
        """Execute defense + query LLM. Override for custom behavior."""
        result = self.execute(target_inst, context)
        if llm is None:
            response = "[PIArena] No LLM provided, response is not available."
        else:
            messages = self._build_messages(target_inst, context, system_prompt)
            response = llm.query(messages)
        return {**result, "response": response}

    def get_response_batch(self, target_insts: List[str], contexts: List[str], llm,
                           system_prompt: str = "You are a helpful assistant.") -> List[dict]:
        """Default batch response path with sequential defense execution."""
        self._validate_batch_inputs(target_insts, contexts)
        defense_results = self.execute_batch(target_insts, contexts)

        if llm is None:
            return [
                {**result, "response": "[PIArena] No LLM provided, response is not available."}
                for result in defense_results
            ]

        messages_batch = [
            self._build_messages(target_inst, context, system_prompt)
            for target_inst, context in zip(target_insts, contexts)
        ]
        responses = llm.batch_query(messages_batch)
        return [
            {**result, "response": response}
            for result, response in zip(defense_results, responses)
        ]

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
