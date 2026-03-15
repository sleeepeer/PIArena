from abc import ABC, abstractmethod
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

    def get_response(self, target_inst: str, context: str, llm,
                     system_prompt: str = "You are a helpful assistant.") -> dict:
        """Execute defense + query LLM. Override for custom behavior."""
        result = self.execute(target_inst, context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_inst}\n\n{context}"},
        ]
        response = llm.query(messages)
        return {**result, "response": response}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
