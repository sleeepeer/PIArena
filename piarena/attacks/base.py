from abc import ABC, abstractmethod
from ..registry import ATTACK_REGISTRY

# Expose decorator at module level for convenience
register_attack = ATTACK_REGISTRY.register


class BaseAttack(ABC):
    """Base class for all attacks in PIArena."""

    name: str
    DEFAULT_CONFIG = {}

    def __init__(self, config: dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    @abstractmethod
    def execute(self, context: str, injected_task: str, **kwargs) -> str:
        """Run attack. Return injected context string."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"
