from .base import BaseAttack
from ..registry import ATTACK_REGISTRY

# Import all attack modules to trigger @register_attack decorators
from . import attack_none  # noqa: F401
from . import heuristic  # noqa: F401
from .nanogcg import attack_nanogcg  # noqa: F401
from .strategy_search import attack_strategy_search  # noqa: F401
from .pair import attack_pair  # noqa: F401
from .tap import attack_tap  # noqa: F401

# Registry is auto-populated by @register_attack decorators
ATTACK_CLASSES = ATTACK_REGISTRY


def get_attack(name: str, config: dict = None) -> BaseAttack:
    """Factory: instantiate an attack by name."""
    cls = ATTACK_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown attack: {name}. Available: {sorted(ATTACK_REGISTRY)}")
    return cls(config=config)
