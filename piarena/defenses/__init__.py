from .base import BaseDefense
from ..registry import DEFENSE_REGISTRY

# Import all defense modules to trigger @register_defense decorators
from . import defense_none  # noqa: F401
from .secalign import defense_secalign  # noqa: F401
from .pisanitizer import defense_pisanitizer  # noqa: F401
from .datasentinel import defense_datasentinel  # noqa: F401
from .attentiontracker import defense_attentiontracker  # noqa: F401
from .promptguard import defense_promptguard  # noqa: F401
from .promptlocate import defense_promptlocate  # noqa: F401
from .promptarmor import defense_promptarmor  # noqa: F401
from .datafilter import defense_datafilter  # noqa: F401
from .piguard import defense_piguard  # noqa: F401


# Registry is auto-populated by @register_defense decorators
DEFENSE_CLASSES = DEFENSE_REGISTRY


def get_defense(name: str, config: dict = None) -> BaseDefense:
    """Factory: instantiate a defense by name."""
    cls = DEFENSE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown defense: {name}. Available: {sorted(DEFENSE_REGISTRY)}")
    return cls(config=config)
