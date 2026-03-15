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



# Batch defense imports
from .defense_none_batch import no_defense_batch
from .attentiontracker.defense_attentiontracker_batch import attentiontracker_batch
from .datafilter.defense_datafilter_batch import datafilter_batch
from .datasentinel.defense_datasentinel_batch import datasentinel_batch
from .piguard.defense_piguard_batch import piguard_batch
from .promptarmor.defense_promptarmor_batch import promptarmor_batch
from .promptguard.defense_promptguard_batch import promptguard_batch
from .promptlocate.defense_promptlocate_batch import promptlocate_batch
from .secalign.defense_secalign_batch import secalign_batch


# Batch versions of defenses (for strategy_search attack)
DEFENSES_BATCH = {
    "none": no_defense_batch,
    "attentiontracker": attentiontracker_batch,
    "datafilter": datafilter_batch,
    "datasentinel": datasentinel_batch,
    "piguard": piguard_batch,
    "promptarmor": promptarmor_batch,
    "promptguard": promptguard_batch,
    "promptlocate": promptlocate_batch,
    "secalign": secalign_batch,
}
