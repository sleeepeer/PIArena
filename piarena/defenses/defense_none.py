from .base import BaseDefense, register_defense


@register_defense
class NoDefense(BaseDefense):
    """Baseline: no defense applied, direct LLM query."""

    name = "none"
    DEFAULT_CONFIG = {}

    def execute(self, target_inst, context):
        return {}
