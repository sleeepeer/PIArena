from .base import BaseAttack, register_attack


@register_attack
class NoAttack(BaseAttack):
    """Baseline: returns context unmodified."""

    name = "none"

    def execute(self, context, injected_task, inject_position="end", inject_times=1, **kwargs):
        return context
