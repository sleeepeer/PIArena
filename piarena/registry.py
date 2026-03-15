class Registry:
    """Simple name -> class registry with decorator-based registration."""

    def __init__(self, kind: str):
        self.kind = kind  # "attack" or "defense", used in error messages
        self._classes: dict[str, type] = {}

    def register(self, cls=None, *, name: str = None):
        """Decorator to register a class.

        Usage:
            @registry.register
            class MyThing(Base):
                name = "my_thing"
                ...

            # or with explicit name override:
            @registry.register(name="custom_name")
            class MyThing(Base):
                ...
        """
        def decorator(cls_):
            key = name or getattr(cls_, "name", None)
            if key is None:
                raise ValueError(
                    f"Cannot register {cls_.__name__}: set a 'name' class attribute "
                    f"or pass name= to @register."
                )
            if key in self._classes:
                raise ValueError(
                    f"{self.kind} name '{key}' already registered by "
                    f"{self._classes[key].__name__}, cannot re-register with {cls_.__name__}."
                )
            self._classes[key] = cls_
            return cls_

        # Support both @register and @register(name="x")
        if cls is not None:
            return decorator(cls)
        return decorator

    def get(self, name: str):
        """Return the registered class, or raise ValueError."""
        if name in self._classes:
            return self._classes[name]
        raise ValueError(
            f"Unknown {self.kind}: '{name}'. Available: {list(self._classes)}"
        )

    def keys(self):
        return self._classes.keys()

    def items(self):
        return self._classes.items()

    def __contains__(self, name):
        return name in self._classes

    def __iter__(self):
        return iter(self._classes)

    def __len__(self):
        return len(self._classes)

    def __repr__(self):
        return f"Registry({self.kind}, {list(self._classes)})"


# Global registries
ATTACK_REGISTRY = Registry("attack")
DEFENSE_REGISTRY = Registry("defense")
