"""Agent registry for the declarative configuration system.

Agent types self-register here with their config schema and factory function.
The registry provides introspection for AI agents and validation at compile time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentRegistryEntry:
    """A registered agent type with its config model and metadata."""

    name: str
    config_model: type  # Pydantic BaseModel subclass (BaseAgentConfig)
    category: str  # "background", "strategy", "execution", "market_maker"
    description: str = ""
    agent_class: type | None = None  # The ABIDES agent class to instantiate


class AgentRegistry:
    """Singleton registry of available agent types.

    Agent types register themselves here at import time. The registry is
    queried by the compiler to validate configs and instantiate agents,
    and by AI discoverability APIs to list available agent types.
    """

    _instance: AgentRegistry | None = None
    _entries: dict[str, AgentRegistryEntry]

    def __new__(cls) -> AgentRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries = {}
        return cls._instance

    def register(
        self,
        name: str,
        config_model: type,
        category: str = "background",
        description: str = "",
        agent_class: type | None = None,
        allow_overwrite: bool = False,
    ) -> None:
        """Register an agent type.

        Args:
            name: Unique identifier (e.g. "noise", "value", "adaptive_market_maker").
            config_model: Pydantic model class with agent-specific parameters
                          and a ``create_agents()`` factory method.
            category: One of "background", "strategy", "execution", "market_maker".
            description: Human-readable description for AI discoverability.
            agent_class: The ABIDES agent class to instantiate. When provided,
                         ``BaseAgentConfig`` can auto-generate ``create_agents()``.
            allow_overwrite: If *True*, silently replace an existing entry with
                the same *name* instead of raising.  Useful for notebook
                workflows where cells that define custom agents are re-executed.
        """
        if name in self._entries and not allow_overwrite:
            raise ValueError(f"Agent type '{name}' is already registered")
        self._entries[name] = AgentRegistryEntry(
            name=name,
            config_model=config_model,
            category=category,
            description=description,
            agent_class=agent_class,
        )

    def get(self, name: str) -> AgentRegistryEntry:
        """Look up a registered agent type by name."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries.keys()))
            raise KeyError(f"Unknown agent type '{name}'. Available types: {available}")
        return self._entries[name]

    def list_agents(self) -> list[dict[str, Any]]:
        """Return metadata for all registered agent types (AI-friendly)."""
        result = []
        for entry in self._entries.values():
            schema = entry.config_model.model_json_schema()  # type: ignore[attr-defined]
            result.append(
                {
                    "name": entry.name,
                    "category": entry.category,
                    "description": entry.description,
                    "parameters": schema.get("properties", {}),
                }
            )
        return result

    def get_json_schema(self, name: str) -> dict[str, Any]:
        """Return the JSON Schema for a registered agent type's config."""
        entry = self.get(name)
        return entry.config_model.model_json_schema()  # type: ignore[attr-defined, no-any-return]

    def registered_names(self) -> list[str]:
        """Return sorted list of all registered agent type names."""
        return sorted(self._entries.keys())

    def _clear(self) -> None:
        """Clear registry (for testing only)."""
        self._entries.clear()


# Module-level convenience — the global singleton
registry = AgentRegistry()


def register_agent(
    name: str,
    category: str = "background",
    description: str = "",
    agent_class: type | None = None,
    allow_overwrite: bool = True,
):
    """Decorator to register an agent config model in the global registry.

    By default ``allow_overwrite=True`` so that re-executing a notebook
    cell that defines a custom agent does not raise an error.

    Usage::

        @register_agent("my_agent", category="strategy",
                        agent_class=MyAgent, description="My custom agent")
        class MyAgentConfig(BaseAgentConfig):
            threshold: float = 0.05
            ...
    """

    def decorator(cls: type) -> type:
        registry.register(
            name,
            cls,
            category=category,
            description=description,
            agent_class=agent_class,
            allow_overwrite=allow_overwrite,
        )
        return cls

    return decorator
