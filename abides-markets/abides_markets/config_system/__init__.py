"""Declarative configuration system for ABIDES market simulations.

This package provides a pluggable, AI-friendly configuration system built
on Pydantic models with YAML/JSON serialization. Agent types self-register
in a global registry, configs are composable from templates, and the system
compiles down to the existing runtime dict that ``Kernel`` expects.

Quick start::

    from abides_markets.config_system import SimulationBuilder, compile

    config = (SimulationBuilder()
        .from_template("rmsc04")
        .market(ticker="AAPL")
        .seed(42)
        .build())

    runtime = compile(config)

    from abides_core import abides
    end_state = abides.run(runtime)

AI discoverability::

    from abides_markets.config_system import list_agent_types, list_templates, get_config_schema

    # What agent types are available?
    agent_types = list_agent_types()

    # What templates are available?
    templates = list_templates()

    # What's the full schema?
    schema = get_config_schema()
"""

from __future__ import annotations

from typing import Any

# Ensure built-in agent types are registered on import
import abides_markets.config_system.builtin_registrations  # noqa: F401

# Re-export agent config base for third-party agents
from abides_markets.config_system.agent_configs import BaseAgentConfig
from abides_markets.config_system.builder import SimulationBuilder
from abides_markets.config_system.compiler import compile
from abides_markets.config_system.models import (
    AgentGroupConfig,
    ExchangeConfig,
    InfrastructureConfig,
    LatencyConfig,
    MarketConfig,
    MeanRevertingOracleConfig,
    SimulationConfig,
    SimulationMeta,
    SparseMeanRevertingOracleConfig,
)
from abides_markets.config_system.registry import (
    AgentRegistry,
    register_agent,
    registry,
)
from abides_markets.config_system.serialization import (
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)
from abides_markets.config_system.templates import (
    list_templates,
)

# ---------------------------------------------------------------------------
# AI Discoverability API
# ---------------------------------------------------------------------------


def list_agent_types() -> list[dict[str, Any]]:
    """Return metadata for all registered agent types.

    Each entry includes: ``name``, ``category``, ``description``,
    and ``parameters`` (JSON Schema of the agent's config model).

    This is intended for AI agents (LLM tool-calling) to discover
    what agent types are available and what parameters they accept.
    """
    return registry.list_agents()


def get_config_schema() -> dict[str, Any]:
    """Return the full JSON Schema for SimulationConfig.

    AI agents can use this to understand the complete config structure.
    """
    return SimulationConfig.model_json_schema()


def validate_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Validate a config dict and return a result.

    Performs two-phase validation:
    1. Validates the overall config structure via Pydantic.
    2. Validates each agent group's params against its registered config model.

    Returns:
        ``{"valid": True}`` if valid, or
        ``{"valid": False, "errors": [...]}`` with validation error details.
    """
    try:
        config = SimulationConfig.model_validate(config_dict)
    except Exception as e:
        return {"valid": False, "errors": [str(e)]}

    errors = []
    for agent_name, group in config.agents.items():
        if not group.enabled:
            continue
        try:
            entry = registry.get(agent_name)
            entry.config_model(**group.params)
        except Exception as e:
            errors.append(f"Agent '{agent_name}': {e}")

    if errors:
        return {"valid": False, "errors": errors}
    return {"valid": True}


__all__ = [
    # Builder
    "SimulationBuilder",
    # Compiler
    "compile",
    # Models
    "SimulationConfig",
    "MarketConfig",
    "AgentGroupConfig",
    "InfrastructureConfig",
    "LatencyConfig",
    "ExchangeConfig",
    "SimulationMeta",
    "SparseMeanRevertingOracleConfig",
    "MeanRevertingOracleConfig",
    # Registry
    "AgentRegistry",
    "register_agent",
    "registry",
    "BaseAgentConfig",
    # Serialization
    "load_config",
    "save_config",
    "config_to_dict",
    "config_from_dict",
    # Discoverability
    "list_agent_types",
    "list_templates",
    "get_config_schema",
    "validate_config",
]
