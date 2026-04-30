"""Declarative configuration system for ABIDES market simulations.

This package provides a pluggable, AI-friendly configuration system built
on Pydantic models with YAML/JSON serialization. Agent types self-register
in a global registry, configs are composable from templates, and the system
compiles down to the existing runtime dict that ``Kernel`` expects.

Quick start::

    from abides_markets.config_system import SimulationBuilder
    from abides_markets.simulation import run_simulation

    config = (SimulationBuilder()
        .from_template("rmsc04")
        .market(ticker="AAPL")
        .seed(42)
        .build())

    result = run_simulation(config)  # compiles fresh agents, returns SimulationResult

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

import warnings
from dataclasses import dataclass, field
from typing import Any

# Ensure built-in agent types are registered on import
import abides_markets.config_system.builtin_registrations  # noqa: F401

# Re-export agent config base for third-party agents
from abides_markets.config_system.agent_configs import BaseAgentConfig
from abides_markets.config_system.builder import SimulationBuilder
from abides_markets.config_system.compiler import compile
from abides_markets.config_system.compiler import derive_seed as derive_seed
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
from abides_markets.config_system.templates import list_templates

# ---------------------------------------------------------------------------
# Structured validation types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding (error or warning)."""

    severity: str  # "error" | "warning"
    message: str
    field_path: str | None = None
    agent_name: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Structured result of :func:`validate_config`.

    Attributes
    ----------
    issues : list[ValidationIssue]
        All discovered errors and warnings.
    """

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True when there are no *error*-severity issues."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        """Legacy-compatible dict form: ``{"valid": bool, "errors": [...]}``."""
        error_strings = [i.message for i in self.errors]
        if error_strings:
            return {"valid": False, "errors": error_strings}
        return {"valid": True}

    # Backward-compatible dict-style access
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()


# ---------------------------------------------------------------------------
# Category taxonomy
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, dict[str, Any]] = {
    "background": {
        "label": "Liquidity & Background",
        "description": (
            "Agents that provide baseline market activity. Most simulations "
            "need at least noise + value agents for a functioning order book."
        ),
        "sort_order": 1,
    },
    "market_maker": {
        "label": "Market Makers",
        "description": (
            "Agents that provide two-sided liquidity with tighter spreads "
            "and deeper books."
        ),
        "sort_order": 2,
    },
    "strategy": {
        "label": "Trading Strategies",
        "description": "Directional trading strategies that consume liquidity.",
        "sort_order": 3,
    },
    "execution": {
        "label": "Execution Algorithms",
        "description": (
            "Algorithms for executing large orders with minimal market impact."
        ),
        "sort_order": 4,
    },
}


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
    return SimulationConfig.model_json_schema()  # type: ignore[no-any-return]


def get_full_manifest() -> dict[str, Any]:
    """Return complete metadata sufficient to auto-generate a configuration UI.

    The manifest includes agent types with their JSON-Schema parameters,
    oracle options, available templates, and the category taxonomy.

    Returns a dict with keys:

    - ``agent_types`` — list from :func:`list_agent_types` (enriched with
      ``requires_oracle``, ``typical_count_range``, ``recommended_with``).
    - ``market_config_schema`` — JSON Schema for :class:`MarketConfig`.
    - ``oracle_options`` — list of oracle type descriptors.
    - ``templates`` — list from :func:`list_templates`.
    - ``categories`` — the :data:`CATEGORIES` taxonomy dict.
    """
    from abides_markets.config_system.models import (
        ExternalDataOracleConfig,
        MarketConfig,
        MeanRevertingOracleConfig,
        SparseMeanRevertingOracleConfig,
    )

    oracle_options: list[dict[str, Any]] = [
        {
            "type": "sparse_mean_reverting",
            "description": "Mean-reverting oracle with sparse noise and mega-shocks.",
            "schema": SparseMeanRevertingOracleConfig.model_json_schema(),
        },
        {
            "type": "mean_reverting",
            "description": (
                "Simplified mean-reverting oracle (deprecated — prefer sparse)."
            ),
            "schema": MeanRevertingOracleConfig.model_json_schema(),
        },
        {
            "type": "external_data",
            "description": (
                "Marker for an externally-injected oracle built with a custom "
                "data provider (no configurable fields)."
            ),
            "schema": ExternalDataOracleConfig.model_json_schema(),
        },
        {
            "type": None,
            "description": "No oracle — LOB-based agents only.",
        },
    ]

    return {
        "agent_types": registry.list_agents(),
        "market_config_schema": MarketConfig.model_json_schema(),
        "oracle_options": oracle_options,
        "templates": list_templates(),
        "categories": CATEGORIES,
    }


def validate_config(config_dict: dict[str, Any]) -> ValidationResult:
    """Validate a config dict and return a structured result.

    Performs three-phase validation:
    1. Validates the overall config structure via Pydantic.
    2. Validates each agent group's params against its registered config model.
    3. Runs cross-agent consistency checks (soft warnings).

    Returns a :class:`ValidationResult` with structured issues.

    .. versionchanged:: 2.3
       Returns :class:`ValidationResult` instead of a plain dict.
       The previous ``{"valid": bool, "errors": [...]}`` shape is still
       accessible via :meth:`ValidationResult.to_dict`.
    """
    issues: list[ValidationIssue] = []

    try:
        config = SimulationConfig.model_validate(config_dict)
    except Exception as e:
        issues.append(
            ValidationIssue(
                severity="error",
                message=str(e),
                field_path="(root)",
            )
        )
        return ValidationResult(issues=issues)

    for agent_name, group in config.agents.items():
        if not group.enabled:
            continue
        try:
            entry = registry.get(agent_name)
            entry.config_model(**group.params)
        except KeyError:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Unknown agent type '{agent_name}'.",
                    agent_name=agent_name,
                    field_path=f"agents.{agent_name}",
                )
            )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Agent '{agent_name}': {e}",
                    agent_name=agent_name,
                    field_path=f"agents.{agent_name}.params",
                )
            )

    # Capture cross-agent warnings from builder._cross_validate
    oracle_present = config.market.oracle is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimulationBuilder._cross_validate(config, oracle_present)
    for w in caught:
        issues.append(
            ValidationIssue(
                severity="warning",
                message=str(w.message),
            )
        )

    return ValidationResult(issues=issues)


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
    # Validation types
    "ValidationIssue",
    "ValidationResult",
    # Category taxonomy
    "CATEGORIES",
    # Manifest
    "get_full_manifest",
]
