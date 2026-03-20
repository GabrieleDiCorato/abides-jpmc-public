"""Fluent builder API for constructing SimulationConfig.

Usage::

    config = (SimulationBuilder()
        .from_template("rmsc04")
        .market(ticker="AAPL", date="20210205")
        .enable_agent("noise", count=1000)
        .enable_agent("value", count=102, r_bar=100_000)
        .disable_agent("momentum")
        .latency(type="deterministic")
        .seed(42)
        .build())
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Union

from abides_markets.config_system.models import (
    SimulationConfig,
)
from abides_markets.config_system.templates import get_template


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. overlay values take precedence."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class SimulationBuilder:
    """Fluent builder for SimulationConfig.

    Supports template stacking, per-agent overrides, and a final
    ``build()`` that validates and returns a ``SimulationConfig``.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def from_template(self, name: str) -> SimulationBuilder:
        """Deep-merge a template into the current config.

        Multiple templates can be stacked — later ones override earlier ones.
        """
        template = get_template(name)
        self._data = _deep_merge(self._data, template)
        return self

    def market(self, **kwargs: Any) -> SimulationBuilder:
        """Set market parameters (ticker, date, start_time, end_time).

        Nested structures can be passed as dicts::

            .market(ticker="AAPL", oracle={"type": "mean_reverting", "r_bar": 150_000})
        """
        market = self._data.setdefault("market", {})
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(market.get(k), dict):
                market[k] = _deep_merge(market[k], v)
            else:
                market[k] = v
        return self

    def oracle(self, **kwargs: Any) -> SimulationBuilder:
        """Set oracle parameters directly (shorthand for ``.market(oracle={...})``)."""
        market = self._data.setdefault("market", {})
        oracle = market.setdefault("oracle", {})
        oracle.update(kwargs)
        return self

    def exchange(self, **kwargs: Any) -> SimulationBuilder:
        """Set exchange parameters directly."""
        market = self._data.setdefault("market", {})
        exchange = market.setdefault("exchange", {})
        exchange.update(kwargs)
        return self

    def enable_agent(self, name: str, count: int, **params: Any) -> SimulationBuilder:
        """Enable an agent type with the given count and optional parameters.

        Merges with any existing params for this agent type.
        """
        agents = self._data.setdefault("agents", {})
        existing = agents.get(name, {})
        existing_params = existing.get("params", {})
        existing_params.update(params)
        agents[name] = {
            "enabled": True,
            "count": count,
            "params": existing_params,
        }
        return self

    def agent_computation_delay(self, name: str, delay: int) -> SimulationBuilder:
        """Set the computation delay for a specific agent type.

        This overrides the simulation-level default_computation_delay for
        all agents of the named type.
        """
        agents = self._data.setdefault("agents", {})
        group = agents.setdefault(name, {"enabled": True, "count": 0, "params": {}})
        group["params"]["computation_delay"] = delay
        return self

    def disable_agent(self, name: str) -> SimulationBuilder:
        """Disable an agent type."""
        agents = self._data.setdefault("agents", {})
        if name in agents:
            agents[name]["enabled"] = False
        else:
            agents[name] = {"enabled": False, "count": 0, "params": {}}
        return self

    def latency(self, **kwargs: Any) -> SimulationBuilder:
        """Set latency model parameters."""
        infra = self._data.setdefault("infrastructure", {})
        lat = infra.setdefault("latency", {})
        lat.update(kwargs)
        return self

    def computation_delay(self, delay: int) -> SimulationBuilder:
        """Set the default computation delay in nanoseconds."""
        infra = self._data.setdefault("infrastructure", {})
        infra["default_computation_delay"] = delay
        return self

    def seed(self, seed: Union[int, str]) -> SimulationBuilder:
        """Set the RNG seed. Use ``"random"`` for a fresh seed."""
        sim = self._data.setdefault("simulation", {})
        sim["seed"] = seed
        return self

    def log_level(self, level: str) -> SimulationBuilder:
        """Set stdout log level."""
        sim = self._data.setdefault("simulation", {})
        sim["log_level"] = level
        return self

    def log_orders(self, enabled: bool) -> SimulationBuilder:
        """Set global order logging."""
        sim = self._data.setdefault("simulation", {})
        sim["log_orders"] = enabled
        return self

    def build(self) -> SimulationConfig:
        """Validate and return the SimulationConfig.

        Performs two-phase validation:
        1. Validates the overall config structure via Pydantic.
        2. Validates each agent group's params against its registered config model,
           catching unknown parameters, type errors, and missing required fields
           at build-time rather than compile-time.

        Raises:
            pydantic.ValidationError: If the configuration is invalid.
        """
        from abides_markets.config_system.registry import registry

        config = SimulationConfig.model_validate(self._data)

        # Eager validation: validate agent params against registry config models
        for agent_name, group in config.agents.items():
            if not group.enabled:
                continue
            try:
                entry = registry.get(agent_name)
            except KeyError as e:
                raise ValueError(
                    f"Agent type '{agent_name}' is not registered. "
                    f"Available types: {', '.join(registry.registered_names())}"
                ) from e
            # Instantiate the config model to validate params
            try:
                entry.config_model(**group.params)
            except Exception as e:
                raise ValueError(
                    f"Invalid parameters for agent type '{agent_name}': {e}"
                ) from e

        return config

    def to_dict(self) -> dict[str, Any]:
        """Return the raw config dict (before validation)."""
        return deepcopy(self._data)
