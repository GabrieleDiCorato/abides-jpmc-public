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

import warnings
from copy import deepcopy
from typing import Any

from abides_markets.config_system.models import SimulationConfig
from abides_markets.config_system.templates import get_template
from abides_markets.oracles.oracle import Oracle


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
        self._oracle_instance: Oracle | None = None

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
        """Set oracle parameters directly (shorthand for ``.market(oracle={...})``).

        Call ``oracle(type=None)`` to explicitly disable the oracle for
        oracle-less simulations (requires ``market.opening_price`` to be set).

        Calling with other keywords (e.g. ``oracle(r_bar=200_000)``)
        merges them into the existing oracle dict.
        """
        if "type" in kwargs and kwargs["type"] is None:
            extra = {k for k in kwargs if k != "type"}
            if extra:
                raise ValueError(
                    f"oracle(type=None) disables the oracle — extra kwargs "
                    f"would be silently discarded: {extra}"
                )
            # Explicit oracle=None
            market = self._data.setdefault("market", {})
            market["oracle"] = None
            return self
        market = self._data.setdefault("market", {})
        oracle = market.setdefault("oracle", {})
        oracle.update(kwargs)
        return self

    def oracle_instance(self, oracle: Oracle) -> SimulationBuilder:
        """Inject a pre-built oracle instance for use at runtime.

        This is the recommended path for ``ExternalDataOracle`` users:
        build the oracle externally with your chosen ``BatchDataProvider``
        or ``PointDataProvider``, then inject it here.  The config system
        should use ``oracle(type="external_data")`` (or any marker) so
        compile-time validation knows an oracle will be present.

        The injected oracle is stored on the builder and merged into the
        runtime dict during ``compile()`` (via ``build_and_compile()`` or
        manual ``compile()`` with ``oracle_instance`` kwarg).
        """
        self._oracle_instance = oracle
        # Auto-set external_data marker in config so compile knows oracle is present
        market = self._data.setdefault("market", {})
        market["oracle"] = {"type": "external_data"}
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

    def seed(self, seed: int | str) -> SimulationBuilder:
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
        3. Validates oracle-related constraints:
           - ValueAgent requires an oracle (oracle must not be None).
           - When oracle is None, opening_price must be set.

        Raises:
            pydantic.ValidationError: If the configuration is invalid.
            ValueError: If semantic constraints are violated.
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

        # Oracle-related validation
        oracle_present = (
            config.market.oracle is not None or self._oracle_instance is not None
        )
        for agent_name, group in config.agents.items():
            if not group.enabled or group.count == 0:
                continue
            if agent_name == "value" and not oracle_present:
                raise ValueError(
                    "ValueAgent requires an oracle for fundamental-value observations, "
                    "but no oracle is configured. Either set market.oracle or use "
                    "oracle_instance() to inject one."
                )
        if not oracle_present and config.market.opening_price is None:
            raise ValueError(
                "When no oracle is configured, market.opening_price must be set "
                "to provide the ExchangeAgent with a seed price "
                "(integer cents, e.g. 10_000 = $100.00)."
            )

        self._cross_validate(config, oracle_present)

        return config  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Cross-agent / cross-section consistency checks
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_validate(config: SimulationConfig, oracle_present: bool) -> None:
        """Emit warnings for semantically suspect but technically valid configs.

        These are soft checks — they emit :mod:`warnings` rather than raising,
        so a consuming dashboard can capture them via ``warnings.catch_warnings()``.
        """
        enabled = {
            name: group
            for name, group in config.agents.items()
            if group.enabled and group.count > 0
        }
        enabled_names = set(enabled)

        # Market maker without background liquidity providers
        if "adaptive_market_maker" in enabled_names and not (
            enabled_names & {"noise", "value"}
        ):
            warnings.warn(
                "adaptive_market_maker is enabled but no noise or value "
                "agents are present — the order book will have no background "
                "liquidity and the market maker may have no counterparties.",
                stacklevel=3,
            )

        # Execution agent without adequate liquidity
        if "pov_execution" in enabled_names:
            bg_count = sum(enabled[n].count for n in ("noise", "value") if n in enabled)
            if bg_count < 10:
                warnings.warn(
                    f"pov_execution is enabled but only {bg_count} background "
                    f"agent(s) are present. POV targeting needs meaningful "
                    f"background volume — consider adding more noise/value agents.",
                    stacklevel=3,
                )

        # start_time >= end_time — now caught by MarketConfig model validator;
        # retained here as defense-in-depth for configs constructed without
        # model validation.
        if config.market.start_time >= config.market.end_time:
            raise ValueError(
                f"Market start_time ({config.market.start_time}) is not before "
                f"end_time ({config.market.end_time}) — the trading window "
                f"is empty or inverted."
            )

        # POV execution window exceeds market hours
        if "pov_execution" in enabled_names:
            from abides_markets.config_system.agent_configs import str_to_ns

            pov_params = enabled["pov_execution"].params
            start_off = pov_params.get("start_time_offset", "00:05:00")
            end_off = pov_params.get("end_time_offset", "00:05:00")
            try:
                mkt_ns = str_to_ns(config.market.end_time) - str_to_ns(
                    config.market.start_time
                )
                window_consumed = str_to_ns(start_off) + str_to_ns(end_off)
                if window_consumed >= mkt_ns:
                    warnings.warn(
                        "POV execution offsets consume the entire market "
                        "window — the execution agent will have no time to trade.",
                        stacklevel=3,
                    )
            except Exception:
                pass  # Best-effort; don't fail on parse issues

        # Excessive agent count
        total_agents = sum(g.count for g in enabled.values())
        if total_agents > 10_000:
            warnings.warn(
                f"Total enabled agent count is {total_agents:,}. Simulations "
                f"with >10,000 agents may be very slow.",
                stacklevel=3,
            )

        # No agents at all
        if total_agents == 0:
            warnings.warn(
                "No agents are enabled — the simulation will have no participants.",
                stacklevel=3,
            )

    def get_oracle_instance(self) -> Oracle | None:
        """Return the pre-built oracle instance, if any was injected via oracle_instance()."""
        return self._oracle_instance

    def build_and_compile(self) -> dict[str, Any]:
        """Build, validate, and compile in one step.

        Convenience method that calls ``build()`` then ``compile()``,
        automatically passing through any pre-built oracle instance.
        """
        from abides_markets.config_system.compiler import compile as compile_config

        config = self.build()
        return compile_config(config, oracle_instance=self._oracle_instance)

    def to_dict(self) -> dict[str, Any]:
        """Return the raw config dict (before validation)."""
        return deepcopy(self._data)
