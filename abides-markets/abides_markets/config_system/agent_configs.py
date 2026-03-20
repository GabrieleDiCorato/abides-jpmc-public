"""Pydantic config models for each built-in agent type.

Each model declares the agent-specific parameters as Pydantic fields.
Parameters are inherited through the config class hierarchy. The base class
auto-generates ``create_agents()`` by inspecting the registered ``agent_class``
constructor and mapping fields by name. Subclasses can override
``_prepare_constructor_kwargs()`` for computed arguments that aren't simple
field-to-parameter mappings.

Required vs. optional parameters follow Pydantic semantics: fields with
defaults are optional; fields without defaults are mandatory.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from abides_core import NanosecondTime
from abides_core.utils import get_wake_time, str_to_ns
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared context passed to create_agents by the compiler
# ---------------------------------------------------------------------------
@dataclass
class AgentCreationContext:
    """Shared state from the compiler passed to every agent factory."""

    ticker: str
    mkt_open: NanosecondTime
    mkt_close: NanosecondTime
    log_orders: bool
    oracle_r_bar: int  # for derived params like SIGMA_N
    date_ns: NanosecondTime = 0  # date component in nanoseconds


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------
class BaseAgentConfig(BaseModel):
    """Common fields shared by all trading-agent configs.

    When a ``agent_class`` is registered in the registry, the default
    ``create_agents()`` implementation auto-maps config fields to the
    agent constructor by matching parameter names. Override
    ``_prepare_constructor_kwargs()`` for computed arguments.
    """

    model_config = {"extra": "forbid"}

    starting_cash: int = Field(
        default=10_000_000,
        description="Initial cash in cents ($100k = 10_000_000).",
    )
    log_orders: bool | None = Field(
        default=None,
        description="Override per-agent order logging. None = use simulation-level setting.",
    )
    computation_delay: int | None = Field(
        default=None,
        description=(
            "Per-agent-type computation delay in nanoseconds. "
            "None = use the simulation-level default_computation_delay."
        ),
    )

    # Fields excluded from automatic constructor mapping
    _EXCLUDE_FROM_KWARGS: frozenset[str] = frozenset(
        {"computation_delay"}
    )

    def create_agents(
        self,
        count: int,
        id_start: int,
        master_rng: np.random.RandomState,
        context: AgentCreationContext,
    ) -> list:
        """Create agent instances. Auto-generated from registry agent_class.

        If the registry entry for this config model has an ``agent_class``,
        the base implementation inspects the constructor signature,
        maps config fields → constructor args by name, injects
        context-provided args (id, name, type, symbol, etc.), and calls
        ``_prepare_constructor_kwargs()`` for any final transformations.

        Subclasses may override this entirely for non-standard instantiation.
        """
        # Look up the agent_class from the registry
        agent_cls = self._resolve_agent_class()
        if agent_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no registered agent_class "
                "and does not override create_agents()."
            )

        # Discover which parameters the constructor accepts
        sig = inspect.signature(agent_cls.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}

        log = self.log_orders if self.log_orders is not None else context.log_orders

        # Collect config fields that map to constructor params
        base_kwargs: dict[str, Any] = {}
        for field_name, _field_info in type(self).model_fields.items():
            if field_name in self._EXCLUDE_FROM_KWARGS:
                continue
            if field_name == "log_orders":
                base_kwargs["log_orders"] = log
            elif field_name in accepted_params:
                base_kwargs[field_name] = getattr(self, field_name)

        agents = []
        for j in range(id_start, id_start + count):
            agent_rng = np.random.RandomState(
                seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
            )
            # Start with the mapped fields, then add context injections
            kwargs = dict(base_kwargs)
            kwargs["id"] = j
            kwargs["name"] = f"{agent_cls.__name__} {j}"
            kwargs["type"] = agent_cls.__name__
            kwargs["random_state"] = agent_rng
            if "symbol" in accepted_params:
                kwargs["symbol"] = context.ticker

            # Let subclasses transform / add computed args
            kwargs = self._prepare_constructor_kwargs(
                kwargs, j, agent_rng, context
            )

            # Filter to only accepted params (in case hook added extras)
            kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}

            agents.append(agent_cls(**kwargs))
        return agents

    def _prepare_constructor_kwargs(
        self,
        kwargs: dict[str, Any],
        agent_id: int,
        agent_rng: np.random.RandomState,
        context: AgentCreationContext,
    ) -> dict[str, Any]:
        """Hook for subclasses to modify constructor kwargs before instantiation.

        Override this to add computed arguments (e.g. wakeup_time),
        convert string durations to nanoseconds, or inject dependencies
        like OrderSizeModel.

        Args:
            kwargs: Current kwargs dict (config fields + context injections).
            agent_id: The agent's sequential ID.
            agent_rng: Per-agent RNG instance.
            context: Shared compiler context.

        Returns:
            Modified kwargs dict.
        """
        return kwargs

    def _resolve_agent_class(self) -> type | None:
        """Look up the agent_class from the registry for this config model."""
        from abides_markets.config_system.registry import registry

        for entry in registry._entries.values():
            if entry.config_model is type(self):
                return entry.agent_class
        return None


# ---------------------------------------------------------------------------
# Noise Agent
# ---------------------------------------------------------------------------
class NoiseAgentConfig(BaseAgentConfig):
    """Configuration for NoiseAgent — simple agents that wake once and place a random order."""

    noise_mkt_open_offset: str = Field(
        default="-00:30:00",
        description="Offset from market open for noise wakeup window start (e.g. '-00:30:00').",
    )
    noise_mkt_close_time: str = Field(
        default="16:00:00",
        description="Time-of-day for noise wakeup window end.",
    )

    _EXCLUDE_FROM_KWARGS: frozenset[str] = frozenset(
        {"computation_delay", "noise_mkt_open_offset", "noise_mkt_close_time"}
    )

    def _prepare_constructor_kwargs(
        self, kwargs, agent_id, agent_rng, context
    ):
        from abides_markets.models import OrderSizeModel

        noise_mkt_open = context.mkt_open + str_to_ns(self.noise_mkt_open_offset)
        noise_mkt_close = context.date_ns + str_to_ns(self.noise_mkt_close_time)

        kwargs["wakeup_time"] = get_wake_time(
            noise_mkt_open, noise_mkt_close, agent_rng
        )
        kwargs["order_size_model"] = OrderSizeModel()
        kwargs["name"] = f"NoiseAgent {agent_id}"
        kwargs["type"] = "NoiseAgent"
        return kwargs


# ---------------------------------------------------------------------------
# Value Agent
# ---------------------------------------------------------------------------
class ValueAgentConfig(BaseAgentConfig):
    """Configuration for ValueAgent — Bayesian learner that estimates fundamental value."""

    r_bar: int = Field(
        default=100_000,
        description="True mean fundamental value in cents.",
    )
    kappa: float = Field(
        default=1.67e-15,
        description="Mean-reversion coefficient for agent's appraisal.",
    )
    lambda_a: float = Field(
        default=5.7e-12,
        description="Arrival rate (per nanosecond) for Poisson wakeups.",
    )
    sigma_n: int | None = Field(
        default=None,
        description="Observation noise variance. Defaults to r_bar / 100.",
    )

    def _prepare_constructor_kwargs(
        self, kwargs, agent_id, agent_rng, context
    ):
        from abides_markets.models import OrderSizeModel

        if kwargs.get("sigma_n") is None:
            kwargs["sigma_n"] = self.r_bar / 100
        kwargs["order_size_model"] = OrderSizeModel()
        kwargs["name"] = f"Value Agent {agent_id}"
        kwargs["type"] = "ValueAgent"
        return kwargs


# ---------------------------------------------------------------------------
# Momentum Agent
# ---------------------------------------------------------------------------
class MomentumAgentConfig(BaseAgentConfig):
    """Configuration for MomentumAgent — trend-follower using moving average crossover."""

    min_size: int = Field(default=1, description="Minimum order size.")
    max_size: int = Field(default=10, description="Maximum order size.")
    wake_up_freq: str = Field(
        default="37s",
        description="Wake-up frequency as duration string (e.g. '37s', '1min').",
    )
    poisson_arrival: bool = Field(
        default=True,
        description="If True, wakeup intervals are Poisson-distributed.",
    )

    def _prepare_constructor_kwargs(
        self, kwargs, agent_id, agent_rng, context
    ):
        from abides_markets.models import OrderSizeModel

        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        kwargs["order_size_model"] = OrderSizeModel()
        kwargs["name"] = f"MOMENTUM_AGENT_{agent_id}"
        kwargs["type"] = "MomentumAgent"
        return kwargs


# ---------------------------------------------------------------------------
# Adaptive Market Maker Agent
# ---------------------------------------------------------------------------
class AdaptiveMarketMakerConfig(BaseAgentConfig):
    """Configuration for AdaptiveMarketMakerAgent — inventory-skewed ladder market maker."""

    pov: float = Field(default=0.025, description="Percentage of volume per level.")
    min_order_size: int = Field(
        default=1, description="Minimum order size at any level."
    )
    window_size: Union[int, str] = Field(
        default="adaptive",
        description="Spread window in ticks or 'adaptive'.",
    )
    num_ticks: int = Field(default=10, description="Number of price levels each side.")
    wake_up_freq: str = Field(
        default="60s",
        description="Wake-up frequency as duration string.",
    )
    poisson_arrival: bool = Field(
        default=True, description="Poisson-distributed wakeups."
    )
    cancel_limit_delay: int = Field(
        default=50,
        description="Delay in nanoseconds before cancel takes effect.",
    )
    skew_beta: float = Field(default=0, description="Inventory skew parameter.")
    price_skew_param: int | None = Field(default=4, description="Price skew parameter.")
    level_spacing: float = Field(
        default=5,
        description="Spacing between price levels as fraction of spread.",
    )
    spread_alpha: float = Field(
        default=0.75,
        description="EWMA parameter for spread estimation.",
    )
    backstop_quantity: int = Field(
        default=0, description="Orders at the outermost level."
    )

    def _prepare_constructor_kwargs(
        self, kwargs, agent_id, agent_rng, context
    ):
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        kwargs["name"] = f"ADAPTIVE_POV_MARKET_MAKER_AGENT_{agent_id}"
        kwargs["type"] = "AdaptivePOVMarketMakerAgent"
        return kwargs


# ---------------------------------------------------------------------------
# POV Execution Agent
# ---------------------------------------------------------------------------
class POVExecutionAgentConfig(BaseAgentConfig):
    """Configuration for POVExecutionAgent — executes large orders as percentage of volume."""

    start_time_offset: str = Field(
        default="00:30:00",
        description="Offset from market open when execution begins.",
    )
    end_time_offset: str = Field(
        default="00:30:00",
        description="Offset before market close when execution ends.",
    )
    freq: str = Field(default="1min", description="Wake-up frequency.")
    pov: float = Field(default=0.1, description="Target % of observed volume.")
    direction: str = Field(
        default="BID",
        description="Order direction: 'BID' (buy) or 'ASK' (sell).",
    )
    quantity: int = Field(default=1_200_000, description="Total target quantity.")
    trade: bool = Field(
        default=True, description="If False, only logs without trading."
    )

    _EXCLUDE_FROM_KWARGS: frozenset[str] = frozenset(
        {"computation_delay", "start_time_offset", "end_time_offset", "freq", "direction"}
    )

    def _prepare_constructor_kwargs(
        self, kwargs, agent_id, agent_rng, context
    ):
        from abides_markets.orders import Side

        freq_ns = str_to_ns(self.freq)
        kwargs["freq"] = freq_ns
        kwargs["lookback_period"] = freq_ns
        kwargs["start_time"] = context.mkt_open + str_to_ns(self.start_time_offset)
        kwargs["end_time"] = context.mkt_close - str_to_ns(self.end_time_offset)
        kwargs["direction"] = Side.BID if self.direction.upper() == "BID" else Side.ASK
        kwargs["name"] = f"POV_EXECUTION_AGENT_{agent_id}"
        kwargs["type"] = "ExecutionAgent"
        return kwargs
