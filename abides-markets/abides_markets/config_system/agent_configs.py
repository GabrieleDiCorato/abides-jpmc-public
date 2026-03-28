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
from typing import Any, Literal, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

from abides_core import NanosecondTime
from abides_core.utils import get_wake_time, str_to_ns

# Base set of fields excluded from automatic constructor kwarg mapping.
# Risk-related fields are bundled into a RiskConfig object by the base class.
# Subclass _EXCLUDE sets should extend this via ``_BASE_EXCLUDE | frozenset({...})``.
_BASE_EXCLUDE: frozenset[str] = frozenset(
    {
        "computation_delay",
        "position_limit",
        "position_limit_clamp",
        "max_drawdown",
        "max_order_rate",
        "order_rate_window",
    }
)


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
    oracle_r_bar: int | None  # auto-inherited by ValueAgent when not explicitly set
    date_ns: NanosecondTime = 0  # date component in nanoseconds
    oracle_kappa: float | None = None  # auto-inherited by ValueAgent
    oracle_sigma_s: float | None = None  # auto-inherited by ValueAgent


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
        ge=0,
        description="Initial cash in cents ($100k = 10_000_000).",
        examples=[10_000_000, 100_000_000],
        json_schema_extra={"unit": "cents"},
    )
    log_orders: bool | None = Field(
        default=None,
        description="Override per-agent order logging. None = use simulation-level setting.",
    )
    computation_delay: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Per-agent-type computation delay in nanoseconds. "
            "None = use the simulation-level default_computation_delay."
        ),
        json_schema_extra={"unit": "nanoseconds"},
    )
    position_limit: int | None = Field(
        default=None,
        description=(
            "Per-symbol position limit (in shares). "
            "None = no limit.  Symmetric: allows [-N, +N]."
        ),
    )
    position_limit_clamp: bool = Field(
        default=False,
        description=(
            "When True, orders that would breach the position limit are "
            "reduced (clamped) instead of fully rejected."
        ),
    )
    max_drawdown: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum loss from starting_cash in cents before the circuit "
            "breaker permanently halts the agent.  None = disabled."
        ),
        json_schema_extra={"unit": "cents"},
    )
    max_order_rate: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum orders per rate window before the circuit breaker "
            "permanently halts the agent.  None = disabled."
        ),
    )
    order_rate_window: str = Field(
        default="1min",
        description=(
            "Tumbling window duration for the order-rate circuit breaker "
            "as a duration string (e.g. '1min', '30s', '2min')."
        ),
        examples=["1min", "30s", "2min"],
        json_schema_extra={"format": "duration"},
    )

    # Fields excluded from automatic constructor mapping.
    # Risk fields are bundled into a RiskConfig object instead.
    _EXCLUDE_FROM_KWARGS: frozenset[str] = _BASE_EXCLUDE

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
            kwargs = self._prepare_constructor_kwargs(kwargs, j, agent_rng, context)

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
        from abides_markets.models import RiskConfig

        kwargs["risk_config"] = RiskConfig(
            position_limit=self.position_limit,
            position_limit_clamp=self.position_limit_clamp,
            max_drawdown=self.max_drawdown,
            max_order_rate=self.max_order_rate,
            order_rate_window_ns=str_to_ns(self.order_rate_window),
        )
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
        description=(
            "Offset from market open for noise wakeup window start. "
            "Format: '[±]HH:MM:SS' — a negative value means before market "
            "open (e.g. '-00:30:00' = 30 minutes before open)."
        ),
        examples=["-00:30:00", "-01:00:00", "00:00:00"],
        json_schema_extra={"format": "duration"},
    )
    noise_mkt_close_time: str = Field(
        default="16:00:00",
        description=(
            "Time-of-day (HH:MM:SS) for noise wakeup window end. The agent "
            "wakes at a uniformly random time between the offset-adjusted "
            "market open and this time."
        ),
        examples=["16:00:00", "12:00:00"],
    )

    _EXCLUDE_FROM_KWARGS: frozenset[str] = _BASE_EXCLUDE | frozenset(
        {
            "noise_mkt_open_offset",
            "noise_mkt_close_time",
        }
    )

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        from abides_markets.models import OrderSizeModel

        kwargs = super()._prepare_constructor_kwargs(
            kwargs, agent_id, agent_rng, context
        )

        noise_mkt_open = context.mkt_open + str_to_ns(self.noise_mkt_open_offset)
        noise_mkt_close = context.date_ns + str_to_ns(self.noise_mkt_close_time)

        if noise_mkt_open >= noise_mkt_close:
            raise ValueError(
                f"NoiseAgent wakeup window is empty or inverted: "
                f"open offset '{self.noise_mkt_open_offset}' → {noise_mkt_open}, "
                f"close time '{self.noise_mkt_close_time}' → {noise_mkt_close}."
            )

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
    """Configuration for ValueAgent — Bayesian learner that estimates fundamental value.

    **Financial meaning**: The ValueAgent is an *informed trader* that observes
    noisy signals of the true fundamental price from the oracle and maintains
    a Bayesian posterior estimate.  Its parameters define the prior belief and
    observation quality:

    - ``r_bar``: The agent's prior mean for the fundamental price (in integer
      cents, e.g. $100.00 = 10_000).  Typically matches the oracle's ``r_bar``
      so the agent's prior is centered on the true long-run mean.  Auto-inherited
      from the oracle config when ``None``.
    - ``kappa`` / ``mean_reversion_half_life``: Mean-reversion speed — how quickly
      the agent's belief reverts to ``r_bar``.  A larger kappa means the agent
      trusts the long-run mean more relative to recent observations.
      Specified as a duration half-life (e.g. '48d').  Auto-inherited
      from the oracle config when ``None``.
    - ``sigma_s``: Variance of fundamental price shocks in the agent's model.
      Controls how much the agent expects the fundamental to move per unit time.
      Auto-inherited from the oracle config when ``None``.
    - ``sigma_n``: Observation noise variance — how noisy the agent's oracle
      observations are.  This is agent-specific (different agents can have
      different observation quality) and defaults to ``r_bar / 100``.
    - ``mean_wakeup_gap``: Average interval between agent wake-ups as a
      duration string (e.g. '175s').

    **Parameter inheritance**: When ``r_bar``, ``kappa``, or ``sigma_s`` are
    ``None``, they are auto-inherited from the oracle config at compile time.
    This ensures the agent's prior matches the generating process by default.
    Explicit values always take precedence.
    """

    r_bar: int | None = Field(
        default=None,
        description=(
            "Prior mean fundamental value in cents.  None = auto-inherit "
            "from oracle config (recommended).  E.g. $100.00 = 10_000."
        ),
        examples=[100_000, 50_000],
        json_schema_extra={"unit": "cents"},
    )
    mean_reversion_half_life: str | None = Field(
        default=None,
        description=(
            "Half-life of mean reversion as a duration string (e.g. '48d'). "
            "None = auto-inherit from oracle config (recommended). "
            "Shorter half-lives mean stronger mean-reversion toward r_bar."
        ),
        examples=["48d", "4.8d", "30d"],
        json_schema_extra={"format": "duration"},
    )
    sigma_s: float | None = Field(
        default=None,
        description=(
            "Fundamental shock variance in the agent's model.  None = "
            "auto-inherit from oracle config (recommended).  Controls how "
            "much price movement the agent expects per unit time."
        ),
    )
    mean_wakeup_gap: str = Field(
        default="175s",
        description=(
            "Average time between agent wake-ups as a duration string "
            "(e.g. '175s', '3min'). Internally converted to a Poisson "
            "arrival rate."
        ),
        examples=["175s", "60s", "5min"],
        json_schema_extra={"format": "duration"},
    )
    sigma_n: int | None = Field(
        default=None,
        description=(
            "Observation noise variance (agent-specific — NOT inherited "
            "from oracle).  Defaults to r_bar / 100.  Larger values mean "
            "noisier observations; 0 means perfect information."
        ),
    )
    depth_spread: int = Field(
        default=2,
        ge=1,
        description="Depth spread multiplier for passive order price adjustment.",
    )

    _EXCLUDE_FROM_KWARGS: frozenset[str] = _BASE_EXCLUDE | frozenset(
        {
            "mean_reversion_half_life",
            "mean_wakeup_gap",
        }
    )

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        import math

        from abides_markets.models import OrderSizeModel

        kwargs = super()._prepare_constructor_kwargs(
            kwargs, agent_id, agent_rng, context
        )

        # Auto-inherit r_bar from oracle if not explicitly set
        r_bar = self.r_bar
        if r_bar is None:
            if context.oracle_r_bar is not None:
                r_bar = context.oracle_r_bar
            else:
                raise ValueError(
                    "ValueAgentConfig.r_bar is None and no oracle r_bar available. "
                    "Either set r_bar explicitly or provide an oracle with r_bar."
                )
        kwargs["r_bar"] = r_bar

        # Auto-inherit kappa from oracle if not explicitly set.
        # context.oracle_kappa is already in per-ns units (converted by compiler).
        hl = self.mean_reversion_half_life
        if hl is not None:
            kwargs["kappa"] = math.log(2) / str_to_ns(hl)
        elif context.oracle_kappa is not None:
            kwargs["kappa"] = context.oracle_kappa
        else:
            raise ValueError(
                "ValueAgentConfig.mean_reversion_half_life is None and no oracle "
                "kappa available. Either set mean_reversion_half_life explicitly "
                "or provide an oracle with a half-life."
            )

        # Auto-inherit sigma_s from oracle if not explicitly set
        sigma_s = self.sigma_s
        if sigma_s is None:
            if context.oracle_sigma_s is not None:
                sigma_s = context.oracle_sigma_s
            else:
                raise ValueError(
                    "ValueAgentConfig.sigma_s is None and no oracle sigma_s available. "
                    "Either set sigma_s explicitly or provide an oracle with sigma_s."
                )
        kwargs["sigma_s"] = sigma_s

        # Convert mean_wakeup_gap (duration string) → lambda_a (per-ns rate)
        kwargs["lambda_a"] = 1.0 / str_to_ns(self.mean_wakeup_gap)

        # sigma_n is agent-specific: default to r_bar / 100
        if kwargs.get("sigma_n") is None:
            kwargs["sigma_n"] = r_bar / 100

        kwargs["order_size_model"] = OrderSizeModel()
        kwargs["name"] = f"Value Agent {agent_id}"
        kwargs["type"] = "ValueAgent"
        return kwargs


# ---------------------------------------------------------------------------
# Momentum Agent
# ---------------------------------------------------------------------------
class MomentumAgentConfig(BaseAgentConfig):
    """Configuration for MomentumAgent — trend-follower using moving average crossover."""

    min_size: int = Field(default=1, ge=1, description="Minimum order size in shares.")
    max_size: int = Field(default=10, ge=1, description="Maximum order size in shares.")
    wake_up_freq: str = Field(
        default="37s",
        description=(
            "Wake-up frequency as a duration string. Supported formats: "
            "'Ns' (seconds), 'Nmin' (minutes), 'Nh' (hours), "
            "'HH:MM:SS'. The default 37s is calibrated to produce "
            "realistic trading frequency for trend-following agents."
        ),
        examples=["37s", "1min", "00:01:00"],
        json_schema_extra={"format": "duration"},
    )
    poisson_arrival: bool = Field(
        default=True,
        description=(
            "If True, wakeup intervals are Poisson-distributed around "
            "wake_up_freq (more realistic). If False, wakeups are "
            "exactly periodic."
        ),
    )
    short_window: int = Field(
        default=20,
        ge=1,
        description="Number of bars for the fast (short) moving average.",
    )
    long_window: int = Field(
        default=50,
        ge=1,
        description="Number of bars for the slow (long) moving average.",
    )
    subscribe: bool = Field(
        default=False,
        description=(
            "If True, subscribe to L2 market data instead of polling. "
            "Subscription mode receives updates asynchronously."
        ),
    )

    @model_validator(mode="after")
    def _check_window_order(self) -> MomentumAgentConfig:
        if self.short_window > self.long_window:
            raise ValueError(
                f"short_window ({self.short_window}) must be <= long_window ({self.long_window})"
            )
        return self

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        from abides_markets.models import OrderSizeModel

        kwargs = super()._prepare_constructor_kwargs(
            kwargs, agent_id, agent_rng, context
        )
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        kwargs["order_size_model"] = OrderSizeModel()
        kwargs["name"] = f"MOMENTUM_AGENT_{agent_id}"
        kwargs["type"] = "MomentumAgent"
        return kwargs


# ---------------------------------------------------------------------------
# Adaptive Market Maker Agent
# ---------------------------------------------------------------------------
class AdaptiveMarketMakerConfig(BaseAgentConfig):
    """Configuration for AdaptiveMarketMakerAgent — inventory-skewed ladder market maker.

    This agent places limit orders on both sides of the book at multiple
    price levels (a "ladder").  It adapts its spread to observed market
    conditions and skews quotes based on inventory to manage risk.
    """

    pov: float = Field(
        default=0.025,
        ge=0,
        le=1,
        description=(
            "Target participation-of-volume fraction per price level "
            "(0.0–1.0).  Controls order size at each level: "
            "order_qty = pov × stream_history_volume.  "
            "0.025 = 2.5%% of recent volume per level."
        ),
    )
    min_order_size: int = Field(
        default=1,
        ge=1,
        description="Minimum order size in shares at any price level.",
    )
    window_size: Union[int, str] = Field(
        default="adaptive",
        description=(
            "Spread window size. An integer sets a fixed spread in ticks "
            "(minimum 1). The string 'adaptive' lets the agent dynamically "
            "estimate the spread from recent order book data using an "
            "EWMA controlled by spread_alpha."
        ),
        examples=["adaptive", 5, 10],
    )
    num_ticks: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of price levels placed on each side of the book. "
            "More levels provide deeper liquidity but spread the agent's "
            "capital thinner."
        ),
    )
    wake_up_freq: str = Field(
        default="60s",
        description=(
            "Wake-up frequency as a duration string. Controls how often "
            "the market maker re-evaluates and updates its quotes. "
            "Supported formats: 'Ns', 'Nmin', 'Nh', 'HH:MM:SS'."
        ),
        examples=["60s", "30s", "2min"],
        json_schema_extra={"format": "duration"},
    )
    poisson_arrival: bool = Field(
        default=True,
        description=(
            "If True, wakeup intervals are Poisson-distributed around "
            "wake_up_freq (more realistic). If False, exactly periodic."
        ),
    )
    cancel_limit_delay: int = Field(
        default=50,
        ge=0,
        description=(
            "Delay in nanoseconds between deciding to cancel an order and "
            "the cancel message being sent. Simulates internal processing "
            "latency. Most users can leave this at the default."
        ),
        json_schema_extra={"unit": "nanoseconds"},
    )
    skew_beta: float = Field(
        default=0,
        description=(
            "Inventory skew strength. Controls how much the agent shifts "
            "its quotes in response to inventory imbalance. Positive values "
            "skew the mid-price away from the side where the agent is long, "
            "encouraging mean-reversion of inventory to zero. "
            "0 = symmetric quoting (no skew)."
        ),
    )
    price_skew_param: int | None = Field(
        default=4,
        description=(
            "Controls non-linear price skewing across ladder levels. "
            "Higher values concentrate skew at levels closer to the mid, "
            "while lower values spread skew more evenly. "
            "None = disable price skewing entirely."
        ),
    )
    level_spacing: float = Field(
        default=5,
        ge=0,
        description=(
            "Spacing multiplier between price levels as a fraction of the "
            "estimated spread. E.g. 5.0 means each level is spaced "
            "5 × spread_estimate apart. Higher values create wider ladders."
        ),
    )
    spread_alpha: float = Field(
        default=0.75,
        ge=0,
        le=1,
        description=(
            "EWMA smoothing parameter for spread estimation (0.0–1.0). "
            "Higher values weight recent observations more heavily "
            "(faster adaptation). Lower values produce smoother, more "
            "stable spread estimates."
        ),
    )
    backstop_quantity: int = Field(
        default=0,
        ge=0,
        description=(
            "Extra order quantity placed at the outermost price level on "
            "each side. Acts as a safety net to capture large sweeping "
            "orders.  0 = no backstop."
        ),
    )
    anchor: str = Field(
        default="middle",
        description=(
            "Anchor point for the price ladder relative to the mid-price. "
            "'top' = ladder top at mid, 'bottom' = ladder bottom at mid, "
            "'middle' = ladder centered on mid."
        ),
    )
    subscribe: bool = Field(
        default=False,
        description=(
            "If True, subscribe to L2 market data instead of polling. "
            "Subscription mode receives book updates asynchronously."
        ),
    )
    subscribe_freq: str = Field(
        default="10s",
        description=(
            "Frequency at which to receive market data updates in "
            "subscribe mode, as a duration string (e.g. '10s', '1s')."
        ),
        examples=["10s", "5s", "1s"],
        json_schema_extra={"format": "duration"},
    )
    subscribe_num_levels: int = Field(
        default=1,
        ge=1,
        description="Number of order book levels to subscribe to.",
    )
    min_imbalance: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description=(
            "Minimum book imbalance ratio to trigger order adjustment. "
            "Values close to 1.0 require extreme imbalance."
        ),
    )

    _EXCLUDE_FROM_KWARGS: frozenset[str] = _BASE_EXCLUDE | frozenset(
        {
            "subscribe_freq",
        }
    )

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        kwargs = super()._prepare_constructor_kwargs(
            kwargs, agent_id, agent_rng, context
        )
        kwargs["wake_up_freq"] = str_to_ns(self.wake_up_freq)
        kwargs["subscribe_freq"] = str_to_ns(self.subscribe_freq)
        kwargs["name"] = f"ADAPTIVE_POV_MARKET_MAKER_AGENT_{agent_id}"
        kwargs["type"] = "AdaptivePOVMarketMakerAgent"
        return kwargs


# ---------------------------------------------------------------------------
# POV Execution Agent
# ---------------------------------------------------------------------------
class POVExecutionAgentConfig(BaseAgentConfig):
    """Configuration for POVExecutionAgent — executes large orders as percentage of volume.

    This agent slices a parent order into child orders, participating at a
    target fraction of observed market volume (POV) within a time window.
    """

    start_time_offset: str = Field(
        default="00:30:00",
        description=(
            "Offset from market open when execution begins.  "
            "Format: 'HH:MM:SS'.  '00:30:00' = start 30 min after open."
        ),
        examples=["00:30:00", "00:00:00", "01:00:00"],
        json_schema_extra={"format": "duration"},
    )
    end_time_offset: str = Field(
        default="00:30:00",
        description=(
            "Offset before market close when execution stops.  "
            "Format: 'HH:MM:SS'.  '00:30:00' = stop 30 min before close."
        ),
        examples=["00:30:00", "00:00:00"],
        json_schema_extra={"format": "duration"},
    )
    freq: str = Field(
        default="1min",
        description=(
            "Wake-up frequency as a duration string. Controls how often "
            "the agent checks volume and submits child orders. "
            "Supported: 'Ns', 'Nmin', 'Nh', 'HH:MM:SS'."
        ),
        examples=["1min", "30s", "5min"],
        json_schema_extra={"format": "duration"},
    )
    pov: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description=(
            "Target participation-of-volume fraction (0.0–1.0).  "
            "Each wakeup the agent submits: child_qty = pov × recent_volume.  "
            "0.1 = target 10%% of market volume."
        ),
    )
    direction: Literal["BID", "ASK"] = Field(
        default="BID",
        description="Order direction: 'BID' (buy) or 'ASK' (sell).",
    )
    quantity: int = Field(
        default=1_200_000,
        ge=1,
        description=(
            "Total parent order quantity in shares.  The agent stops "
            "when this many shares have been executed or when the "
            "execution window closes."
        ),
        examples=[1_200_000, 500_000],
    )
    trade: bool = Field(
        default=True,
        description=(
            "If True, the agent submits live orders.  If False, it only "
            "logs what it *would* trade (dry-run / shadow mode)."
        ),
    )

    _EXCLUDE_FROM_KWARGS: frozenset[str] = _BASE_EXCLUDE | frozenset(
        {
            "start_time_offset",
            "end_time_offset",
            "freq",
            "direction",
        }
    )

    def _prepare_constructor_kwargs(self, kwargs, agent_id, agent_rng, context):
        from abides_markets.orders import Side

        kwargs = super()._prepare_constructor_kwargs(
            kwargs, agent_id, agent_rng, context
        )
        freq_ns = str_to_ns(self.freq)
        kwargs["freq"] = freq_ns
        kwargs["lookback_period"] = freq_ns
        kwargs["start_time"] = context.mkt_open + str_to_ns(self.start_time_offset)
        kwargs["end_time"] = context.mkt_close - str_to_ns(self.end_time_offset)

        if kwargs["start_time"] >= kwargs["end_time"]:
            raise ValueError(
                f"POV execution window is empty or inverted: "
                f"start offset '{self.start_time_offset}' → {kwargs['start_time']}, "
                f"end offset '{self.end_time_offset}' → {kwargs['end_time']}."
            )

        kwargs["direction"] = Side.BID if self.direction.upper() == "BID" else Side.ASK
        kwargs["name"] = f"POV_EXECUTION_AGENT_{agent_id}"
        kwargs["type"] = "ExecutionAgent"
        return kwargs
