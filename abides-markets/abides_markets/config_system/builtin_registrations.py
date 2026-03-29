"""Register all built-in agent types in the global registry.

This module is imported by ``config_system.__init__`` to ensure all
standard agents are available without explicit user action.

For custom agents, use the ``@register_agent`` decorator directly on
your config class — see ``ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md``.
"""

from abides_markets.agents import (
    AdaptiveMarketMakerAgent,
    MeanReversionAgent,
    MomentumAgent,
    NoiseAgent,
    POVExecutionAgent,
    TWAPExecutionAgent,
    ValueAgent,
)
from abides_markets.config_system.agent_configs import (
    AdaptiveMarketMakerConfig,
    MeanReversionAgentConfig,
    MomentumAgentConfig,
    NoiseAgentConfig,
    POVExecutionAgentConfig,
    TWAPExecutionAgentConfig,
    ValueAgentConfig,
)
from abides_markets.config_system.registry import register_agent

_ALREADY_REGISTERED = False


def _register_builtins() -> None:
    """Register all built-in agent types. Safe to call multiple times."""
    global _ALREADY_REGISTERED
    if _ALREADY_REGISTERED:
        return
    _ALREADY_REGISTERED = True

    # We apply @register_agent imperatively here (rather than on the class
    # definitions in agent_configs.py) to avoid circular imports — agent
    # classes live in abides_markets.agents which must not import the config
    # system.  The pattern is equivalent to the decorator form.

    register_agent(
        "noise",
        agent_class=NoiseAgent,
        category="background",
        description=(
            "Simple agent that wakes once at a random time and places "
            "one random limit order at bid or ask."
        ),
        requires_oracle=False,
        typical_count_range=(50, 5000),
        recommended_with=("value", "adaptive_market_maker"),
    )(NoiseAgentConfig)

    register_agent(
        "value",
        agent_class=ValueAgent,
        category="background",
        description=(
            "Bayesian learner that estimates fundamental value from noisy "
            "observations and trades when price deviates from estimate."
        ),
        requires_oracle=True,
        typical_count_range=(10, 500),
        recommended_with=("noise",),
    )(ValueAgentConfig)

    register_agent(
        "momentum",
        agent_class=MomentumAgent,
        category="strategy",
        description=(
            "Trend-follower using 20-bar vs 50-bar moving average crossover. "
            "Buys when MA(20) >= MA(50), sells otherwise."
        ),
        requires_oracle=False,
        typical_count_range=(1, 50),
        recommended_with=("noise", "value"),
    )(MomentumAgentConfig)

    register_agent(
        "mean_reversion",
        agent_class=MeanReversionAgent,
        category="strategy",
        description=(
            "Contrarian agent using Bollinger-band / z-score mean reversion. "
            "Buys when price is unusually low, sells when unusually high."
        ),
        requires_oracle=False,
        typical_count_range=(1, 50),
        recommended_with=("noise", "value"),
    )(MeanReversionAgentConfig)

    register_agent(
        "adaptive_market_maker",
        agent_class=AdaptiveMarketMakerAgent,
        category="market_maker",
        description=(
            "Market maker using inventory-skewed ladder strategy with "
            "adaptive spread. Places liquidity on both sides in multiple "
            "price levels."
        ),
        requires_oracle=False,
        typical_count_range=(1, 5),
        recommended_with=("noise", "value"),
    )(AdaptiveMarketMakerConfig)

    register_agent(
        "pov_execution",
        agent_class=POVExecutionAgent,
        category="execution",
        description=(
            "Percentage-of-volume execution agent that sizes orders based "
            "on observed transacted volume to execute a large target quantity."
        ),
        requires_oracle=False,
        typical_count_range=(1, 1),
        recommended_with=("noise", "value", "adaptive_market_maker"),
    )(POVExecutionAgentConfig)

    register_agent(
        "twap_execution",
        agent_class=TWAPExecutionAgent,
        category="execution",
        description=(
            "Time-weighted average price execution agent that divides a "
            "parent order into uniform time slices with catch-up logic."
        ),
        requires_oracle=False,
        typical_count_range=(1, 1),
        recommended_with=("noise", "value", "adaptive_market_maker"),
    )(TWAPExecutionAgentConfig)


_register_builtins()
