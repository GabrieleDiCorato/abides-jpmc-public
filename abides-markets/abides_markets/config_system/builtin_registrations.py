"""Register all built-in agent types in the global registry.

This module is imported by ``config_system.__init__`` to ensure all
standard agents are available without explicit user action.
"""

from abides_markets.agents import (
    AdaptiveMarketMakerAgent,
    MomentumAgent,
    NoiseAgent,
    POVExecutionAgent,
    ValueAgent,
)
from abides_markets.config_system.agent_configs import (
    AdaptiveMarketMakerConfig,
    MomentumAgentConfig,
    NoiseAgentConfig,
    POVExecutionAgentConfig,
    ValueAgentConfig,
)
from abides_markets.config_system.registry import registry

_BUILTIN_NAMES = frozenset(
    {"noise", "value", "momentum", "adaptive_market_maker", "pov_execution"}
)


def _register_builtins() -> None:
    """Register all built-in agent types. Safe to call multiple times."""
    if _BUILTIN_NAMES.issubset(registry.registered_names()):
        return  # already registered

    registry.register(
        name="noise",
        config_model=NoiseAgentConfig,
        agent_class=NoiseAgent,
        category="background",
        description=(
            "Simple agent that wakes once at a random time and places "
            "one random limit order at bid or ask."
        ),
    )

    registry.register(
        name="value",
        config_model=ValueAgentConfig,
        agent_class=ValueAgent,
        category="background",
        description=(
            "Bayesian learner that estimates fundamental value from noisy "
            "observations and trades when price deviates from estimate."
        ),
    )

    registry.register(
        name="momentum",
        config_model=MomentumAgentConfig,
        agent_class=MomentumAgent,
        category="strategy",
        description=(
            "Trend-follower using 20-bar vs 50-bar moving average crossover. "
            "Buys when MA(20) >= MA(50), sells otherwise."
        ),
    )

    registry.register(
        name="adaptive_market_maker",
        config_model=AdaptiveMarketMakerConfig,
        agent_class=AdaptiveMarketMakerAgent,
        category="market_maker",
        description=(
            "Market maker using inventory-skewed ladder strategy with "
            "adaptive spread. Places liquidity on both sides in multiple "
            "price levels."
        ),
    )

    registry.register(
        name="pov_execution",
        config_model=POVExecutionAgentConfig,
        agent_class=POVExecutionAgent,
        category="execution",
        description=(
            "Percentage-of-volume execution agent that sizes orders based "
            "on observed transacted volume to execute a large target quantity."
        ),
    )


_register_builtins()
