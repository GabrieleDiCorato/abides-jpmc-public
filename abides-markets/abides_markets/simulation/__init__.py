"""abides_markets.simulation — typed simulation runner and result wrapper.

Public API
----------
Runner functions::

    from abides_markets.simulation import run_simulation, run_batch

    result = run_simulation(config)
    results = run_batch([cfg1, cfg2], n_workers=4)

Result models::

    from abides_markets.simulation import (
        SimulationResult,
        SimulationMetadata,
        MarketSummary,
        L1Close,
        LiquidityMetrics,
        L1Snapshots,
        L2Snapshots,
        AgentData,
    )

Extraction profile::

    from abides_markets.simulation import ResultProfile

    result = run_simulation(config, profile=ResultProfile.QUANT)

Custom extractors::

    from abides_markets.simulation import FunctionExtractor, BaseResultExtractor

    ext = FunctionExtractor("n_agents", lambda e: len(e["agents"]))
    result = run_simulation(config, extractors=[ext])

DataFrame schemas (Pandera)::

    from abides_markets.simulation import (
        L1DataFrameSchema,
        L2DataFrameSchema,
        RawLogsSchema,
        OrderLogsSchema,
    )
"""

from .extractors import BaseResultExtractor, FunctionExtractor, ResultExtractor
from .metrics import (
    compute_agent_pnl,
    compute_avg_liquidity,
    compute_effective_spread,
    compute_equity_curve,
    compute_execution_metrics,
    compute_inventory_std,
    compute_l1_close,
    compute_l1_series,
    compute_l2_series,
    compute_liquidity_metrics,
    compute_lob_imbalance,
    compute_market_ott_ratio,
    compute_mean_spread,
    compute_metrics,
    compute_order_fill_rate,
    compute_resilience,
    compute_sharpe_ratio,
    compute_trade_attribution,
    compute_volatility,
    compute_vpin,
    compute_vwap,
)
from .profiles import ResultProfile
from .result import (
    AgentData,
    EquityCurve,
    ExecutionMetrics,
    L1Close,
    L1Snapshots,
    L2Snapshots,
    LiquidityMetrics,
    MarketSummary,
    SimulationMetadata,
    SimulationResult,
    TradeAttribution,
)
from .runner import run_batch, run_simulation
from .schemas import (
    L1DataFrameSchema,
    L2DataFrameSchema,
    OrderLogsSchema,
    RawLogsSchema,
)

__all__ = [
    # Runner
    "run_simulation",
    "run_batch",
    # Standalone metric computation
    "compute_metrics",
    "compute_vwap",
    "compute_liquidity_metrics",
    "compute_l1_close",
    "compute_l1_series",
    "compute_l2_series",
    "compute_trade_attribution",
    "compute_agent_pnl",
    "compute_execution_metrics",
    "compute_equity_curve",
    "compute_mean_spread",
    "compute_effective_spread",
    "compute_volatility",
    "compute_sharpe_ratio",
    "compute_avg_liquidity",
    "compute_lob_imbalance",
    "compute_inventory_std",
    "compute_market_ott_ratio",
    "compute_vpin",
    "compute_resilience",
    "compute_order_fill_rate",
    # Result models
    "SimulationResult",
    "SimulationMetadata",
    "MarketSummary",
    "L1Close",
    "LiquidityMetrics",
    "L1Snapshots",
    "L2Snapshots",
    "AgentData",
    "ExecutionMetrics",
    "EquityCurve",
    "TradeAttribution",
    # Profile
    "ResultProfile",
    # Extractors
    "ResultExtractor",
    "BaseResultExtractor",
    "FunctionExtractor",
    # Schemas
    "L1DataFrameSchema",
    "L2DataFrameSchema",
    "RawLogsSchema",
    "OrderLogsSchema",
]
