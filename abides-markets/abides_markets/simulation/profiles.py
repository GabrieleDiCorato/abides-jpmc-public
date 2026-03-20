"""ResultProfile — controls which data is extracted from a simulation's end_state.

Flags compose via bitwise OR.  Three ready-made tiers are provided:

* ``SUMMARY`` (default): metadata, per-agent PnL, market liquidity stats,
  and the last L1 snapshot at market close.  Output is on the order of
  kilobytes — suitable for logging, REST responses, and LLM tool calls.

* ``QUANT``:  adds the full L1 and sparse L2 time-series for all symbols.
  Use for backtesting, plotting, and quantitative analysis.

* ``FULL``:   adds the raw agent log DataFrame.  Use for debugging and
  deep inspection of individual agent behaviour.

The profile is passed to ``run_simulation()`` at call time.  Data **not**
covered by the profile is skipped entirely during extraction — skipped data
is never computed, not just hidden.
"""

from __future__ import annotations

from enum import Flag, auto


class ResultProfile(Flag):
    """Tiered extraction flags for :class:`~abides_markets.simulation.SimulationResult`.

    Each flag controls an extraction step inside ``_extract_result()``.
    """

    METADATA = auto()
    """Simulation parameters, timing, config snapshot."""

    AGENT_PNL = auto()
    """Per-agent final holdings, starting cash, mark-to-market, PnL."""

    LIQUIDITY = auto()
    """Per-symbol MetricTracker stats (dropout %, volume) and final L1 snapshot."""

    L1_SERIES = auto()
    """Full bid/ask L1 time-series as parallel numpy arrays."""

    L2_SERIES = auto()
    """Full sparse L2 book snapshots (variable-length levels, no zero-padding)."""

    AGENT_LOGS = auto()
    """Result of ``parse_logs_df()`` — one row per agent log entry."""

    # ------------------------------------------------------------------ tiers
    SUMMARY: "ResultProfile" = METADATA | AGENT_PNL | LIQUIDITY
    """Default tier.  Kilobyte-scale output; good for REST, LLM, alerting."""

    QUANT: "ResultProfile" = SUMMARY | L1_SERIES | L2_SERIES
    """Adds full time-series.  Use for backtesting and quantitative analysis."""

    FULL: "ResultProfile" = QUANT | AGENT_LOGS
    """All data including raw agent logs.  Primarily for debugging."""
