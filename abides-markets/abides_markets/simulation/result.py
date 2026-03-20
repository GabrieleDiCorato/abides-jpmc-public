"""Pydantic v2 value objects representing an ABIDES simulation result.

All models are *frozen* (immutable after construction) and *arbitrary_types_allowed*
to accommodate numpy arrays and pandas DataFrames.  Numpy arrays are additionally
marked ``writeable=False`` in ``model_post_init`` so that mutation is impossible
even via the underlying buffer — making the entire object tree safe to share
across threads without locking.

Hierarchy
---------
SimulationResult
├── metadata: SimulationMetadata
├── markets: dict[str, MarketSummary]
│   ├── l1_close: L1Close          (always — O(1) extraction cost)
│   ├── liquidity: LiquidityMetrics (always — from MetricTracker)
│   ├── l1_series: L1Snapshots | None   (ResultProfile.L1_SERIES)
│   └── l2_series: L2Snapshots | None   (ResultProfile.L2_SERIES)
├── agents: list[AgentData]
├── logs: DataFrame[RawLogsSchema] | None   (ResultProfile.AGENT_LOGS)
├── extensions: dict[str, Any]
└── profile: ResultProfile
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import DataFrame
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from .profiles import ResultProfile
from .schemas import L1DataFrameSchema, L2DataFrameSchema, OrderLogsSchema, RawLogsSchema, _ORDER_EVENT_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Return *arr* with ``writeable=False``.  No copy if already read-only."""
    if arr.flags.writeable:
        arr = arr.copy()
        arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------------
# SimulationMetadata
# ---------------------------------------------------------------------------

class SimulationMetadata(BaseModel):
    """Simulation-level identifiers and timing."""

    model_config = ConfigDict(frozen=True)

    seed: int
    """RNG seed used for the run."""

    tickers: list[str]
    """Symbols traded in the simulation."""

    sim_start_ns: int
    """Simulation start time (nanoseconds, Unix epoch)."""

    sim_end_ns: int
    """Simulation stop time (nanoseconds, Unix epoch)."""

    wall_clock_elapsed_s: float
    """Wall-clock seconds elapsed during the discrete-event loop."""

    config_snapshot: dict[str, Any]
    """JSON-serialisable subset of the ``SimulationConfig`` that produced this result."""


# ---------------------------------------------------------------------------
# L1Close — single snapshot, always extracted (O(1))
# ---------------------------------------------------------------------------

class L1Close(BaseModel):
    """Best bid/ask at the last logged book event (proxy for market close)."""

    model_config = ConfigDict(frozen=True)

    time_ns: int
    """Timestamp of the snapshot (nanoseconds, Unix epoch)."""

    bid_price_cents: Optional[int] = None
    """Best bid in integer cents; ``None`` if the book had no bid at close."""

    ask_price_cents: Optional[int] = None
    """Best ask in integer cents; ``None`` if the book had no ask at close."""


# ---------------------------------------------------------------------------
# LiquidityMetrics — from ExchangeAgent.MetricTracker
# ---------------------------------------------------------------------------

class LiquidityMetrics(BaseModel):
    """Market-microstructure summary stats for one symbol."""

    model_config = ConfigDict(frozen=True)

    pct_time_no_bid: float
    """Percentage of simulated trading time with an empty bid side (0–100)."""

    pct_time_no_ask: float
    """Percentage of simulated trading time with an empty ask side (0–100)."""

    total_exchanged_volume: int
    """Total shares traded across the session."""

    last_trade_cents: Optional[int] = None
    """Price of the last executed trade in integer cents; ``None`` if no trades."""

    vwap_cents: Optional[int] = None
    """Volume-weighted average price in integer cents; ``None`` if no trades.

    Derived from ``OrderBook.buy_transactions`` and ``OrderBook.sell_transactions``.
    """


# ---------------------------------------------------------------------------
# L1Snapshots — full time-series, optional (ResultProfile.L1_SERIES)
# ---------------------------------------------------------------------------

class L1Snapshots(BaseModel):
    """Full L1 bid/ask time-series stored as parallel read-only numpy arrays.

    All five arrays share the same length *T* (number of book events logged).
    Price/quantity arrays use ``pd.NA``-compatible ``Int64`` semantics:
    ``None``/``np.nan`` entries indicate an empty book side at that tick.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    times_ns: np.ndarray
    """Shape (T,), dtype int64.  Event timestamps (ns, Unix epoch)."""

    bid_prices: np.ndarray
    """Shape (T,), dtype object (int or None).  Best bid price in cents."""

    bid_quantities: np.ndarray
    """Shape (T,), dtype object (int or None).  Quantity at best bid."""

    ask_prices: np.ndarray
    """Shape (T,), dtype object (int or None).  Best ask price in cents."""

    ask_quantities: np.ndarray
    """Shape (T,), dtype object (int or None).  Quantity at best ask."""

    @model_validator(mode="after")
    def _freeze(self) -> "L1Snapshots":
        object.__setattr__(self, "times_ns", _freeze_array(self.times_ns))
        # bid/ask arrays may contain None — stored as object dtype, can't freeze in-place
        for field in ("bid_prices", "bid_quantities", "ask_prices", "ask_quantities"):
            arr = getattr(self, field)
            if arr.dtype != object:
                object.__setattr__(self, field, _freeze_array(arr))
        return self

    @pa.check_types
    def as_dataframe(self) -> DataFrame[L1DataFrameSchema]:
        """Return a validated :class:`~pandas.DataFrame` with schema :class:`~abides_markets.simulation.schemas.L1DataFrameSchema`."""
        return pd.DataFrame(
            {
                "time_ns": pd.array(self.times_ns, dtype="Int64"),
                "bid_price_cents": pd.array(self.bid_prices, dtype="Int64"),
                "bid_qty": pd.array(self.bid_quantities, dtype="Int64"),
                "ask_price_cents": pd.array(self.ask_prices, dtype="Int64"),
                "ask_qty": pd.array(self.ask_quantities, dtype="Int64"),
            }
        )

    @field_serializer("times_ns", "bid_prices", "bid_quantities", "ask_prices", "ask_quantities")
    def _serialize_array(self, arr: np.ndarray) -> list:
        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v) for v in arr.tolist()]


# ---------------------------------------------------------------------------
# L2Snapshots — sparse, optional (ResultProfile.L2_SERIES)
# ---------------------------------------------------------------------------

class L2Snapshots(BaseModel):
    """Full sparse L2 order-book snapshot series.

    Stored as a timestamp array plus two Python lists of variable-length
    level lists.  This is the natural representation of ``book_log2`` and
    avoids the zero-padding artefacts of ``OrderBook.get_L2_snapshots()``.

    ``bids[t]`` and ``asks[t]`` are lists of ``(price_cents, qty)`` tuples,
    ordered best-to-worst.  The length of each inner list equals the
    instantaneous book depth at time *t* — an empty list means a one-sided
    market (no resting orders on that side).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    times_ns: np.ndarray
    """Shape (T,), dtype int64.  Snapshot timestamps (ns, Unix epoch)."""

    bids: list[list[tuple[int, int]]]
    """Length T.  Each element is a variable-length list of (price_cents, qty) tuples."""

    asks: list[list[tuple[int, int]]]
    """Length T.  Each element is a variable-length list of (price_cents, qty) tuples."""

    @model_validator(mode="after")
    def _freeze(self) -> "L2Snapshots":
        object.__setattr__(self, "times_ns", _freeze_array(self.times_ns))
        return self

    @pa.check_types
    def as_dataframe(self) -> DataFrame[L2DataFrameSchema]:
        """Return a validated long/tidy :class:`~pandas.DataFrame` with schema :class:`~abides_markets.simulation.schemas.L2DataFrameSchema`.

        Each row represents one resting price level on one side at one timestamp.
        Only populated levels appear — zero-padded phantom entries are banned.
        """
        rows: list[dict] = []
        for t, ts in enumerate(self.times_ns):
            for level, (price, qty) in enumerate(self.bids[t]):
                rows.append({"time_ns": int(ts), "side": "bid", "level": level, "price_cents": price, "qty": qty})
            for level, (price, qty) in enumerate(self.asks[t]):
                rows.append({"time_ns": int(ts), "side": "ask", "level": level, "price_cents": price, "qty": qty})
        if not rows:
            return pd.DataFrame(columns=["time_ns", "side", "level", "price_cents", "qty"]).astype(
                {"time_ns": "Int64", "level": "Int64", "price_cents": "Int64", "qty": "Int64"}
            )
        df = pd.DataFrame(rows).astype(
            {"time_ns": "Int64", "level": "Int64", "price_cents": "Int64", "qty": "Int64"}
        )
        return df

    @field_serializer("times_ns")
    def _serialize_times(self, arr: np.ndarray) -> list:
        return arr.tolist()


# ---------------------------------------------------------------------------
# MarketSummary — per-symbol container
# ---------------------------------------------------------------------------

class MarketSummary(BaseModel):
    """All extracted data for a single symbol."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbol: str
    """Trading symbol (ticker)."""

    l1_close: L1Close
    """Last recorded L1 snapshot; always present regardless of profile."""

    liquidity: LiquidityMetrics
    """Session-level liquidity and volume metrics; always present."""

    l1_series: Optional[L1Snapshots] = None
    """Full L1 time-series; ``None`` unless ``ResultProfile.L1_SERIES`` was set."""

    l2_series: Optional[L2Snapshots] = None
    """Full sparse L2 series; ``None`` unless ``ResultProfile.L2_SERIES`` was set."""


# ---------------------------------------------------------------------------
# AgentData — per-agent PnL summary
# ---------------------------------------------------------------------------

class AgentData(BaseModel):
    """Final state and performance summary for one trading agent."""

    model_config = ConfigDict(frozen=True)

    agent_id: int
    agent_type: str
    agent_name: str

    final_holdings: dict[str, int]
    """Holdings at simulation end: ``{"CASH": <cents>, "<TICKER>": <shares>, ...}``."""

    starting_cash_cents: int
    """Initial cash position in cents."""

    mark_to_market_cents: int
    """Portfolio value at close (cash + shares × last_trade price) in cents."""

    pnl_cents: int
    """Absolute PnL in cents: ``mark_to_market_cents - starting_cash_cents``."""

    pnl_pct: float
    """PnL as percentage of starting cash: ``pnl_cents / starting_cash_cents * 100``."""


# ---------------------------------------------------------------------------
# SimulationResult — top-level value object
# ---------------------------------------------------------------------------

class SimulationResult(BaseModel):
    """Immutable, thread-safe, JSON-serialisable simulation result.

    Construct via :func:`~abides_markets.simulation.run_simulation` or
    :func:`~abides_markets.simulation.run_batch` — do not instantiate directly.

    Parameters
    ----------
    metadata:
        Simulation parameters and timing.
    markets:
        Per-symbol market data.  Always contains ``l1_close`` and ``liquidity``;
        ``l1_series`` / ``l2_series`` are populated only if the corresponding
        :class:`~abides_markets.simulation.ResultProfile` flags were set.
    agents:
        Per-agent PnL summary for all :class:`~abides_markets.agents.TradingAgent`
        subclasses.  Exchange agents are excluded.
    logs:
        Raw agent log DataFrame; ``None`` unless ``ResultProfile.AGENT_LOGS`` was set.
    extensions:
        Output of custom :class:`~abides_markets.simulation.ResultExtractor` plugins.
    profile:
        The :class:`~abides_markets.simulation.ResultProfile` used to produce this result.
        Consumers can inspect this to know which optional fields are populated.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    metadata: SimulationMetadata
    markets: dict[str, MarketSummary]
    agents: list[AgentData]
    logs: Optional[DataFrame[RawLogsSchema]] = None
    extensions: dict[str, Any] = {}
    profile: ResultProfile

    # ------------------------------------------------------------------ serializers

    @field_serializer("profile")
    def _serialize_profile(self, p: ResultProfile) -> str:
        return str(p)

    @field_serializer("logs")
    def _serialize_logs(self, df: Optional[pd.DataFrame]) -> Optional[list]:
        if df is None:
            return None
        # Coerce non-JSON-native types (NaN → None, numpy ints → int)
        return json.loads(df.to_json(orient="records"))

    # ------------------------------------------------------------------ public API

    def summary(self) -> str:
        """Return a concise human/LLM-readable narrative of the simulation result.

        Always succeeds regardless of which :class:`ResultProfile` was used.
        """
        lines: list[str] = []

        m = self.metadata
        lines.append(
            f"Simulation | seed={m.seed} | "
            f"tickers={', '.join(m.tickers)} | "
            f"elapsed={m.wall_clock_elapsed_s:.2f}s"
        )

        for symbol, mkt in self.markets.items():
            liq = mkt.liquidity
            close = mkt.l1_close
            bid_str = f"${close.bid_price_cents / 100:.2f}" if close.bid_price_cents is not None else "—"
            ask_str = f"${close.ask_price_cents / 100:.2f}" if close.ask_price_cents is not None else "—"
            last_str = f"${liq.last_trade_cents / 100:.2f}" if liq.last_trade_cents else "—"
            vwap_str = f"${liq.vwap_cents / 100:.2f}" if liq.vwap_cents else "—"
            lines.append(
                f"  {symbol}: last={last_str} vwap={vwap_str} "
                f"close_bid={bid_str} close_ask={ask_str} "
                f"vol={liq.total_exchanged_volume:,} "
                f"no_bid={liq.pct_time_no_bid:.1f}% no_ask={liq.pct_time_no_ask:.1f}%"
            )

        if self.agents:
            sorted_agents = sorted(self.agents, key=lambda a: a.pnl_cents, reverse=True)
            lines.append("  Top agents by PnL:")
            for a in sorted_agents[:5]:
                sign = "+" if a.pnl_cents >= 0 else ""
                lines.append(
                    f"    [{a.agent_id}] {a.agent_type}: "
                    f"MtM=${a.mark_to_market_cents / 100:.2f} "
                    f"PnL={sign}${a.pnl_cents / 100:.2f} ({sign}{a.pnl_pct:.2f}%)"
                )

        profile_flags = [f.name for f in ResultProfile if f in self.profile and f.name]
        lines.append(f"  Profile: {', '.join(profile_flags)}")
        if ResultProfile.L1_SERIES not in self.profile:
            lines.append("  (L1 time-series not extracted — re-run with ResultProfile.QUANT)")
        if ResultProfile.AGENT_LOGS not in self.profile:
            lines.append("  (Agent logs not extracted — re-run with ResultProfile.FULL)")

        return "\n".join(lines)

    def order_logs(self) -> DataFrame[OrderLogsSchema]:
        """Return the order-event subset of the log DataFrame, schema-validated.

        Raises
        ------
        RuntimeError
            If ``ResultProfile.AGENT_LOGS`` was not set when the simulation ran.
        """
        if self.logs is None or ResultProfile.AGENT_LOGS not in self.profile:
            raise RuntimeError(
                "Agent logs were not extracted. Re-run with ResultProfile.AGENT_LOGS "
                "(or ResultProfile.FULL) to access order logs."
            )
        mask = self.logs["EventType"].isin(_ORDER_EVENT_TYPES)
        filtered = self.logs.loc[mask].copy()
        # Ensure nullable int columns exist even if absent from thin logs
        for col in ("fill_price", "limit_price"):
            if col not in filtered.columns:
                filtered[col] = pd.array([None] * len(filtered), dtype="Int64")
        return filtered  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serialisable dict (no numpy arrays, no DataFrames).

        Use :meth:`to_json` for a JSON string.  Your server can store / forward
        this dict without any ABIDES-internal knowledge.
        """
        return json.loads(self.model_dump_json())

    def to_json(self) -> str:
        """Return a JSON string representation of this result."""
        return self.model_dump_json()
