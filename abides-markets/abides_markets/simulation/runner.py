"""Runner functions: run_simulation() and run_batch().

Both are pure functions — no side effects beyond writing ABIDES log files to disk.

Thread / process safety
-----------------------
* ``run_simulation`` is safe to call from multiple threads as long as each
  invocation uses a distinct ``log_dir`` (auto-assigned via UUID when omitted).
* ``run_batch`` uses ``multiprocessing`` (spawn on Windows, fork on POSIX) and
  is therefore also safe from a GIL standpoint.  Each worker process compiles
  its own ``SimulationConfig`` → runtime dict independently.
* The returned :class:`~abides_markets.simulation.SimulationResult` objects are
  frozen Pydantic models with read-only numpy arrays — inherently thread-safe.

Custom extractors in run_batch
------------------------------
Extractor objects must be **picklable** to be sent to worker processes.
:class:`~abides_markets.simulation.FunctionExtractor` wrapping a ``lambda``
is **not** picklable on most Python implementations.  Use
:class:`~abides_markets.simulation.BaseResultExtractor` subclasses instead,
or top-level functions wrapped in ``FunctionExtractor``.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from abides_markets.agents.exchange_agent import ExchangeAgent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.config_system import compile as compile_config
from abides_markets.config_system.models import SimulationConfig
from abides_core.abides import run as abides_run
from abides_core.utils import parse_logs_df

from .extractors import ResultExtractor
from .profiles import ResultProfile
from .result import (
    AgentData,
    L1Close,
    L1Snapshots,
    L2Snapshots,
    LiquidityMetrics,
    MarketSummary,
    SimulationMetadata,
    SimulationResult,
)
from .schemas import RawLogsSchema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_simulation(
    config: SimulationConfig,
    *,
    profile: ResultProfile = ResultProfile.SUMMARY,
    log_dir: Optional[str] = None,
    extractors: Optional[list[ResultExtractor]] = None,
) -> SimulationResult:
    """Run a single simulation and return a typed, immutable result.

    Parameters
    ----------
    config:
        A validated :class:`~abides_markets.config_system.SimulationConfig`.
        Build one with
        :class:`~abides_markets.config_system.SimulationBuilder` or construct
        it directly.
    profile:
        Controls which data is extracted.  Defaults to
        :attr:`~abides_markets.simulation.ResultProfile.SUMMARY` (kilobyte-scale
        output).  Use :attr:`~abides_markets.simulation.ResultProfile.QUANT`
        to add L1/L2 time-series, or
        :attr:`~abides_markets.simulation.ResultProfile.FULL` to also include
        the raw agent log DataFrame.
    log_dir:
        Directory path where ABIDES writes its compressed log files.
        Auto-assigned to a UUID-based subdirectory when ``None`` — this avoids
        the timestamp-collision hazard present in the default ABIDES behaviour.
    extractors:
        Optional list of :class:`~abides_markets.simulation.ResultExtractor`
        plugins.  Each extractor receives the raw ``end_state`` dict and
        contributes a value to ``SimulationResult.extensions``.

    Returns
    -------
    SimulationResult
        Frozen, thread-safe value object.  No live agent references are
        retained after this function returns.
    """
    runtime = compile_config(config)
    effective_log_dir = log_dir if log_dir is not None else uuid4().hex

    end_state = abides_run(
        runtime,
        log_dir=effective_log_dir,
        kernel_random_state=runtime["random_state_kernel"],
    )

    return _extract_result(end_state, config, runtime, profile, extractors or [])


def run_batch(
    configs: list[SimulationConfig],
    *,
    profile: ResultProfile = ResultProfile.SUMMARY,
    n_workers: Optional[int] = None,
    extractors: Optional[list[ResultExtractor]] = None,
    log_dir_prefix: Optional[str] = None,
) -> list[SimulationResult]:
    """Run multiple simulations in parallel and return results in input order.

    Each simulation runs in a separate process (``multiprocessing``).  Results
    are collected and returned in the same order as *configs*.

    Parameters
    ----------
    configs:
        List of :class:`~abides_markets.config_system.SimulationConfig` objects.
        Each should have an explicit integer seed (not ``"random"``) for
        reproducibility.
    profile:
        Extraction profile applied to every simulation.
    n_workers:
        Number of worker processes.  Defaults to ``os.cpu_count()``.
    extractors:
        Extractor plugins applied in every worker.  Must be **picklable**
        (avoid lambdas; use top-level functions or
        :class:`~abides_markets.simulation.BaseResultExtractor` subclasses).
    log_dir_prefix:
        Optional prefix for worker log directories.
        Worker *i* writes to ``{prefix}_{i}``; a UUID is appended when
        ``None`` to guarantee uniqueness.

    Returns
    -------
    list[SimulationResult]
        One result per input config, in the same order.
    """
    effective_workers = n_workers if n_workers is not None else os.cpu_count()

    # Build arg tuples; each worker gets a unique log dir
    args = [
        (
            cfg,
            profile,
            f"{log_dir_prefix}_{i}" if log_dir_prefix else uuid4().hex,
            extractors or [],
        )
        for i, cfg in enumerate(configs)
    ]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=effective_workers) as pool:
        results = pool.starmap(_worker, args)

    return results


# ---------------------------------------------------------------------------
# Worker (top-level so it is picklable on Windows/spawn)
# ---------------------------------------------------------------------------


def _worker(
    config: SimulationConfig,
    profile: ResultProfile,
    log_dir: str,
    extractors: list[ResultExtractor],
) -> SimulationResult:
    """Single-simulation worker for ``run_batch``.

    Compiles the config fresh inside the worker to avoid serialising numpy
    ``RandomState`` objects across the process boundary.
    """
    runtime = compile_config(config)
    end_state = abides_run(
        runtime,
        log_dir=log_dir,
        kernel_random_state=runtime["random_state_kernel"],
    )
    return _extract_result(end_state, config, runtime, profile, extractors)


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def _extract_result(
    end_state: dict[str, Any],
    config: SimulationConfig,
    runtime: dict[str, Any],
    profile: ResultProfile,
    extractors: list[ResultExtractor],
) -> SimulationResult:
    """Build a :class:`SimulationResult` from a raw ABIDES ``end_state`` dict."""

    agents_list = end_state["agents"]

    # -- Identify exchange and trading agents ---------------------------------
    exchange: ExchangeAgent = agents_list[0]  # ExchangeAgent is always id=0
    if not isinstance(exchange, ExchangeAgent):
        raise RuntimeError(
            "Expected agents[0] to be ExchangeAgent but got "
            f"{type(exchange).__name__}"
        )

    trading_agents: list[TradingAgent] = [
        a for a in agents_list[1:] if isinstance(a, TradingAgent)
    ]

    ticker = config.market.ticker
    symbols = list(exchange.order_books.keys())

    # -- Simulation metadata --------------------------------------------------
    seed = runtime["seed"]
    sim_start_ns = int(runtime["start_time"])
    sim_end_ns = int(runtime["stop_time"])
    wall_clock_s = (
        end_state.get("kernel_event_queue_elapsed_wallclock")
        or pd.Timedelta(0)
    ).total_seconds()

    config_snapshot = _safe_config_snapshot(config)

    metadata = SimulationMetadata(
        seed=seed,
        tickers=symbols,
        sim_start_ns=sim_start_ns,
        sim_end_ns=sim_end_ns,
        wall_clock_elapsed_s=wall_clock_s,
        config_snapshot=config_snapshot,
    )

    # -- Per-symbol market data -----------------------------------------------
    markets: dict[str, MarketSummary] = {}
    for symbol in symbols:
        order_book = exchange.order_books[symbol]
        book_log2 = order_book.book_log2

        l1_close = _extract_l1_close(book_log2)
        liquidity = _extract_liquidity(exchange, symbol, order_book)

        l1_series: Optional[L1Snapshots] = None
        l2_series: Optional[L2Snapshots] = None

        if ResultProfile.L1_SERIES in profile:
            l1_series = _extract_l1_series(book_log2)

        if ResultProfile.L2_SERIES in profile:
            l2_series = _extract_l2_series(book_log2)

        markets[symbol] = MarketSummary(
            symbol=symbol,
            l1_close=l1_close,
            liquidity=liquidity,
            l1_series=l1_series,
            l2_series=l2_series,
        )

    # -- Per-agent PnL --------------------------------------------------------
    agent_data: list[AgentData] = []
    if ResultProfile.AGENT_PNL in profile:
        exchange_last_trades = {
            sym: ob.last_trade for sym, ob in exchange.order_books.items()
        }
        for agent in trading_agents:
            agent_data.append(_extract_agent_data(agent, exchange_last_trades))

    # -- Agent logs -----------------------------------------------------------
    logs_df: Optional[pd.DataFrame] = None
    if ResultProfile.AGENT_LOGS in profile:
        raw_df = parse_logs_df(end_state)
        # Ensure the four guaranteed base columns are correctly typed
        raw_df["EventTime"] = pd.array(raw_df["EventTime"], dtype="Int64")
        raw_df["agent_id"] = pd.array(raw_df["agent_id"], dtype="Int64")
        raw_df["EventType"] = raw_df["EventType"].astype(str)
        raw_df["agent_type"] = raw_df["agent_type"].astype(str)
        logs_df = raw_df

    # -- Custom extractors ----------------------------------------------------
    extensions: dict[str, Any] = {}
    for extractor in extractors:
        extensions[extractor.key] = extractor.extract(end_state)

    return SimulationResult(
        metadata=metadata,
        markets=markets,
        agents=agent_data,
        logs=logs_df,
        extensions=extensions,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_l1_close(book_log2: list[dict]) -> L1Close:
    """Return an L1Close from the last entry in book_log2, or empty if no log."""
    if not book_log2:
        return L1Close(time_ns=0, bid_price_cents=None, ask_price_cents=None)

    last = book_log2[-1]
    time_ns = int(last["QuoteTime"])

    bids = last["bids"]
    asks = last["asks"]

    bid_price = int(bids[0][0]) if len(bids) > 0 else None
    ask_price = int(asks[0][0]) if len(asks) > 0 else None

    return L1Close(time_ns=time_ns, bid_price_cents=bid_price, ask_price_cents=ask_price)


def _extract_liquidity(
    exchange: ExchangeAgent, symbol: str, order_book: Any
) -> LiquidityMetrics:
    """Build LiquidityMetrics from MetricTracker and order book state."""
    has_trackers = hasattr(exchange, "metric_trackers") and symbol in exchange.metric_trackers
    if has_trackers:
        mt = exchange.metric_trackers[symbol]
        pct_no_bid = float(mt.pct_time_no_liquidity_bids)
        pct_no_ask = float(mt.pct_time_no_liquidity_asks)
        total_vol = int(mt.total_exchanged_volume)
        last_trade = int(mt.last_trade) if mt.last_trade else None
    else:
        pct_no_bid = 0.0
        pct_no_ask = 0.0
        total_vol = 0
        last_trade = None

    # If MetricTracker didn't have a last_trade, fall back to order book
    if last_trade is None and order_book.last_trade:
        last_trade = int(order_book.last_trade)

    return LiquidityMetrics(
        pct_time_no_bid=pct_no_bid,
        pct_time_no_ask=pct_no_ask,
        total_exchanged_volume=total_vol,
        last_trade_cents=last_trade,
        vwap_cents=None,  # Requires price-per-trade data not available in public book state.
                          # Compute via result.order_logs() from a FULL profile run.
    )


def _extract_l1_series(book_log2: list[dict]) -> L1Snapshots:
    """Build L1Snapshots from book_log2."""
    if not book_log2:
        empty = np.array([], dtype=np.int64)
        empty_obj = np.array([], dtype=object)
        return L1Snapshots(
            times_ns=empty,
            bid_prices=empty_obj,
            bid_quantities=empty_obj,
            ask_prices=empty_obj,
            ask_quantities=empty_obj,
        )

    times, bid_prices, bid_quantities, ask_prices, ask_quantities = [], [], [], [], []

    for entry in book_log2:
        times.append(int(entry["QuoteTime"]))
        bids = entry["bids"]
        asks = entry["asks"]

        if len(bids) > 0:
            bid_prices.append(int(bids[0][0]))
            bid_quantities.append(int(bids[0][1]))
        else:
            bid_prices.append(None)
            bid_quantities.append(None)

        if len(asks) > 0:
            ask_prices.append(int(asks[0][0]))
            ask_quantities.append(int(asks[0][1]))
        else:
            ask_prices.append(None)
            ask_quantities.append(None)

    return L1Snapshots(
        times_ns=np.array(times, dtype=np.int64),
        bid_prices=np.array(bid_prices, dtype=object),
        bid_quantities=np.array(bid_quantities, dtype=object),
        ask_prices=np.array(ask_prices, dtype=object),
        ask_quantities=np.array(ask_quantities, dtype=object),
    )


def _extract_l2_series(book_log2: list[dict]) -> L2Snapshots:
    """Build L2Snapshots directly from book_log2.

    Reads the *already-sparse* ``bids`` and ``asks`` arrays from each snapshot
    (populated by ``get_l2_bid_data()`` / ``get_l2_ask_data()`` which filter
    out empty price levels).  No zero-padding is applied.
    """
    if not book_log2:
        return L2Snapshots(times_ns=np.array([], dtype=np.int64), bids=[], asks=[])

    times = []
    bids_list: list[list[tuple[int, int]]] = []
    asks_list: list[list[tuple[int, int]]] = []

    for entry in book_log2:
        times.append(int(entry["QuoteTime"]))
        bids_list.append([(int(p), int(q)) for p, q in entry["bids"]])
        asks_list.append([(int(p), int(q)) for p, q in entry["asks"]])

    return L2Snapshots(
        times_ns=np.array(times, dtype=np.int64),
        bids=bids_list,
        asks=asks_list,
    )


def _extract_agent_data(
    agent: TradingAgent, exchange_last_trades: dict[str, int]
) -> AgentData:
    """Build AgentData for a single TradingAgent.

    Uses exchange last-trade prices for mark-to-market to avoid
    calling ``agent.mark_to_market()`` (which has logging side-effects and
    can raise ``KeyError`` if the agent never observed a trade).
    """
    holdings = dict(agent.holdings)
    cash = holdings.get("CASH", 0)

    mtm = cash + agent.basket_size * agent.nav_diff
    for symbol, shares in holdings.items():
        if symbol == "CASH":
            continue
        price = exchange_last_trades.get(symbol)
        if price is None:
            price = agent.last_trade.get(symbol, 0)
        mtm += price * shares

    starting = agent.starting_cash
    pnl = mtm - starting
    pnl_pct = (pnl / starting * 100.0) if starting != 0 else 0.0

    return AgentData(
        agent_id=agent.id,
        agent_type=agent.type or type(agent).__name__,
        agent_name=agent.name or f"agent_{agent.id}",
        final_holdings=holdings,
        starting_cash_cents=starting,
        mark_to_market_cents=mtm,
        pnl_cents=pnl,
        pnl_pct=pnl_pct,
    )


def _safe_config_snapshot(config: SimulationConfig) -> dict[str, Any]:
    """Return a JSON-serialisable subset of SimulationConfig."""
    return {
        "ticker": config.market.ticker,
        "date": config.market.date,
        "start_time": config.market.start_time,
        "end_time": config.market.end_time,
        "seed": config.simulation.seed,
        "log_level": config.simulation.log_level,
        "agent_groups": {
            name: {"count": g.count, "enabled": g.enabled}
            for name, g in config.agents.items()
        },
    }
