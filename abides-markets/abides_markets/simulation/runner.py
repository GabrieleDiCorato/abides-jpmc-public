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
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from abides_core.abides import run as abides_run
from abides_core.agent import Agent
from abides_core.run_result import KernelRunResult
from abides_core.utils import parse_logs_df
from abides_markets.agents.exchange_agent import ExchangeAgent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.config_system import compile as compile_config
from abides_markets.config_system.compiler import derive_seed
from abides_markets.config_system.models import SimulationConfig
from abides_markets.utils import config_add_agents

from .extractors import ResultExtractor
from .metrics import (
    compute_agent_pnl,
    compute_equity_curve,
    compute_execution_metrics,
    compute_l1_close,
    compute_l1_series,
    compute_l2_series,
    compute_liquidity_metrics,
    compute_trade_attribution,
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

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_simulation(
    config: SimulationConfig,
    *,
    profile: ResultProfile = ResultProfile.SUMMARY,
    log_dir: str | None = None,
    extractors: list[ResultExtractor] | None = None,
    runtime_agents: list[TradingAgent] | None = None,
    oracle_instance: Any | None = None,
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
    runtime_agents:
        Optional list of pre-built :class:`TradingAgent` instances to inject
        into the simulation after compilation.  These agents are appended
        to the compiled agent list with a properly regenerated latency model.
        Use this for agents that cannot be expressed as declarative
        ``AgentGroupConfig`` entries (e.g. agents wrapping non-serializable
        runtime state).  Each agent's ``category`` attribute is preserved
        if already set; agents without a category get ``"runtime"``.
        **Note:** runtime agents bypass the config-system seed derivation
        and risk-guard pipeline — the caller is responsible for configuring
        them.
    oracle_instance:
        Optional pre-built oracle object to use instead of building one from
        the config's ``market.oracle`` section.  Required when the config
        uses :class:`~abides_markets.config_system.ExternalDataOracleConfig`
        (a marker type that cannot be compiled directly).  Forwarded to
        :func:`~abides_markets.config_system.compile`.

    Returns
    -------
    SimulationResult
        Frozen, thread-safe value object.  No live agent references are
        retained after this function returns.
    """
    runtime = compile_config(config, oracle_instance=oracle_instance)

    # -- Inject runtime agents (post-compile) --
    if runtime_agents:
        next_id = len(runtime["agents"])
        latency_rng = np.random.RandomState(
            seed=derive_seed(runtime["seed"], "runtime_agents")
        )
        config_add_agents(runtime, runtime_agents, latency_rng)
        for i, agent in enumerate(runtime_agents):
            if agent.id is None or agent.id < 0:
                agent.id = next_id + i
            if not getattr(agent, "category", ""):
                agent.category = "runtime"

    effective_log_dir = log_dir if log_dir is not None else uuid4().hex

    result, agents = abides_run(
        runtime,
        log_dir=effective_log_dir,
        kernel_random_state=runtime["random_state_kernel"],
    )

    return _extract_result(result, agents, config, runtime, profile, extractors or [])


def run_batch(
    configs: list[SimulationConfig],
    *,
    profile: ResultProfile = ResultProfile.SUMMARY,
    n_workers: int | None = None,
    extractors: list[ResultExtractor] | None = None,
    log_dir_prefix: str | None = None,
    worker_initializer: Callable[[], None] | None = None,
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
    worker_initializer:
        Optional callable invoked once per worker process before any
        simulation runs.  Typically used to register custom agent types
        (via ``@register_agent``) that must be available during
        ``compile_config()`` in spawned processes.  Must be **picklable**
        (top-level function, not a lambda or closure).

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
    with ctx.Pool(
        processes=effective_workers,
        initializer=worker_initializer,
    ) as pool:
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
    result, agents = abides_run(
        runtime,
        log_dir=log_dir,
        kernel_random_state=runtime["random_state_kernel"],
    )
    return _extract_result(result, agents, config, runtime, profile, extractors)


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------


def _extract_result(
    result: KernelRunResult,
    agents_list: list[Agent],
    config: SimulationConfig,
    runtime: dict[str, Any],
    profile: ResultProfile,
    extractors: list[ResultExtractor],
) -> SimulationResult:
    """Build a :class:`SimulationResult` from kernel run output."""

    # -- Identify exchange and trading agents ---------------------------------
    exchange = agents_list[0]  # ExchangeAgent is always id=0
    if not isinstance(exchange, ExchangeAgent):
        raise RuntimeError(
            f"Expected agents[0] to be ExchangeAgent but got {type(exchange).__name__}"
        )

    trading_agents: list[TradingAgent] = [
        a for a in agents_list[1:] if isinstance(a, TradingAgent)
    ]

    symbols = list(exchange.order_books.keys())

    # -- Simulation metadata --------------------------------------------------
    seed = runtime["seed"]
    sim_start_ns = int(runtime["start_time"])
    sim_end_ns = int(runtime["stop_time"])
    wall_clock_s = result.elapsed.total_seconds()

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

        l1_series: L1Snapshots | None = None
        l2_series: L2Snapshots | None = None
        trades: list[TradeAttribution] | None = None

        if ResultProfile.L1_SERIES in profile:
            l1_series = _extract_l1_series(book_log2)

        if ResultProfile.L2_SERIES in profile:
            l2_series = _extract_l2_series(book_log2)

        if ResultProfile.TRADE_ATTRIBUTION in profile:
            trades = _extract_trades(order_book)

        markets[symbol] = MarketSummary(
            symbol=symbol,
            l1_close=l1_close,
            liquidity=liquidity,
            l1_series=l1_series,
            l2_series=l2_series,
            trades=trades,
        )

    # -- Per-agent PnL --------------------------------------------------------
    agent_data: list[AgentData] = []
    if ResultProfile.AGENT_PNL in profile:
        exchange_last_trades = {
            sym: ob.last_trade for sym, ob in exchange.order_books.items()
        }
        # Collect per-symbol liquidity for execution metrics
        symbol_liquidity = {sym: mkt.liquidity for sym, mkt in markets.items()}
        extract_equity = ResultProfile.EQUITY_CURVE in profile
        for agent in trading_agents:
            equity_curve = _extract_equity_curve(agent) if extract_equity else None
            agent_data.append(
                _extract_agent_data(
                    agent, exchange_last_trades, symbol_liquidity, equity_curve  # type: ignore[arg-type]
                )
            )

    # -- Agent logs -----------------------------------------------------------
    logs_df: pd.DataFrame | None = None
    if ResultProfile.AGENT_LOGS in profile:
        raw_df = parse_logs_df(agents_list)
        # Ensure the four guaranteed base columns are correctly typed
        raw_df["EventTime"] = pd.array(raw_df["EventTime"], dtype="Int64")
        raw_df["agent_id"] = pd.array(raw_df["agent_id"], dtype="Int64")
        raw_df["EventType"] = raw_df["EventType"].astype(str)
        raw_df["agent_type"] = raw_df["agent_type"].astype(str)
        logs_df = raw_df

    # -- Custom extractors ----------------------------------------------------
    extensions: dict[str, Any] = {}
    for extractor in extractors:
        extensions[extractor.key] = extractor.extract(result, agents_list)

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
    return compute_l1_close(book_log2)


def _extract_liquidity(
    exchange: ExchangeAgent, symbol: str, order_book: Any
) -> LiquidityMetrics:
    """Build LiquidityMetrics from MetricTracker and order book state."""
    has_trackers = (
        hasattr(exchange, "metric_trackers") and symbol in exchange.metric_trackers
    )
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

    # Build VWAP trade tuples from order book history (EXEC entries)
    vwap_trades: list[tuple[int, int]] = []
    if hasattr(order_book, "history"):
        for entry in order_book.history:
            if entry.get("type") == "EXEC" and entry.get("price") is not None:
                vwap_trades.append((int(entry["price"]), int(entry["quantity"])))

    return compute_liquidity_metrics(
        vwap_trades,
        pct_time_no_bid=pct_no_bid,
        pct_time_no_ask=pct_no_ask,
        total_exchanged_volume=total_vol,
        last_trade_cents=last_trade,
    )


def _extract_trades(order_book: Any) -> list[TradeAttribution]:
    """Build a list of :class:`TradeAttribution` from EXEC entries in order book history."""
    if not hasattr(order_book, "history"):
        return []
    exec_entries = [
        entry for entry in order_book.history if entry.get("type") == "EXEC"
    ]
    return compute_trade_attribution(exec_entries)


def _extract_equity_curve(agent: TradingAgent) -> EquityCurve | None:
    """Build an :class:`EquityCurve` from FILL_PNL events in agent.log.

    Returns ``None`` if the agent has no FILL_PNL events.
    """
    if not hasattr(agent, "log") or not agent.log:
        return None

    fill_events: list[tuple[int, int, int]] = []
    for entry in agent.log:
        # entry = (timestamp_ns, event_type, event_data)
        if len(entry) < 3 or entry[1] != "FILL_PNL":
            continue
        data = entry[2]
        if not isinstance(data, dict):
            continue
        nav = data.get("nav")
        peak = data.get("peak_nav")
        if nav is None or peak is None:
            continue
        fill_events.append((int(entry[0]), int(nav), int(peak)))

    return compute_equity_curve(fill_events)


def _extract_l1_series(book_log2: list[dict]) -> L1Snapshots:
    """Build L1Snapshots from book_log2."""
    return compute_l1_series(book_log2)


def _extract_l2_series(book_log2: list[dict]) -> L2Snapshots:
    """Build L2Snapshots directly from book_log2."""
    return compute_l2_series(book_log2)


def _extract_agent_data(
    agent: TradingAgent,
    exchange_last_trades: dict[str, int],
    symbol_liquidity: dict[str, LiquidityMetrics],
    equity_curve: EquityCurve | None = None,
) -> AgentData:
    """Build AgentData for a single TradingAgent.

    Uses exchange last-trade prices for mark-to-market to avoid
    calling ``agent.mark_to_market()`` (which has logging side-effects and
    can raise ``KeyError`` if the agent never observed a trade).
    """
    holdings = dict(agent.holdings)

    # Build a merged price map: prefer exchange last-trade, fall back to agent's
    merged_prices: dict[str, int | None] = {}
    for symbol in holdings:
        if symbol == "CASH":
            continue
        price = exchange_last_trades.get(symbol)
        if price is None:
            price = agent.last_trade.get(symbol, 0)
        merged_prices[symbol] = price

    exec_metrics = _extract_execution_metrics(agent, symbol_liquidity)

    return compute_agent_pnl(
        holdings=holdings,
        starting_cash_cents=agent.starting_cash,
        last_trade_prices=merged_prices,
        basket_value_cents=agent.basket_size * agent.nav_diff,
        agent_id=agent.id,
        agent_type=agent.type or type(agent).__name__,
        agent_name=agent.name or f"agent_{agent.id}",
        agent_category=getattr(agent, "category", ""),
        execution_metrics=exec_metrics,
        equity_curve=equity_curve,
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


def _extract_execution_metrics(
    agent: TradingAgent,
    symbol_liquidity: dict[str, LiquidityMetrics],
) -> ExecutionMetrics | None:
    """Build ExecutionMetrics for execution-category agents (duck-typed).

    Returns ``None`` for non-execution agents or when required attributes
    are missing.
    """
    # Duck-type: execution agents expose execution_history, quantity, executed_quantity
    execution_history: list[dict] | None = getattr(agent, "execution_history", None)
    target_qty: int | None = getattr(agent, "quantity", None)
    filled_qty: int | None = getattr(agent, "executed_quantity", None)
    if execution_history is None or target_qty is None or filled_qty is None:
        return None

    # Build fill tuples from execution history
    fills: list[tuple[int, int]] = []
    for fill in execution_history:
        price = fill.get("fill_price")
        qty = fill.get("quantity", 0)
        if price is not None and qty > 0:
            fills.append((int(price), int(qty)))

    # Arrival price: mid-price from the agent's known_bids/known_asks at first order
    arrival: int | None = None
    last_bid = getattr(agent, "last_bid", None)
    last_ask = getattr(agent, "last_ask", None)
    if last_bid is not None and last_ask is not None:
        arrival = (int(last_bid) + int(last_ask)) // 2

    # Session VWAP and total volume from liquidity metrics
    symbol: str | None = getattr(agent, "symbol", None)
    session_vwap: int | None = None
    total_volume: int = 0
    if symbol is not None and symbol in symbol_liquidity:
        liq = symbol_liquidity[symbol]
        session_vwap = liq.vwap_cents
        total_volume = liq.total_exchanged_volume

    return compute_execution_metrics(
        fills=fills,
        target_quantity=target_qty,
        filled_quantity=filled_qty,
        session_vwap_cents=session_vwap,
        total_volume=total_volume,
        arrival_price_cents=arrival,
    )
