"""Standalone metric computation functions.

All functions accept plain Python data (dicts, lists, tuples, ints) and
return the same Pydantic models used by :class:`SimulationResult`.  This
allows external consumers to compute the canonical metric set without
running a full simulation.

Example — compute VWAP from a trade list::

    from abides_markets.simulation.metrics import compute_vwap

    vwap = compute_vwap([(10_000, 100), (10_200, 50)])
    # 10_066  (integer cents, floor-divided)

Example — compute all market + agent metrics at once::

    from abides_markets.simulation.metrics import compute_metrics

    bundle = compute_metrics(
        book_log2=book_snapshots,
        exec_trades=exec_entries,
        agent_holdings=[
            {"holdings": {"CASH": 10_000_000, "AAPL": 50},
             "starting_cash_cents": 10_000_000,
             "agent_id": 1, "agent_type": "noise"},
        ],
        last_trade_prices={"AAPL": 10_050},
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .result import (
    AgentData,
    EquityCurve,
    ExecutionMetrics,
    L1Close,
    L1Snapshots,
    L2Snapshots,
    LiquidityMetrics,
    MarketSummary,
    TradeAttribution,
)

# Nanoseconds per calendar year (365.25 days).
_NS_PER_YEAR: int = int(365.25 * 24 * 3600 * 1_000_000_000)

# ---------------------------------------------------------------------------
# Primitive metric helpers
# ---------------------------------------------------------------------------


def compute_vwap(trades: Sequence[tuple[int, int]]) -> int | None:
    """Volume-weighted average price from ``(price_cents, quantity)`` pairs.

    Returns ``None`` when *trades* is empty or total quantity is zero.
    Uses floor division (integer cents).
    """
    total_value = 0
    total_qty = 0
    for price, qty in trades:
        if qty > 0:
            total_value += int(price) * int(qty)
            total_qty += int(qty)
    return total_value // total_qty if total_qty > 0 else None


# ---------------------------------------------------------------------------
# Market-level metrics
# ---------------------------------------------------------------------------


def compute_liquidity_metrics(
    trades: Sequence[tuple[int, int]] = (),
    *,
    pct_time_no_bid: float = 0.0,
    pct_time_no_ask: float = 0.0,
    total_exchanged_volume: int = 0,
    last_trade_cents: int | None = None,
) -> LiquidityMetrics:
    """Build :class:`LiquidityMetrics` from pre-computed stats and a trade list.

    Parameters
    ----------
    trades:
        Sequence of ``(price_cents, quantity)`` tuples used to derive VWAP.
        Pass an empty sequence if VWAP is not needed.
    pct_time_no_bid:
        Percentage of session time with an empty bid side (0–100).
    pct_time_no_ask:
        Percentage of session time with an empty ask side (0–100).
    total_exchanged_volume:
        Total shares traded across the session.
    last_trade_cents:
        Price of the last executed trade (integer cents), or ``None``.
    """
    return LiquidityMetrics(
        pct_time_no_bid=pct_time_no_bid,
        pct_time_no_ask=pct_time_no_ask,
        total_exchanged_volume=total_exchanged_volume,
        last_trade_cents=last_trade_cents,
        vwap_cents=compute_vwap(trades),
    )


# ---------------------------------------------------------------------------
# Microstructure metrics (Tier 1) — derived from L1 snapshots
# ---------------------------------------------------------------------------


def compute_mean_spread(l1: L1Snapshots) -> float | None:
    """Time-averaged quoted spread from two-sided L1 snapshots.

    Considers only rows where both bid and ask prices are present.
    Returns the arithmetic mean spread in cents, or ``None`` if no
    two-sided rows exist.
    """
    if len(l1.times_ns) == 0:
        return None

    total = 0.0
    count = 0
    for i in range(len(l1.times_ns)):
        bid = l1.bid_prices[i]
        ask = l1.ask_prices[i]
        if bid is not None and ask is not None:
            total += float(ask) - float(bid)
            count += 1

    return total / count if count > 0 else None


def compute_effective_spread(
    fills: Sequence[tuple[int, int, int]],
    l1: L1Snapshots,
) -> float | None:
    """Average effective spread from fills joined to the nearest L1 mid-price.

    Parameters
    ----------
    fills:
        Sequence of ``(fill_price_cents, quantity, fill_time_ns)`` tuples.
    l1:
        L1 snapshot series with ``times_ns``, ``bid_prices``, ``ask_prices``.

    Returns
    -------
    float | None
        Mean effective spread in cents:
        ``mean(2 * |fill_price - mid_price|)`` over fills with a valid
        two-sided L1 quote.  ``None`` if no fills could be matched.
    """
    if not fills or len(l1.times_ns) == 0:
        return None

    # Pre-compute two-sided mid-prices and their timestamps.
    mid_times: list[int] = []
    mid_prices: list[float] = []
    for i in range(len(l1.times_ns)):
        bid = l1.bid_prices[i]
        ask = l1.ask_prices[i]
        if bid is not None and ask is not None:
            mid_times.append(int(l1.times_ns[i]))
            mid_prices.append((float(bid) + float(ask)) / 2.0)

    if not mid_times:
        return None

    mid_times_arr = np.array(mid_times, dtype=np.int64)

    total = 0.0
    count = 0
    for fill_price, _qty, fill_time in fills:
        idx = int(np.searchsorted(mid_times_arr, fill_time, side="right")) - 1
        if idx < 0:
            idx = 0
        mid = mid_prices[idx]
        total += 2.0 * abs(float(fill_price) - mid)
        count += 1

    return total / count if count > 0 else None


def compute_volatility(l1: L1Snapshots) -> float | None:
    """Annualised mid-price return volatility from two-sided L1 snapshots.

    Algorithm (per Rohan §2.1):

    1. Filter two-sided rows.
    2. Compute mid-price = (bid + ask) / 2.
    3. Compute returns: r_t = mid_t / mid_{t-1} - 1.
    4. Annualise from median inter-snapshot interval.

    Returns ``None`` if fewer than 30 two-sided return observations.
    """
    if len(l1.times_ns) == 0:
        return None

    mids: list[float] = []
    times: list[int] = []
    for i in range(len(l1.times_ns)):
        bid = l1.bid_prices[i]
        ask = l1.ask_prices[i]
        if bid is not None and ask is not None:
            mids.append((float(bid) + float(ask)) / 2.0)
            times.append(int(l1.times_ns[i]))

    if len(mids) < 31:
        return None

    mids_arr = np.array(mids)
    returns = mids_arr[1:] / mids_arr[:-1] - 1.0

    if len(returns) < 30:
        return None

    std_r = float(np.std(returns, ddof=1))
    if std_r == 0.0:
        return None

    dt = np.diff(np.array(times, dtype=np.int64))
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        return None

    return std_r * float(np.sqrt(_NS_PER_YEAR / median_dt))


def compute_sharpe_ratio(curve: EquityCurve | None) -> float | None:
    """Annualised Sharpe ratio from an equity curve.

    Algorithm (per Rohan §1.3):

    1. Compute fill-by-fill NAV returns.
    2. Sharpe = mean(r) / std(r) * sqrt(NS_PER_YEAR / median_dt).

    Returns ``None`` if the curve is ``None``, has fewer than 30 observations,
    or the return standard deviation is zero.
    """
    if curve is None or len(curve.nav_cents) < 31:
        return None

    navs = np.array(curve.nav_cents, dtype=np.float64)
    returns = navs[1:] / navs[:-1] - 1.0

    if len(returns) < 30:
        return None

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))

    if std_r == 0.0:
        return None

    dt = np.diff(np.array(curve.times_ns, dtype=np.int64))
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        return None

    return (mean_r / std_r) * float(np.sqrt(_NS_PER_YEAR / median_dt))


# ---------------------------------------------------------------------------
# Microstructure metrics (Tier 2)
# ---------------------------------------------------------------------------


def compute_avg_liquidity(
    l1: L1Snapshots,
) -> tuple[float | None, float | None]:
    """Average resting quantity at best bid and ask from two-sided L1 snapshots.

    Returns ``(avg_bid_qty, avg_ask_qty)``.  Each component is ``None`` when
    no two-sided rows exist.
    """
    if len(l1.times_ns) == 0:
        return None, None

    bid_total = 0.0
    ask_total = 0.0
    count = 0
    for i in range(len(l1.times_ns)):
        bq = l1.bid_quantities[i]
        aq = l1.ask_quantities[i]
        if bq is not None and aq is not None:
            bid_total += float(bq)
            ask_total += float(aq)
            count += 1

    if count == 0:
        return None, None
    return bid_total / count, ask_total / count


def compute_lob_imbalance(
    l1: L1Snapshots,
) -> tuple[float | None, float | None]:
    """LOB imbalance mean and std from L1 snapshots (Cont, Kukanov & Stoikov 2014).

    .. math::

       I_t = \\frac{Q^{\\text{bid}}_t - Q^{\\text{ask}}_t}
                    {Q^{\\text{bid}}_t + Q^{\\text{ask}}_t} \\in [-1, 1]

    Considers only rows where both ``bid_qty`` and ``ask_qty`` are non-zero.
    Returns ``(mean, std)``; both ``None`` if no valid rows exist.
    """
    if len(l1.times_ns) == 0:
        return None, None

    values: list[float] = []
    for i in range(len(l1.times_ns)):
        bq = l1.bid_quantities[i]
        aq = l1.ask_quantities[i]
        if bq is not None and aq is not None:
            bq_f = float(bq)
            aq_f = float(aq)
            denom = bq_f + aq_f
            if denom > 0:
                values.append((bq_f - aq_f) / denom)

    if not values:
        return None, None

    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def compute_inventory_std(
    fills: Sequence[tuple[str, int]],
) -> float | None:
    """Standard deviation of intraday inventory reconstructed from fills.

    Parameters
    ----------
    fills:
        Sequence of ``(side, quantity)`` tuples where *side* is ``"BUY"`` or
        ``"SELL"``.  A buy increases inventory; a sell decreases it.

    Returns ``None`` if fewer than 2 fills.
    """
    if len(fills) < 2:
        return None

    inventory = 0
    positions: list[int] = []
    for side, qty in fills:
        if side == "BUY":
            inventory += qty
        else:
            inventory -= qty
        positions.append(inventory)

    return float(np.std(positions, ddof=1))


def compute_market_ott_ratio(
    n_submissions: int,
    n_fills: int,
) -> float | None:
    """Market-wide order-to-trade ratio (MiFID II RTS 9).

    Returns ``n_submissions / n_fills``, or ``None`` if *n_fills* is zero.
    """
    if n_fills <= 0:
        return None
    return n_submissions / n_fills


def compute_l1_close(book_log2: list[dict[str, Any]]) -> L1Close:
    """Extract :class:`L1Close` from the last entry in *book_log2*.

    Returns an empty ``L1Close`` (time_ns=0, both prices ``None``) when
    *book_log2* is empty.
    """
    if not book_log2:
        return L1Close(time_ns=0, bid_price_cents=None, ask_price_cents=None)

    last = book_log2[-1]
    time_ns = int(last["QuoteTime"])
    bids = last["bids"]
    asks = last["asks"]
    bid_price = int(bids[0][0]) if len(bids) > 0 else None
    ask_price = int(asks[0][0]) if len(asks) > 0 else None
    return L1Close(time_ns=time_ns, bid_price_cents=bid_price, ask_price_cents=ask_price)


def compute_l1_series(book_log2: list[dict[str, Any]]) -> L1Snapshots:
    """Build :class:`L1Snapshots` from *book_log2*."""
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

    times: list[int] = []
    bid_prices: list[int | None] = []
    bid_quantities: list[int | None] = []
    ask_prices: list[int | None] = []
    ask_quantities: list[int | None] = []

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


def compute_l2_series(book_log2: list[dict[str, Any]]) -> L2Snapshots:
    """Build :class:`L2Snapshots` from *book_log2*.

    Reads the *already-sparse* ``bids`` and ``asks`` arrays from each
    snapshot. No zero-padding is applied.
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


def compute_trade_attribution(
    exec_entries: list[dict[str, Any]],
) -> list[TradeAttribution]:
    """Build :class:`TradeAttribution` records from raw execution dicts.

    Each dict must have keys: ``time``, ``agent_id``, ``oppos_agent_id``,
    ``side``, ``price``, ``quantity``.  Entries with ``price=None`` are
    skipped.
    """
    trades: list[TradeAttribution] = []
    for entry in exec_entries:
        price = entry.get("price")
        if price is None:
            continue
        trades.append(
            TradeAttribution(
                time_ns=int(entry["time"]),
                passive_agent_id=int(entry["agent_id"]),
                aggressive_agent_id=int(entry["oppos_agent_id"]),
                side=str(entry["side"]),
                price_cents=int(price),
                quantity=int(entry["quantity"]),
            )
        )
    return trades


# ---------------------------------------------------------------------------
# Agent-level metrics
# ---------------------------------------------------------------------------


def compute_agent_pnl(
    holdings: dict[str, int],
    starting_cash_cents: int,
    last_trade_prices: dict[str, int | None],
    *,
    basket_value_cents: int = 0,
    agent_id: int = 0,
    agent_type: str = "",
    agent_name: str = "",
    agent_category: str = "",
    execution_metrics: ExecutionMetrics | None = None,
    equity_curve: EquityCurve | None = None,
) -> AgentData:
    """Compute mark-to-market and PnL from plain holdings data.

    Parameters
    ----------
    holdings:
        Portfolio at simulation end: ``{"CASH": <cents>, "<TICKER>": <shares>, ...}``.
    starting_cash_cents:
        Initial cash position in cents.
    last_trade_prices:
        Mapping of symbol → last trade price (integer cents).  Symbols not
        present are valued at 0.
    basket_value_cents:
        Additional portfolio value in cents (e.g. ``basket_size * nav_diff``
        from ABIDES internal accounting).  Default 0 for external callers.
    agent_id, agent_type, agent_name, agent_category:
        Agent identifiers for the returned :class:`AgentData`.
    execution_metrics:
        Pre-computed :class:`ExecutionMetrics`, or ``None``.
    equity_curve:
        Pre-computed :class:`EquityCurve`, or ``None``.
    """
    cash = holdings.get("CASH", 0)
    mtm = cash + basket_value_cents
    for symbol, shares in holdings.items():
        if symbol == "CASH":
            continue
        price = last_trade_prices.get(symbol) or 0
        mtm += price * shares

    pnl = mtm - starting_cash_cents
    pnl_pct = (pnl / starting_cash_cents * 100.0) if starting_cash_cents != 0 else 0.0

    return AgentData(
        agent_id=agent_id,
        agent_type=agent_type,
        agent_name=agent_name or f"agent_{agent_id}",
        agent_category=agent_category,
        final_holdings=holdings,
        starting_cash_cents=starting_cash_cents,
        mark_to_market_cents=mtm,
        pnl_cents=pnl,
        pnl_pct=pnl_pct,
        execution_metrics=execution_metrics,
        equity_curve=equity_curve,
    )


def compute_execution_metrics(
    fills: Sequence[tuple[int, int]],
    target_quantity: int,
    filled_quantity: int,
    *,
    session_vwap_cents: int | None = None,
    total_volume: int = 0,
    arrival_price_cents: int | None = None,
) -> ExecutionMetrics:
    """Compute execution quality metrics from a fill list.

    Parameters
    ----------
    fills:
        Sequence of ``(fill_price_cents, quantity)`` tuples.
    target_quantity:
        Number of shares the agent intended to execute.
    filled_quantity:
        Number of shares actually filled.
    session_vwap_cents:
        Session VWAP in integer cents (for slippage computation).
    total_volume:
        Total session volume (for participation rate).
    arrival_price_cents:
        Market mid-price at first order, in integer cents.
    """
    fill_rate = filled_quantity / target_quantity * 100.0 if target_quantity > 0 else 0.0

    # Average fill price
    avg_fill: int | None = None
    if fills:
        total_value = 0
        total_qty = 0
        for price, qty in fills:
            if qty > 0:
                total_value += int(price) * int(qty)
                total_qty += int(qty)
        if total_qty > 0:
            avg_fill = total_value // total_qty

    # Derived metrics
    vwap_slippage: int | None = None
    if avg_fill is not None and session_vwap_cents is not None and session_vwap_cents > 0:
        vwap_slippage = (avg_fill - session_vwap_cents) * 10_000 // session_vwap_cents

    participation: float | None = None
    if filled_quantity > 0 and total_volume > 0:
        participation = filled_quantity / total_volume * 100.0

    impl_shortfall: int | None = None
    if avg_fill is not None and arrival_price_cents is not None and arrival_price_cents > 0:
        impl_shortfall = (avg_fill - arrival_price_cents) * 10_000 // arrival_price_cents

    return ExecutionMetrics(
        target_quantity=target_quantity,
        filled_quantity=filled_quantity,
        fill_rate_pct=fill_rate,
        avg_fill_price_cents=avg_fill,
        vwap_cents=session_vwap_cents,
        vwap_slippage_bps=vwap_slippage,
        participation_rate_pct=participation,
        arrival_price_cents=arrival_price_cents,
        implementation_shortfall_bps=impl_shortfall,
    )


def compute_equity_curve(
    fill_events: Sequence[tuple[int, int, int]],
) -> EquityCurve | None:
    """Build an :class:`EquityCurve` from fill event tuples.

    Parameters
    ----------
    fill_events:
        Sequence of ``(time_ns, nav_cents, peak_nav_cents)`` tuples.

    Returns ``None`` if *fill_events* is empty.
    """
    if not fill_events:
        return None
    times = [int(t) for t, _, _ in fill_events]
    navs = [int(n) for _, n, _ in fill_events]
    peaks = [int(p) for _, _, p in fill_events]
    return EquityCurve(times_ns=times, nav_cents=navs, peak_nav_cents=peaks)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def compute_metrics(
    *,
    book_log2: list[dict[str, Any]] | None = None,
    exec_trades: list[dict[str, Any]] | None = None,
    agent_holdings: list[dict[str, Any]] | None = None,
    last_trade_prices: dict[str, int | None] | None = None,
    pct_time_no_bid: float = 0.0,
    pct_time_no_ask: float = 0.0,
    total_exchanged_volume: int = 0,
    last_trade_cents: int | None = None,
    symbol: str = "UNKNOWN",
) -> dict[str, Any]:
    """Compute the canonical metric set from raw data sources.

    This is the single entry-point for external consumers who want the same
    metrics that :class:`SimulationResult` provides, without running a
    full simulation.

    Parameters
    ----------
    book_log2:
        Order-book snapshot list (same format as ``OrderBook.book_log2``).
        Each entry must have keys ``QuoteTime``, ``bids``, ``asks``.
    exec_trades:
        List of execution dicts with keys ``time``, ``agent_id``,
        ``oppos_agent_id``, ``side``, ``price``, ``quantity``.
    agent_holdings:
        List of agent dicts, each with keys ``holdings``, ``starting_cash_cents``,
        and optionally ``agent_id``, ``agent_type``, ``agent_name``,
        ``basket_value_cents``.
    last_trade_prices:
        Mapping of symbol → last trade price (integer cents) for mark-to-market.
    pct_time_no_bid, pct_time_no_ask:
        Percentage of session time with an empty bid/ask side (0–100).
    total_exchanged_volume:
        Total shares traded across the session.
    last_trade_cents:
        Price of the last executed trade (integer cents).
    symbol:
        Ticker symbol for the :class:`MarketSummary` key.

    Returns
    -------
    dict
        Keys:

        - ``"market"`` → :class:`MarketSummary`
        - ``"agents"`` → ``list[AgentData]``
        - ``"vwap_cents"`` → ``int | None``
        - ``"trades"`` → ``list[TradeAttribution]``
    """
    book_log2 = book_log2 or []
    exec_trades = exec_trades or []
    agent_holdings = agent_holdings or []
    last_trade_prices = last_trade_prices or {}

    # -- Trade attribution & VWAP trades ------------------------------------
    exec_entries = [e for e in exec_trades if e.get("type") == "EXEC" or "time" in e]
    trade_attributions = compute_trade_attribution(
        [e for e in exec_entries if e.get("price") is not None]
    )
    vwap_trades: list[tuple[int, int]] = [
        (int(e["price"]), int(e["quantity"])) for e in exec_entries if e.get("price") is not None
    ]
    vwap_cents = compute_vwap(vwap_trades)

    # -- Market metrics -----------------------------------------------------
    l1_close = compute_l1_close(book_log2)
    liquidity = compute_liquidity_metrics(
        vwap_trades,
        pct_time_no_bid=pct_time_no_bid,
        pct_time_no_ask=pct_time_no_ask,
        total_exchanged_volume=total_exchanged_volume,
        last_trade_cents=last_trade_cents,
    )
    market_summary = MarketSummary(
        symbol=symbol,
        l1_close=l1_close,
        liquidity=liquidity,
    )

    # -- Agent PnL ----------------------------------------------------------
    agents: list[AgentData] = []
    for agent_info in agent_holdings:
        agents.append(
            compute_agent_pnl(
                holdings=agent_info["holdings"],
                starting_cash_cents=agent_info["starting_cash_cents"],
                last_trade_prices=last_trade_prices,
                basket_value_cents=agent_info.get("basket_value_cents", 0),
                agent_id=agent_info.get("agent_id", 0),
                agent_type=agent_info.get("agent_type", ""),
                agent_name=agent_info.get("agent_name", ""),
            )
        )

    return {
        "market": market_summary,
        "agents": agents,
        "vwap_cents": vwap_cents,
        "trades": trade_attributions,
    }
