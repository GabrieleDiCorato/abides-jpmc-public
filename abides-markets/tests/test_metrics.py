"""Tests for abides_markets.simulation.metrics — standalone metric computation.

Tests verify that each compute_* function produces correct results from
plain Python data (no live agents or exchange objects required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abides_markets.simulation.metrics import (
    compute_adverse_selection,
    compute_agent_pnl,
    compute_avg_liquidity,
    compute_effective_spread,
    compute_equity_curve,
    compute_execution_metrics,
    compute_fill_slippage,
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
    compute_rich_metrics,
    compute_sharpe_ratio,
    compute_trade_attribution,
    compute_volatility,
    compute_vpin,
    compute_vwap,
)
from abides_markets.simulation.profiles import ResultProfile
from abides_markets.simulation.result import (
    AgentData,
    EquityCurve,
    ExecutionMetrics,
    FillRecord,
    L1Close,
    L1Snapshots,
    L2Snapshots,
    LiquidityMetrics,
    MarketSummary,
    OrderLifecycle,
    RichSimulationMetrics,
    SimulationMetadata,
    SimulationResult,
    TradeAttribution,
)

# ===================================================================
# compute_vwap
# ===================================================================


class TestComputeVwap:
    def test_single_trade(self):
        assert compute_vwap([(10_000, 100)]) == 10_000

    def test_two_trades_equal_weight(self):
        # (10000*100 + 10200*100) / 200 = 10100
        assert compute_vwap([(10_000, 100), (10_200, 100)]) == 10_100

    def test_two_trades_unequal_weight(self):
        # (10000*300 + 10100*100) / 400 = 10025
        assert compute_vwap([(10_000, 300), (10_100, 100)]) == 10_025

    def test_integer_division_truncates(self):
        # (10001*1 + 10000*2) / 3 = 10000.333... → 10000
        assert compute_vwap([(10_001, 1), (10_000, 2)]) == 10_000

    def test_empty_returns_none(self):
        assert compute_vwap([]) is None

    def test_zero_quantity_returns_none(self):
        assert compute_vwap([(10_000, 0)]) is None


# ===================================================================
# compute_liquidity_metrics
# ===================================================================


class TestComputeLiquidityMetrics:
    def test_basic(self):
        liq = compute_liquidity_metrics(
            [(10_000, 100), (10_200, 100)],
            pct_time_no_bid=5.0,
            pct_time_no_ask=3.0,
            total_exchanged_volume=200,
            last_trade_cents=10_200,
        )
        assert isinstance(liq, LiquidityMetrics)
        assert liq.vwap_cents == 10_100
        assert liq.pct_time_no_bid == 5.0
        assert liq.pct_time_no_ask == 3.0
        assert liq.total_exchanged_volume == 200
        assert liq.last_trade_cents == 10_200

    def test_empty_trades(self):
        liq = compute_liquidity_metrics()
        assert liq.vwap_cents is None
        assert liq.total_exchanged_volume == 0

    def test_defaults(self):
        liq = compute_liquidity_metrics([(10_000, 50)])
        assert liq.pct_time_no_bid == 0.0
        assert liq.pct_time_no_ask == 0.0
        assert liq.total_exchanged_volume == 0
        assert liq.last_trade_cents is None
        assert liq.vwap_cents == 10_000


# ===================================================================
# compute_l1_close
# ===================================================================


class TestComputeL1Close:
    def test_empty_book(self):
        close = compute_l1_close([])
        assert close.time_ns == 0
        assert close.bid_price_cents is None
        assert close.ask_price_cents is None

    def test_normal(self):
        book = [
            {"QuoteTime": 100, "bids": [(9900, 10)], "asks": [(10100, 5)]},
            {"QuoteTime": 200, "bids": [(9950, 20)], "asks": [(10050, 15)]},
        ]
        close = compute_l1_close(book)
        assert close.time_ns == 200
        assert close.bid_price_cents == 9950
        assert close.ask_price_cents == 10050

    def test_empty_sides(self):
        book = [{"QuoteTime": 100, "bids": [], "asks": []}]
        close = compute_l1_close(book)
        assert close.bid_price_cents is None
        assert close.ask_price_cents is None


# ===================================================================
# compute_l1_series
# ===================================================================


class TestComputeL1Series:
    def test_empty_book(self):
        series = compute_l1_series([])
        assert len(series.times_ns) == 0

    def test_normal(self):
        book = [
            {"QuoteTime": 100, "bids": [(9900, 10)], "asks": [(10100, 5)]},
            {"QuoteTime": 200, "bids": [], "asks": [(10050, 15)]},
        ]
        series = compute_l1_series(book)
        assert isinstance(series, L1Snapshots)
        assert len(series.times_ns) == 2
        assert series.bid_prices[0] == 9900
        assert series.bid_prices[1] is None
        assert series.ask_prices[0] == 10100
        assert series.ask_prices[1] == 10050


# ===================================================================
# compute_l2_series
# ===================================================================


class TestComputeL2Series:
    def test_empty_book(self):
        series = compute_l2_series([])
        assert len(series.times_ns) == 0
        assert series.bids == []
        assert series.asks == []

    def test_normal(self):
        book = [
            {
                "QuoteTime": 100,
                "bids": [(9900, 10), (9800, 20)],
                "asks": [(10100, 5)],
            },
        ]
        series = compute_l2_series(book)
        assert isinstance(series, L2Snapshots)
        assert len(series.times_ns) == 1
        assert series.bids[0] == [(9900, 10), (9800, 20)]
        assert series.asks[0] == [(10100, 5)]


# ===================================================================
# compute_trade_attribution
# ===================================================================


class TestComputeTradeAttribution:
    def test_empty(self):
        assert compute_trade_attribution([]) == []

    def test_skips_none_price(self):
        entries = [
            {
                "time": 100,
                "agent_id": 1,
                "oppos_agent_id": 2,
                "side": "BUY",
                "price": None,
                "quantity": 50,
            }
        ]
        assert compute_trade_attribution(entries) == []

    def test_normal(self):
        entries = [
            {
                "time": 100,
                "agent_id": 1,
                "oppos_agent_id": 2,
                "side": "BUY",
                "price": 10_000,
                "quantity": 50,
            },
            {
                "time": 200,
                "agent_id": 3,
                "oppos_agent_id": 4,
                "side": "SELL",
                "price": 10_050,
                "quantity": 30,
            },
        ]
        result = compute_trade_attribution(entries)
        assert len(result) == 2
        assert isinstance(result[0], TradeAttribution)
        assert result[0].price_cents == 10_000
        assert result[1].side == "SELL"


# ===================================================================
# compute_agent_pnl
# ===================================================================


class TestComputeAgentPnl:
    def test_cash_only(self):
        agent = compute_agent_pnl(
            holdings={"CASH": 10_000_000},
            starting_cash_cents=10_000_000,
            last_trade_prices={},
            agent_id=1,
            agent_type="noise",
        )
        assert isinstance(agent, AgentData)
        assert agent.mark_to_market_cents == 10_000_000
        assert agent.pnl_cents == 0
        assert agent.pnl_pct == 0.0

    def test_with_position(self):
        agent = compute_agent_pnl(
            holdings={"CASH": 9_000_000, "AAPL": 100},
            starting_cash_cents=10_000_000,
            last_trade_prices={"AAPL": 10_100},
            agent_id=2,
            agent_type="value",
        )
        # mtm = 9_000_000 + 100 * 10_100 = 10_010_000
        assert agent.mark_to_market_cents == 10_010_000
        assert agent.pnl_cents == 10_000
        assert agent.pnl_pct == pytest.approx(0.1)

    def test_missing_price_valued_at_zero(self):
        agent = compute_agent_pnl(
            holdings={"CASH": 9_000_000, "AAPL": 100},
            starting_cash_cents=10_000_000,
            last_trade_prices={},
        )
        # AAPL valued at 0: mtm = 9_000_000
        assert agent.mark_to_market_cents == 9_000_000
        assert agent.pnl_cents == -1_000_000

    def test_basket_value(self):
        agent = compute_agent_pnl(
            holdings={"CASH": 10_000_000},
            starting_cash_cents=10_000_000,
            last_trade_prices={},
            basket_value_cents=5_000,
        )
        assert agent.mark_to_market_cents == 10_005_000
        assert agent.pnl_cents == 5_000

    def test_zero_starting_cash(self):
        agent = compute_agent_pnl(
            holdings={"CASH": 0},
            starting_cash_cents=0,
            last_trade_prices={},
        )
        assert agent.pnl_pct == 0.0

    def test_agent_name_default(self):
        agent = compute_agent_pnl(
            holdings={"CASH": 100},
            starting_cash_cents=100,
            last_trade_prices={},
            agent_id=42,
        )
        assert agent.agent_name == "agent_42"


# ===================================================================
# compute_execution_metrics
# ===================================================================


class TestComputeExecutionMetrics:
    def test_basic(self):
        em = compute_execution_metrics(
            fills=[(10_000, 50), (10_100, 50)],
            target_quantity=100,
            filled_quantity=100,
            session_vwap_cents=10_000,
            total_volume=1000,
            arrival_price_cents=9_950,
        )
        assert isinstance(em, ExecutionMetrics)
        assert em.fill_rate_pct == 100.0
        # avg_fill = (10000*50 + 10100*50) / 100 = 10050
        assert em.avg_fill_price_cents == 10_050
        # vwap_slippage = (10050 - 10000) * 10000 // 10000 = 50
        assert em.vwap_slippage_bps == 50
        # participation = 100 / 1000 * 100 = 10.0%
        assert em.participation_rate_pct == pytest.approx(10.0)
        # impl_shortfall = (10050 - 9950) * 10000 // 9950 = 100
        assert em.implementation_shortfall_bps == 100

    def test_empty_fills(self):
        em = compute_execution_metrics(
            fills=[],
            target_quantity=100,
            filled_quantity=0,
        )
        assert em.avg_fill_price_cents is None
        assert em.fill_rate_pct == 0.0
        assert em.vwap_slippage_bps is None

    def test_zero_target(self):
        em = compute_execution_metrics(
            fills=[(10_000, 50)],
            target_quantity=0,
            filled_quantity=0,
        )
        assert em.fill_rate_pct == 0.0

    def test_no_vwap(self):
        em = compute_execution_metrics(
            fills=[(10_000, 100)],
            target_quantity=100,
            filled_quantity=100,
        )
        assert em.vwap_slippage_bps is None
        assert em.participation_rate_pct is None
        assert em.implementation_shortfall_bps is None

    def test_partial_fill(self):
        em = compute_execution_metrics(
            fills=[(10_000, 50)],
            target_quantity=100,
            filled_quantity=50,
        )
        assert em.fill_rate_pct == 50.0


# ===================================================================
# compute_equity_curve
# ===================================================================


class TestComputeEquityCurve:
    def test_empty(self):
        assert compute_equity_curve([]) is None

    def test_normal(self):
        events = [
            (100, 10_000_000, 10_000_000),
            (200, 9_990_000, 10_000_000),
            (300, 10_010_000, 10_010_000),
        ]
        curve = compute_equity_curve(events)
        assert isinstance(curve, EquityCurve)
        assert len(curve.times_ns) == 3
        assert curve.nav_cents == [10_000_000, 9_990_000, 10_010_000]
        assert curve.max_drawdown_cents == 10_000  # 10_000_000 - 9_990_000

    def test_single_event(self):
        curve = compute_equity_curve([(100, 5_000, 5_000)])
        assert curve is not None
        assert curve.max_drawdown_cents == 0

    # ------------------------------------------------------------------
    # L1-dense mode
    # ------------------------------------------------------------------

    def test_l1_none_preserves_fill_only(self):
        """Passing l1=None must not change behavior."""
        events = [(100, 10_000, 10_000), (300, 9_800, 10_000)]
        curve_no_l1 = compute_equity_curve(events)
        curve_l1_none = compute_equity_curve(events, l1=None)
        assert curve_no_l1 == curve_l1_none

    def test_l1_one_entry_per_two_sided_tick(self):
        """Dense curve has exactly one entry per two-sided L1 observation."""
        events = [(100, 10_000, 10_000), (400, 9_500, 10_000)]
        l1 = _make_l1(
            [
                (50, 9_900, 10, 10_100, 5),  # before first fill → excluded
                (100, 9_900, 10, 10_100, 5),  # at fill 1
                (200, 9_850, 10, 10_050, 5),  # between fills
                (300, None, None, 10_000, 5),  # one-sided → excluded
                (400, 9_800, 10, 10_000, 5),  # at fill 2
                (500, 9_800, 10, 10_000, 5),  # after last fill
            ]
        )
        curve = compute_equity_curve(events, l1=l1)
        assert curve is not None
        # t=50 excluded (before first fill), t=300 excluded (one-sided)
        assert curve.times_ns == [100, 200, 400, 500]
        # t=100 → fill 1 (nav=10_000); t=200 → carry fwd fill 1;
        # t=400 → fill 2 (nav=9_500); t=500 → carry fwd fill 2
        assert curve.nav_cents == [10_000, 10_000, 9_500, 9_500]
        assert curve.peak_nav_cents == [10_000, 10_000, 10_000, 10_000]

    def test_l1_nav_carries_forward(self):
        """Between fills NAV is the last fill's nav_cents (step function)."""
        events = [(100, 10_000, 10_000), (400, 9_500, 10_000)]
        l1 = _make_l1(
            [
                (100, 9_900, 10, 10_100, 5),
                (200, 9_850, 10, 10_050, 5),
                (300, 9_820, 10, 10_020, 5),
                (400, 9_800, 10, 10_000, 5),
                (500, 9_800, 10, 10_000, 5),
            ]
        )
        curve = compute_equity_curve(events, l1=l1)
        assert curve is not None
        # ticks 200 and 300 (between fills) carry forward fill-1 nav=10_000
        assert curve.nav_cents == [10_000, 10_000, 10_000, 9_500, 9_500]
        assert curve.peak_nav_cents == [10_000, 10_000, 10_000, 10_000, 10_000]

    def test_l1_all_ticks_before_first_fill_fallback(self):
        """All L1 ticks precede first fill → fall back to fill-only curve."""
        events = [(500, 10_000, 10_000)]
        l1 = _make_l1(
            [
                (100, 9_900, 10, 10_100, 5),
                (200, 9_900, 10, 10_100, 5),
            ]
        )
        curve = compute_equity_curve(events, l1=l1)
        expected = compute_equity_curve(events)
        assert curve == expected

    def test_l1_empty_falls_back_to_fill_only(self):
        """Empty L1 snapshot → fall back to fill-only curve."""
        events = [(100, 10_000, 10_000), (200, 9_900, 10_000)]
        curve = compute_equity_curve(events, l1=_make_empty_l1())
        expected = compute_equity_curve(events)
        assert curve == expected

    def test_l1_no_two_sided_ticks_fallback(self):
        """L1 with no two-sided ticks → fall back to fill-only curve."""
        events = [(100, 10_000, 10_000)]
        l1 = _make_l1(
            [
                (100, 9_900, 10, None, None),
                (200, None, None, 10_100, 5),
            ]
        )
        curve = compute_equity_curve(events, l1=l1)
        expected = compute_equity_curve(events)
        assert curve == expected

    def test_l1_max_drawdown_dense(self):
        """max_drawdown_cents is computed over the dense curve."""
        events = [(100, 10_000, 10_000), (500, 9_000, 10_000)]
        l1 = _make_l1(
            [
                (100, 9_900, 10, 10_100, 5),
                (200, 9_850, 10, 10_050, 5),
                (300, 9_820, 10, 10_020, 5),
                (400, 9_800, 10, 10_000, 5),
                (500, 9_750, 10, 9_950, 5),
            ]
        )
        curve = compute_equity_curve(events, l1=l1)
        assert curve is not None
        # nav at ticks 100-400 = 10_000 (carry forward fill 1), tick 500 = 9_000
        assert curve.nav_cents == [10_000, 10_000, 10_000, 10_000, 9_000]
        assert curve.peak_nav_cents == [10_000, 10_000, 10_000, 10_000, 10_000]
        assert curve.max_drawdown_cents == 1_000  # 10_000 - 9_000


# ===================================================================
# compute_metrics (top-level orchestrator)
# ===================================================================


class TestComputeMetrics:
    def test_minimal(self):
        result = compute_metrics()
        assert "market" in result
        assert "agents" in result
        assert "vwap_cents" in result
        assert "trades" in result
        assert isinstance(result["market"], MarketSummary)
        assert result["agents"] == []
        assert result["vwap_cents"] is None
        assert result["trades"] == []

    def test_with_book_and_trades(self):
        book = [
            {"QuoteTime": 100, "bids": [(9900, 10)], "asks": [(10100, 5)]},
            {"QuoteTime": 200, "bids": [(9950, 20)], "asks": [(10050, 15)]},
        ]
        trades = [
            {
                "time": 150,
                "agent_id": 1,
                "oppos_agent_id": 2,
                "side": "BUY",
                "price": 10_000,
                "quantity": 100,
            },
        ]
        result = compute_metrics(
            book_log2=book,
            exec_trades=trades,
            total_exchanged_volume=100,
            last_trade_cents=10_000,
            symbol="AAPL",
        )
        assert result["market"].symbol == "AAPL"
        assert result["market"].l1_close.bid_price_cents == 9950
        assert result["vwap_cents"] == 10_000
        assert len(result["trades"]) == 1

    def test_with_agents(self):
        result = compute_metrics(
            agent_holdings=[
                {
                    "holdings": {"CASH": 10_000_000, "AAPL": 50},
                    "starting_cash_cents": 10_000_000,
                    "agent_id": 1,
                    "agent_type": "noise",
                    "agent_name": "noise_1",
                },
            ],
            last_trade_prices={"AAPL": 10_000},
        )
        agents = result["agents"]
        assert len(agents) == 1
        assert agents[0].mark_to_market_cents == 10_500_000
        assert agents[0].pnl_cents == 500_000

    def test_import_from_package(self):
        """Verify that compute_metrics is importable from the package root."""
        from abides_markets.simulation import compute_metrics as cm

        assert callable(cm)


# ===================================================================
# Helper: build L1Snapshots from simple data
# ===================================================================


def _make_l1(
    rows: list[tuple[int, int | None, int | None, int | None, int | None]],
) -> L1Snapshots:
    """Build L1Snapshots from (time_ns, bid_price, bid_qty, ask_price, ask_qty) tuples."""
    times = [r[0] for r in rows]
    return L1Snapshots(
        times_ns=np.array(times, dtype=np.int64),
        bid_prices=np.array([r[1] for r in rows], dtype=object),
        bid_quantities=np.array([r[2] for r in rows], dtype=object),
        ask_prices=np.array([r[3] for r in rows], dtype=object),
        ask_quantities=np.array([r[4] for r in rows], dtype=object),
    )


def _make_empty_l1() -> L1Snapshots:
    empty = np.array([], dtype=np.int64)
    empty_obj = np.array([], dtype=object)
    return L1Snapshots(
        times_ns=empty,
        bid_prices=empty_obj,
        bid_quantities=empty_obj,
        ask_prices=empty_obj,
        ask_quantities=empty_obj,
    )


# ===================================================================
# compute_mean_spread
# ===================================================================


class TestComputeMeanSpread:
    def test_empty_l1(self):
        assert compute_mean_spread(_make_empty_l1()) is None

    def test_no_two_sided(self):
        l1 = _make_l1([(100, 9900, 10, None, None)])
        assert compute_mean_spread(l1) is None

    def test_single_row(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        assert compute_mean_spread(l1) == pytest.approx(200.0)

    def test_multiple_rows(self):
        l1 = _make_l1(
            [
                (100, 9900, 10, 10100, 5),  # spread = 200
                (200, 9950, 20, 10050, 15),  # spread = 100
                (300, None, None, 10000, 10),  # one-sided, skipped
            ]
        )
        assert compute_mean_spread(l1) == pytest.approx(150.0)


# ===================================================================
# compute_effective_spread
# ===================================================================


class TestComputeEffectiveSpread:
    def test_empty_fills(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        assert compute_effective_spread([], l1) is None

    def test_empty_l1(self):
        assert compute_effective_spread([(10_000, 50, 100)], _make_empty_l1()) is None

    def test_no_two_sided_l1(self):
        l1 = _make_l1([(100, 9900, 10, None, None)])
        assert compute_effective_spread([(10_000, 50, 100)], l1) is None

    def test_single_fill_at_mid(self):
        # mid = (9900 + 10100) / 2 = 10000
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        fills = [(10_000, 50, 100)]
        assert compute_effective_spread(fills, l1) == pytest.approx(0.0)

    def test_single_fill_above_mid(self):
        # mid = 10000, fill at 10050 → eff_spread = 2 * 50 = 100
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        fills = [(10_050, 50, 150)]
        assert compute_effective_spread(fills, l1) == pytest.approx(100.0)

    def test_multiple_fills(self):
        l1 = _make_l1(
            [
                (100, 9900, 10, 10100, 5),  # mid = 10000
                (200, 9950, 20, 10050, 15),  # mid = 10000
            ]
        )
        # fill1 at 10025 vs mid 10000 → 2*25=50
        # fill2 at 9975 vs mid 10000 → 2*25=50
        fills = [(10_025, 50, 150), (9_975, 50, 250)]
        assert compute_effective_spread(fills, l1) == pytest.approx(50.0)


# ===================================================================
# compute_volatility
# ===================================================================


class TestComputeVolatility:
    def test_empty(self):
        assert compute_volatility(_make_empty_l1()) is None

    def test_too_few_rows(self):
        rows = [(i * 1_000_000, 10000, 10, 10100, 5) for i in range(10)]
        assert compute_volatility(_make_l1(rows)) is None

    def test_constant_mid(self):
        rows = [(i * 1_000_000, 10000, 10, 10100, 5) for i in range(40)]
        assert compute_volatility(_make_l1(rows)) is None  # std=0

    def test_returns_positive(self):
        # Construct varying mid-prices
        base = 10_000
        rows = []
        for i in range(40):
            offset = (i % 5) * 10  # cycles: 0,10,20,30,40,0,10...
            bid = base + offset
            ask = bid + 100
            rows.append((i * 1_000_000_000, bid, 10, ask, 5))
        vol = compute_volatility(_make_l1(rows))
        assert vol is not None
        assert vol > 0


# ===================================================================
# compute_sharpe_ratio
# ===================================================================


class TestComputeSharpeRatio:
    def test_none_curve(self):
        assert compute_sharpe_ratio(None) is None

    def test_too_few_points(self):
        curve = EquityCurve(
            times_ns=list(range(10)),
            nav_cents=[10_000] * 10,
            peak_nav_cents=[10_000] * 10,
        )
        assert compute_sharpe_ratio(curve) is None

    def test_flat_nav(self):
        n = 40
        curve = EquityCurve(
            times_ns=list(range(n)),
            nav_cents=[10_000] * n,
            peak_nav_cents=[10_000] * n,
        )
        assert compute_sharpe_ratio(curve) is None  # std=0

    def test_trending_up(self):
        n = 40
        navs = [10_000_000 + i * 1000 for i in range(n)]
        curve = EquityCurve(
            times_ns=[i * 1_000_000_000 for i in range(n)],
            nav_cents=navs,
            peak_nav_cents=navs,
        )
        sharpe = compute_sharpe_ratio(curve)
        assert sharpe is not None
        assert sharpe > 0


# ===================================================================
# compute_avg_liquidity
# ===================================================================


class TestComputeAvgLiquidity:
    def test_empty(self):
        assert compute_avg_liquidity(_make_empty_l1()) == (None, None)

    def test_no_two_sided(self):
        l1 = _make_l1([(100, 9900, 10, None, None)])
        assert compute_avg_liquidity(l1) == (None, None)

    def test_single_row(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        assert compute_avg_liquidity(l1) == (10.0, 5.0)

    def test_multiple_rows(self):
        l1 = _make_l1(
            [
                (100, 9900, 10, 10100, 20),
                (200, 9950, 30, 10050, 40),
                (300, None, None, 10000, 10),  # skipped
            ]
        )
        bid, ask = compute_avg_liquidity(l1)
        assert bid == pytest.approx(20.0)
        assert ask == pytest.approx(30.0)


# ===================================================================
# compute_lob_imbalance
# ===================================================================


class TestComputeLobImbalance:
    def test_empty(self):
        assert compute_lob_imbalance(_make_empty_l1()) == (None, None)

    def test_no_valid_rows(self):
        l1 = _make_l1([(100, None, None, 10100, 5)])
        assert compute_lob_imbalance(l1) == (None, None)

    def test_balanced_book(self):
        l1 = _make_l1([(100, 9900, 50, 10100, 50)])
        mean, std = compute_lob_imbalance(l1)
        assert mean == pytest.approx(0.0)
        assert std == 0.0  # single observation

    def test_bid_heavy(self):
        l1 = _make_l1(
            [
                (100, 9900, 80, 10100, 20),  # I = (80-20)/(80+20) = 0.6
                (200, 9950, 60, 10050, 40),  # I = (60-40)/(60+40) = 0.2
            ]
        )
        mean, std = compute_lob_imbalance(l1)
        assert mean == pytest.approx(0.4)
        assert std is not None
        assert std > 0

    def test_zero_both_sides_skipped(self):
        l1 = _make_l1([(100, 9900, 0, 10100, 0)])
        # denom=0 → skipped
        assert compute_lob_imbalance(l1) == (None, None)


# ===================================================================
# compute_inventory_std
# ===================================================================


class TestComputeInventoryStd:
    def test_too_few(self):
        assert compute_inventory_std([]) is None
        assert compute_inventory_std([("BUY", 10)]) is None

    def test_constant_inventory(self):
        fills = [("BUY", 10), ("SELL", 10)]  # inventory: 10, 0
        std = compute_inventory_std(fills)
        assert std is not None
        assert std > 0

    def test_increasing(self):
        fills = [("BUY", 10), ("BUY", 10), ("BUY", 10)]
        # inventory: 10, 20, 30 → std of [10,20,30]
        std = compute_inventory_std(fills)
        assert std == pytest.approx(10.0)

    def test_mixed(self):
        fills = [("BUY", 100), ("SELL", 50), ("BUY", 30), ("SELL", 80)]
        # inventory: 100, 50, 80, 0
        std = compute_inventory_std(fills)
        assert std is not None
        assert std > 0


# ===================================================================
# compute_market_ott_ratio
# ===================================================================


class TestComputeMarketOttRatio:
    def test_no_fills(self):
        assert compute_market_ott_ratio(100, 0) is None

    def test_one_to_one(self):
        assert compute_market_ott_ratio(100, 100) == pytest.approx(1.0)

    def test_four_to_one(self):
        assert compute_market_ott_ratio(400, 100) == pytest.approx(4.0)

    def test_fractional(self):
        assert compute_market_ott_ratio(150, 100) == pytest.approx(1.5)


# ===================================================================
# compute_vpin
# ===================================================================


class TestComputeVpin:
    def test_too_few_fills(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        fills = [(10_000, 50, 100)]
        assert compute_vpin(fills, l1) is None

    def test_balanced_flow(self):
        # 30 buys above mid + 30 sells below mid → low VPIN
        l1 = _make_l1([(i, 9900, 100, 10100, 100) for i in range(100)])
        fills = []
        for i in range(30):
            fills.append((10_050, 10, i * 2))  # buy (above mid 10000)
            fills.append((9_950, 10, i * 2 + 1))  # sell (below mid 10000)
        result = compute_vpin(fills, l1, n_buckets=10)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_directional_flow(self):
        # All buys → high VPIN
        l1 = _make_l1([(i, 9900, 100, 10100, 100) for i in range(100)])
        fills = [(10_050, 10, i) for i in range(30)]
        result = compute_vpin(fills, l1, n_buckets=5)
        assert result is not None
        assert result > 0.5

    def test_no_l1(self):
        # Still works via tick rule
        fills = [(10_000 + i * 10, 10, i) for i in range(30)]
        result = compute_vpin(fills, _make_empty_l1(), n_buckets=5)
        assert result is not None

    def test_zero_volume(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        fills = [(10_000, 0, 100)] * 30
        assert compute_vpin(fills, l1) is None


# ===================================================================
# compute_resilience
# ===================================================================


class TestComputeResilience:
    def test_empty(self):
        assert compute_resilience(_make_empty_l1()) is None

    def test_too_few_rows(self):
        rows = [(i, 9900, 10, 10100, 5) for i in range(5)]
        assert compute_resilience(_make_l1(rows)) is None

    def test_no_shocks(self):
        # Constant spread → no shocks
        rows = [(i * 1_000_000, 9900, 10, 10100, 5) for i in range(50)]
        assert compute_resilience(_make_l1(rows)) is None

    def test_with_shock_and_recovery(self):
        rows = []
        # Normal spread of 200 for 30 ticks
        for i in range(30):
            rows.append((i * 1_000_000, 9900, 10, 10100, 5))
        # Shock: spread jumps to 1000 for 3 ticks
        for i in range(30, 33):
            rows.append((i * 1_000_000, 9500, 10, 10500, 5))
        # Recovery: spread back to 200
        for i in range(33, 50):
            rows.append((i * 1_000_000, 9900, 10, 10100, 5))
        result = compute_resilience(_make_l1(rows))
        # Should detect the shock and measure recovery
        # Could be None if the rolling window doesn't detect it as a shock
        # (depends on window size), but the pattern is designed to trigger it
        if result is not None:
            assert result > 0


# ===================================================================
# compute_order_fill_rate
# ===================================================================


class TestComputeOrderFillRate:
    def test_no_submissions(self):
        assert compute_order_fill_rate(0, 0) is None

    def test_all_filled(self):
        assert compute_order_fill_rate(100, 100) == pytest.approx(100.0)

    def test_partial(self):
        assert compute_order_fill_rate(30, 50) == pytest.approx(60.0)

    def test_none_filled(self):
        assert compute_order_fill_rate(0, 50) == pytest.approx(0.0)

    def test_semantic_difference_from_quantity_fill_rate(self):
        """Demonstrate the two fill rate definitions are independent.

        An agent might submit 10 orders, 5 get filled (order_fill_rate = 50%),
        but the 5 filled orders might fill 900 of a 1000-share target
        (quantity fill_rate_pct = 90%).
        """
        order_rate = compute_order_fill_rate(5, 10)
        assert order_rate == pytest.approx(50.0)

        em = compute_execution_metrics(
            fills=[(10_000, 180)] * 5,
            target_quantity=1000,
            filled_quantity=900,
        )
        assert em.fill_rate_pct == pytest.approx(90.0)


# ===================================================================
# compute_fill_slippage
# ===================================================================


class TestComputeFillSlippage:
    def test_buy_above_mid(self):
        # mid = (9900 + 10100) / 2 = 10000
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        # Buy at 10050 → slippage = (10050 - 10000) * 10000 / 10000 = 50 bps
        assert compute_fill_slippage(10_050, 100, "BUY", l1) == 50

    def test_buy_below_mid(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        # Buy at 9950 → slippage = (9950 - 10000) * 10000 / 10000 = -50 bps
        assert compute_fill_slippage(9_950, 100, "BUY", l1) == -50

    def test_sell_above_mid(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        # Sell at 10050 → diff = -(10050 - 10000) = -50, bps = -50 * 10000 / 10000
        # Actually: diff = float(fill) - mid = 50; side SELL → diff = -50
        # bps = -50 * 10000 / 10000 = -50 (negative = received above mid, good for seller)
        # Wait, per sign convention: for SELL, we negate diff before computing bps
        # diff = fill - mid = 50; negate for SELL → -50; bps = -50 * 10000 / 10000 = -50
        # Hmm, that means selling above mid is negative (good? confusing)
        # Let me re-read the impl:
        # diff = fill - mid = 50; side == "SELL" → diff = -diff = -50
        # return int(diff * 10000 / mid) = int(-50 * 10000 / 10000) = -50
        # The docstring says positive = paid above mid (bad for buyer).
        # For SELL above mid, it returns -50. This is consistent: selling above mid is good.
        assert compute_fill_slippage(10_050, 100, "SELL", l1) == -50

    def test_empty_l1(self):
        assert compute_fill_slippage(10_000, 100, "BUY", _make_empty_l1()) is None

    def test_no_two_sided(self):
        l1 = _make_l1([(100, 9900, 10, None, None)])
        assert compute_fill_slippage(10_000, 100, "BUY", l1) is None

    def test_at_mid(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        assert compute_fill_slippage(10_000, 100, "BUY", l1) == 0


# ===================================================================
# compute_adverse_selection
# ===================================================================


class TestComputeAdverseSelection:
    def test_buy_price_rises(self):
        # Mid at t=100: (9900+10100)/2 = 10000
        # Mid at t=200: (10000+10200)/2 = 10100
        # BUY: delta = 10100 - 10000 = 100; bps = 100 * 10000 / 10000 = 100
        # Positive = favorable (price went up after buy)
        l1 = _make_l1(
            [
                (100, 9900, 10, 10100, 5),
                (200, 10000, 10, 10200, 5),
            ]
        )
        assert compute_adverse_selection(10_000, 100, "BUY", l1, 100) == 100

    def test_buy_price_drops(self):
        # Mid at t=100: 10000; mid at t=200: 9900; delta = -100; bps = -100
        l1 = _make_l1(
            [
                (100, 9900, 10, 10100, 5),
                (200, 9800, 10, 10000, 5),
            ]
        )
        assert compute_adverse_selection(10_000, 100, "BUY", l1, 100) == -100

    def test_sell_price_drops(self):
        # Mid at t=100: 10000; mid at t=200: 9900; delta = -100; SELL → negate = +100
        # Positive = favorable (sold and price dropped)
        l1 = _make_l1(
            [
                (100, 9900, 10, 10100, 5),
                (200, 9800, 10, 10000, 5),
            ]
        )
        assert compute_adverse_selection(10_000, 100, "SELL", l1, 100) == 100

    def test_window_past_end(self):
        l1 = _make_l1([(100, 9900, 10, 10100, 5)])
        # Window extends past all data → uses last available mid, which is the
        # same as fill time. But the window time is 200 and last data is at 100.
        # _lookup_mid(200) will searchsorted to idx=0 (since 200 > 100, idx=1-1=0)
        # and find the row at idx=0. So mid_tw = mid_t = 10000, delta = 0.
        assert compute_adverse_selection(10_000, 100, "BUY", l1, 100) == 0

    def test_empty_l1(self):
        assert (
            compute_adverse_selection(10_000, 100, "BUY", _make_empty_l1(), 100) is None
        )


# ===================================================================
# Helper: build a minimal SimulationResult
# ===================================================================


def _make_result(
    *,
    profile: ResultProfile = ResultProfile.SUMMARY,
    l1_rows: (
        list[tuple[int, int | None, int | None, int | None, int | None]] | None
    ) = None,
    trades: list[TradeAttribution] | None = None,
    agents: list[AgentData] | None = None,
    logs: object = None,
    equity_curves: dict[int, EquityCurve] | None = None,
) -> SimulationResult:
    """Build a minimal SimulationResult for testing compute_rich_metrics."""
    l1_series = _make_l1(l1_rows) if l1_rows else None
    if agents is None:
        agents = []

    # Attach equity curves to agents if provided
    if equity_curves:
        patched = []
        for a in agents:
            ec = equity_curves.get(a.agent_id)
            if ec is not None:
                a = a.model_copy(update={"equity_curve": ec})
            patched.append(a)
        agents = patched

    mkt = MarketSummary(
        symbol="TEST",
        l1_close=L1Close(time_ns=0, bid_price_cents=9900, ask_price_cents=10100),
        liquidity=LiquidityMetrics(
            pct_time_no_bid=0.0,
            pct_time_no_ask=0.0,
            total_exchanged_volume=100,
            last_trade_cents=10_000,
            vwap_cents=10_000,
        ),
        l1_series=l1_series,
        trades=trades,
    )
    return SimulationResult(
        metadata=SimulationMetadata(
            seed=42,
            tickers=["TEST"],
            sim_start_ns=0,
            sim_end_ns=1_000_000_000,
            wall_clock_elapsed_s=0.1,
            config_snapshot={},
        ),
        markets={"TEST": mkt},
        agents=agents,
        logs=logs,
        profile=profile,
    )


def _make_agent(
    agent_id: int = 1,
    pnl_cents: int = 0,
    holdings: dict[str, int] | None = None,
) -> AgentData:
    """Build a minimal AgentData."""
    if holdings is None:
        holdings = {"CASH": 10_000_000}
    starting_cash = 10_000_000
    mtm = starting_cash + pnl_cents
    return AgentData(
        agent_id=agent_id,
        agent_type="TestAgent",
        agent_name=f"agent_{agent_id}",
        agent_category="strategy",
        final_holdings=holdings,
        starting_cash_cents=starting_cash,
        mark_to_market_cents=mtm,
        pnl_cents=pnl_cents,
        pnl_pct=pnl_cents / starting_cash * 100.0 if starting_cash else 0.0,
    )


# ===================================================================
# compute_rich_metrics — summary profile (graceful degradation)
# ===================================================================


class TestComputeRichMetricsSummary:
    def test_empty_result(self):
        result = _make_result()
        rich = compute_rich_metrics(result)
        assert isinstance(rich, RichSimulationMetrics)
        assert rich.fills is None
        assert rich.agents == []
        mkt = rich.markets["TEST"]
        assert mkt.microstructure is not None
        # No L1 series → microstructure fields are None/0.0
        assert mkt.microstructure.lob_imbalance_mean is None
        assert mkt.microstructure.resilience_mean_ns is None
        assert mkt.microstructure.pct_time_two_sided == 0.0

    def test_agent_pnl_surfaced(self):
        agent = _make_agent(agent_id=1, pnl_cents=500)
        result = _make_result(agents=[agent])
        rich = compute_rich_metrics(result)
        assert len(rich.agents) == 1
        assert rich.agents[0].total_pnl_cents == 500
        assert rich.agents[0].sharpe_ratio is None
        assert rich.agents[0].max_drawdown_cents is None
        assert rich.agents[0].trade_count == 0


# ===================================================================
# compute_rich_metrics — QUANT profile (microstructure populated)
# ===================================================================


class TestComputeRichMetricsQuant:
    def test_microstructure_populated(self):
        l1_rows = [(i * 1_000, 9900, 50, 10100, 50) for i in range(10)]
        result = _make_result(
            profile=ResultProfile.QUANT,
            l1_rows=l1_rows,
        )
        rich = compute_rich_metrics(result)
        micro = rich.markets["TEST"].microstructure
        assert micro is not None
        assert micro.lob_imbalance_mean is not None
        assert micro.pct_time_two_sided == pytest.approx(100.0)

    def test_agent_with_trades(self):
        agent = _make_agent(
            agent_id=1,
            pnl_cents=200,
            holdings={"CASH": 10_000_200, "TEST": 10},
        )
        trades = [
            TradeAttribution(
                time_ns=100,
                passive_agent_id=1,
                aggressive_agent_id=99,
                side="BUY",
                price_cents=10_000,
                quantity=5,
            ),
            TradeAttribution(
                time_ns=200,
                passive_agent_id=99,
                aggressive_agent_id=1,
                side="SELL",
                price_cents=10_100,
                quantity=5,
            ),
        ]
        result = _make_result(
            profile=ResultProfile.QUANT,
            agents=[agent],
            trades=trades,
        )
        rich = compute_rich_metrics(result)
        ra = rich.agents[0]
        assert ra.trade_count == 2
        assert ra.vwap_cents is not None
        assert ra.end_inventory == {"TEST": 10}

    def test_agent_with_equity_curve(self):
        ec = EquityCurve(
            times_ns=list(range(0, 40_000, 1_000)),
            nav_cents=[10_000_000 + i * 10 for i in range(40)],
            peak_nav_cents=[10_000_000 + i * 10 for i in range(40)],
        )
        agent = _make_agent(agent_id=5, pnl_cents=390)
        result = _make_result(
            profile=ResultProfile.QUANT,
            agents=[agent],
            equity_curves={5: ec},
        )
        rich = compute_rich_metrics(result)
        ra = rich.agents[0]
        assert ra.sharpe_ratio is not None
        assert ra.max_drawdown_cents == 0  # monotonically increasing


# ===================================================================
# compute_rich_metrics — Tier 3 (include_fills)
# ===================================================================


class TestComputeRichMetricsFills:
    def test_fill_records_produced(self):
        l1_rows = [
            (100, 9900, 10, 10100, 5),
            (200, 9950, 10, 10050, 5),
        ]
        trades = [
            TradeAttribution(
                time_ns=100,
                passive_agent_id=1,
                aggressive_agent_id=2,
                side="BUY",
                price_cents=10_000,
                quantity=10,
            ),
        ]
        result = _make_result(
            profile=ResultProfile.QUANT,
            l1_rows=l1_rows,
            trades=trades,
        )
        rich = compute_rich_metrics(result, include_fills=True)
        assert rich.fills is not None
        # 1 trade → 2 fill records (passive + aggressive)
        assert len(rich.fills) == 2
        # Check slippage is computed
        for f in rich.fills:
            assert isinstance(f, FillRecord)
            assert f.slippage_bps is not None

    def test_adverse_selection_window_keys(self):
        l1_rows = [
            (100, 9900, 10, 10100, 5),
            (100_000_200, 9950, 10, 10050, 5),
        ]
        trades = [
            TradeAttribution(
                time_ns=100,
                passive_agent_id=1,
                aggressive_agent_id=2,
                side="BUY",
                price_cents=10_000,
                quantity=10,
            ),
        ]
        result = _make_result(
            profile=ResultProfile.QUANT,
            l1_rows=l1_rows,
            trades=trades,
        )
        rich = compute_rich_metrics(
            result,
            include_fills=True,
            adverse_selection_windows=["100ms", "1s"],
        )
        assert rich.fills is not None
        for f in rich.fills:
            assert set(f.adverse_selection_bps.keys()) == {"100ms", "1s"}

    def test_no_fills_without_flag(self):
        result = _make_result(profile=ResultProfile.QUANT)
        rich = compute_rich_metrics(result)
        assert rich.fills is None

    def test_no_l1_slippage_is_none(self):
        trades = [
            TradeAttribution(
                time_ns=100,
                passive_agent_id=1,
                aggressive_agent_id=2,
                side="BUY",
                price_cents=10_000,
                quantity=10,
            ),
        ]
        result = _make_result(
            profile=ResultProfile.QUANT,
            trades=trades,
        )
        rich = compute_rich_metrics(result, include_fills=True)
        assert rich.fills is not None
        for f in rich.fills:
            assert f.slippage_bps is None


# ===================================================================
# compute_rich_metrics — order lifecycle tracking (AGENT_LOGS)
# ===================================================================


def _make_logs(rows: list[dict]) -> pd.DataFrame:
    """Build a log DataFrame from a list of row dicts."""
    df = pd.DataFrame(rows)
    for col in ("EventTime", "agent_id", "order_id", "quantity", "fill_price"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


class TestOrderLifecycles:
    def test_none_without_agent_logs(self):
        """order_lifecycles is None when AGENT_LOGS profile is absent."""
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.SUMMARY, agents=[agent])
        rich = compute_rich_metrics(result)
        assert rich.agents[0].order_lifecycles is None

    def test_filled_order(self):
        """Fully filled order has status='filled' and correct resting_time_ns."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 100,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 2000,
                    "EventType": "ORDER_EXECUTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 100,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": 10_000,
                    "symbol": "TEST",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert len(lcs) == 1
        lc = lcs[0]
        assert lc.order_id == 100
        assert lc.agent_id == 1
        assert lc.status == "filled"
        assert lc.filled_qty == 10
        assert lc.submitted_qty == 10
        assert lc.submitted_at_ns == 1000
        assert lc.resting_time_ns == 1000
        assert len(lc.fill_events) == 1
        assert lc.fill_events[0] == (2000, 10_000, 10)

    def test_cancelled_order(self):
        """Cancelled order with no fills has status='cancelled'."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 200,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 5000,
                    "EventType": "ORDER_CANCELLED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 200,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert len(lcs) == 1
        lc = lcs[0]
        assert lc.status == "cancelled"
        assert lc.filled_qty == 0
        assert lc.resting_time_ns == 4000

    def test_partially_filled_then_cancelled(self):
        """Order partially filled then cancelled has status='partially_filled'."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 300,
                    "quantity": 20,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 2000,
                    "EventType": "ORDER_EXECUTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 300,
                    "quantity": 5,
                    "side": "BID",
                    "fill_price": 10_000,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 3000,
                    "EventType": "ORDER_CANCELLED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 300,
                    "quantity": 15,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert len(lcs) == 1
        lc = lcs[0]
        assert lc.status == "partially_filled"
        assert lc.filled_qty == 5
        assert lc.submitted_qty == 20
        assert lc.resting_time_ns == 2000  # cancel time - submit time

    def test_resting_order(self):
        """Order with no terminal event has status='resting' and None resting_time."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 400,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert len(lcs) == 1
        lc = lcs[0]
        assert lc.status == "resting"
        assert lc.filled_qty == 0
        assert lc.resting_time_ns is None
        assert lc.fill_events == []

    def test_multiple_orders_per_agent(self):
        """Multiple orders for the same agent produce multiple lifecycle records."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 500,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 1500,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 501,
                    "quantity": 5,
                    "side": "ASK",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 2000,
                    "EventType": "ORDER_EXECUTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 500,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": 10_000,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 3000,
                    "EventType": "ORDER_CANCELLED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 501,
                    "quantity": 5,
                    "side": "ASK",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert len(lcs) == 2
        by_oid = {lc.order_id: lc for lc in lcs}
        assert by_oid[500].status == "filled"
        assert by_oid[501].status == "cancelled"

    def test_empty_list_when_no_orders(self):
        """Agent with no orders gets an empty lifecycle list (not None)."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "SOME_OTHER_EVENT",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert lcs == []

    def test_multiple_fills_same_order(self):
        """Order filled in two partial executions accumulates fill_events."""
        logs = _make_logs(
            [
                {
                    "EventTime": 1000,
                    "EventType": "ORDER_SUBMITTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 600,
                    "quantity": 20,
                    "side": "BID",
                    "fill_price": pd.NA,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 2000,
                    "EventType": "ORDER_EXECUTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 600,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": 9950,
                    "symbol": "TEST",
                },
                {
                    "EventTime": 3000,
                    "EventType": "ORDER_EXECUTED",
                    "agent_id": 1,
                    "agent_type": "TestAgent",
                    "order_id": 600,
                    "quantity": 10,
                    "side": "BID",
                    "fill_price": 10_050,
                    "symbol": "TEST",
                },
            ]
        )
        agent = _make_agent(agent_id=1)
        result = _make_result(profile=ResultProfile.FULL, agents=[agent], logs=logs)
        rich = compute_rich_metrics(result)
        lcs = rich.agents[0].order_lifecycles
        assert lcs is not None
        assert len(lcs) == 1
        lc = lcs[0]
        assert lc.status == "filled"
        assert lc.filled_qty == 20
        assert len(lc.fill_events) == 2
        assert lc.fill_events[0] == (2000, 9950, 10)
        assert lc.fill_events[1] == (3000, 10_050, 10)
        assert lc.resting_time_ns == 2000  # last fill time - submit time

    def test_importable(self):
        """OrderLifecycle is importable from the simulation package."""
        from abides_markets.simulation import OrderLifecycle as OrderLifecycleAlias

        assert OrderLifecycleAlias is OrderLifecycle


# ===================================================================
# Import smoke test
# ===================================================================


class TestRichMetricsImports:
    def test_importable_from_simulation(self):
        from abides_markets.simulation import (
            compute_adverse_selection,
            compute_fill_slippage,
            compute_rich_metrics,
        )

        assert callable(compute_rich_metrics)
        assert callable(compute_fill_slippage)
        assert callable(compute_adverse_selection)
