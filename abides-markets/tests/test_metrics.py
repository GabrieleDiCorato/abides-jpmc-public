"""Tests for abides_markets.simulation.metrics — standalone metric computation.

Tests verify that each compute_* function produces correct results from
plain Python data (no live agents or exchange objects required).
"""

from __future__ import annotations

import numpy as np
import pytest

from abides_markets.simulation.metrics import (
    compute_agent_pnl,
    compute_equity_curve,
    compute_execution_metrics,
    compute_l1_close,
    compute_l1_series,
    compute_l2_series,
    compute_liquidity_metrics,
    compute_metrics,
    compute_trade_attribution,
    compute_vwap,
)
from abides_markets.simulation.result import (
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
