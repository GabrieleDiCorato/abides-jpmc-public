"""Numerical tests for VWAP, execution metrics, and LiquidityMetrics computation.

These tests verify the VWAP formula (total_value // total_qty from EXEC entries),
ExecutionMetrics derivation (avg_fill, vwap_slippage, participation, impl_shortfall),
and edge cases like zero volume, single trade, and integer division behavior.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from abides_markets.simulation.result import LiquidityMetrics
from abides_markets.simulation.runner import (
    _extract_execution_metrics,
    _extract_liquidity,
)

# ===================================================================
# VWAP computation via _extract_liquidity
# ===================================================================


class _FakeBook:
    """Minimal order book stub with history and last_trade."""

    def __init__(self, history=None, last_trade=None):
        self.history = history or []
        self.last_trade = last_trade


class _FakeExchange:
    """Minimal exchange stub without metric trackers."""

    pass


class TestVWAPfromHistory:
    """Verify VWAP = total_value // total_qty from EXEC entries."""

    def test_single_trade(self):
        """One execution at 10000 cents, qty 100 → VWAP=10000."""
        book = _FakeBook(history=[{"type": "EXEC", "price": 10_000, "quantity": 100}])
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.vwap_cents == 10_000

    def test_two_trades_equal_weight(self):
        """Two trades at same qty: (10000*100 + 10200*100) // 200 = 10100."""
        book = _FakeBook(
            history=[
                {"type": "EXEC", "price": 10_000, "quantity": 100},
                {"type": "EXEC", "price": 10_200, "quantity": 100},
            ]
        )
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.vwap_cents == 10_100

    def test_two_trades_unequal_weight(self):
        """Unequal qty: (10000*300 + 10100*100) // 400 = 10025."""
        book = _FakeBook(
            history=[
                {"type": "EXEC", "price": 10_000, "quantity": 300},
                {"type": "EXEC", "price": 10_100, "quantity": 100},
            ]
        )
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        # total_value = 3_000_000 + 1_010_000 = 4_010_000
        # total_qty = 400
        # vwap = 4_010_000 // 400 = 10_025
        assert liq.vwap_cents == 10_025

    def test_integer_division_truncates(self):
        """VWAP uses floor division — verify truncation behavior."""
        # 10001*1 + 10000*2 = 30001 // 3 = 10000 (truncated from 10000.333...)
        book = _FakeBook(
            history=[
                {"type": "EXEC", "price": 10_001, "quantity": 1},
                {"type": "EXEC", "price": 10_000, "quantity": 2},
            ]
        )
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.vwap_cents == 10_000  # not 10001

    def test_no_exec_entries(self):
        """No EXEC entries → VWAP is None."""
        book = _FakeBook(
            history=[
                {"type": "LIMIT_ORDER", "price": 10_000, "quantity": 100},
                {"type": "CANCEL", "price": 10_000, "quantity": 100},
            ]
        )
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.vwap_cents is None

    def test_empty_history(self):
        """Empty history → VWAP is None."""
        book = _FakeBook(history=[])
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.vwap_cents is None

    def test_exec_with_none_price_skipped(self):
        """EXEC entries with None price are ignored."""
        book = _FakeBook(
            history=[
                {"type": "EXEC", "price": None, "quantity": 100},
                {"type": "EXEC", "price": 10_000, "quantity": 50},
            ]
        )
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.vwap_cents == 10_000  # only the valid entry

    def test_many_trades_numerical(self):
        """Pin VWAP for a 5-trade sequence."""
        # Prices: 100, 102, 98, 101, 99 (cents). Qty: 10 each.
        prices = [100, 102, 98, 101, 99]
        history = [{"type": "EXEC", "price": p, "quantity": 10} for p in prices]
        book = _FakeBook(history=history)
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        expected = sum(prices) * 10 // (10 * len(prices))
        assert liq.vwap_cents == expected

    def test_last_trade_from_book(self):
        """last_trade_cents falls back to order_book.last_trade."""
        book = _FakeBook(history=[], last_trade=9_500)
        liq = _extract_liquidity(_FakeExchange(), "TEST", book)
        assert liq.last_trade_cents == 9_500


# ===================================================================
# ExecutionMetrics computation
# ===================================================================


class TestExecutionMetrics:
    """Test _extract_execution_metrics derived metrics."""

    def _make_agent(
        self,
        execution_history=None,
        quantity=100,
        executed_quantity=100,
        symbol="TEST",
        last_bid=None,
        last_ask=None,
    ):
        agent = MagicMock()
        agent.execution_history = execution_history
        agent.quantity = quantity
        agent.executed_quantity = executed_quantity
        agent.symbol = symbol
        agent.last_bid = last_bid
        agent.last_ask = last_ask
        return agent

    def test_avg_fill_single_fill(self):
        agent = self._make_agent(
            execution_history=[{"fill_price": 10_000, "quantity": 100}]
        )
        em = _extract_execution_metrics(agent, {})
        assert em is not None
        assert em.avg_fill_price_cents == 10_000

    def test_avg_fill_multiple_fills(self):
        """avg_fill = total_value // total_qty from execution_history."""
        agent = self._make_agent(
            execution_history=[
                {"fill_price": 10_000, "quantity": 60},
                {"fill_price": 10_100, "quantity": 40},
            ]
        )
        em = _extract_execution_metrics(agent, {})
        # total_value = 600_000 + 404_000 = 1_004_000
        # total_qty = 100
        # avg_fill = 1_004_000 // 100 = 10_040
        assert em.avg_fill_price_cents == 10_040

    def test_vwap_slippage_bps(self):
        """Slippage = (avg_fill - vwap) * 10_000 // vwap."""
        agent = self._make_agent(
            execution_history=[{"fill_price": 10_100, "quantity": 100}],
            symbol="TEST",
        )
        liquidity = {
            "TEST": LiquidityMetrics(
                pct_time_no_bid=0,
                pct_time_no_ask=0,
                total_exchanged_volume=1000,
                vwap_cents=10_000,
            )
        }
        em = _extract_execution_metrics(agent, liquidity)
        # slippage = (10_100 - 10_000) * 10_000 // 10_000 = 100
        assert em.vwap_slippage_bps == 100

    def test_vwap_slippage_negative(self):
        """Negative slippage: filled below VWAP (good for buyer)."""
        agent = self._make_agent(
            execution_history=[{"fill_price": 9_900, "quantity": 100}],
            symbol="TEST",
        )
        liquidity = {
            "TEST": LiquidityMetrics(
                pct_time_no_bid=0,
                pct_time_no_ask=0,
                total_exchanged_volume=1000,
                vwap_cents=10_000,
            )
        }
        em = _extract_execution_metrics(agent, liquidity)
        # slippage = (9_900 - 10_000) * 10_000 // 10_000 = -100
        assert em.vwap_slippage_bps == -100

    def test_participation_rate(self):
        """participation = filled / total_volume * 100."""
        agent = self._make_agent(
            execution_history=[{"fill_price": 10_000, "quantity": 50}],
            executed_quantity=50,
            symbol="TEST",
        )
        liquidity = {
            "TEST": LiquidityMetrics(
                pct_time_no_bid=0,
                pct_time_no_ask=0,
                total_exchanged_volume=1000,
                vwap_cents=10_000,
            )
        }
        em = _extract_execution_metrics(agent, liquidity)
        assert em.participation_rate_pct == pytest.approx(5.0)

    def test_fill_rate(self):
        """fill_rate = filled / target * 100."""
        agent = self._make_agent(
            execution_history=[{"fill_price": 10_000, "quantity": 75}],
            quantity=100,
            executed_quantity=75,
        )
        em = _extract_execution_metrics(agent, {})
        assert em.fill_rate_pct == pytest.approx(75.0)

    def test_impl_shortfall(self):
        """impl_shortfall = (avg_fill - arrival) * 10_000 // arrival."""
        agent = self._make_agent(
            execution_history=[{"fill_price": 10_200, "quantity": 100}],
            last_bid=10_000,
            last_ask=10_100,
        )
        em = _extract_execution_metrics(agent, {})
        # arrival = (10_000 + 10_100) // 2 = 10_050
        # shortfall = (10_200 - 10_050) * 10_000 // 10_050 = 149
        assert em.arrival_price_cents == 10_050
        assert em.implementation_shortfall_bps == 149

    def test_none_for_non_execution_agent(self):
        """Non-execution agents (no execution_history) return None."""
        agent = MagicMock()
        agent.execution_history = None
        agent.quantity = None
        agent.filled_quantity = None
        em = _extract_execution_metrics(agent, {})
        assert em is None

    def test_zero_target_quantity(self):
        """Target quantity = 0 → fill_rate = 0."""
        agent = self._make_agent(
            execution_history=[],
            quantity=0,
            executed_quantity=0,
        )
        em = _extract_execution_metrics(agent, {})
        assert em.fill_rate_pct == 0.0

    def test_no_fills_avg_none(self):
        """Empty execution history → avg_fill is None."""
        agent = self._make_agent(
            execution_history=[],
            quantity=100,
            executed_quantity=0,
        )
        em = _extract_execution_metrics(agent, {})
        assert em.avg_fill_price_cents is None
        assert em.vwap_slippage_bps is None


# ===================================================================
# LiquidityMetrics model constraints
# ===================================================================


class TestLiquidityMetricsModel:
    """Verify LiquidityMetrics frozen model behavior."""

    def test_frozen(self):
        liq = LiquidityMetrics(
            pct_time_no_bid=0,
            pct_time_no_ask=0,
            total_exchanged_volume=100,
        )
        with pytest.raises((TypeError, ValidationError)):
            liq.total_exchanged_volume = 999

    def test_defaults(self):
        liq = LiquidityMetrics(
            pct_time_no_bid=5.0,
            pct_time_no_ask=3.0,
            total_exchanged_volume=0,
        )
        assert liq.last_trade_cents is None
        assert liq.vwap_cents is None
