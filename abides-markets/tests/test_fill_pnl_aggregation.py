"""Tests for FILL_PNL aggregation / EquityCurve (P1 item 10).

Covers:
- EquityCurve model construction, immutability, max_drawdown_cents
- _extract_equity_curve from mock agent logs
- EQUITY_CURVE profile flag gating
- AgentData integration
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# EquityCurve model
# ---------------------------------------------------------------------------
class TestEquityCurve:
    def test_construction(self):
        from abides_markets.simulation.result import EquityCurve

        ec = EquityCurve(
            times_ns=[1_000, 2_000, 3_000],
            nav_cents=[10_000, 10_500, 10_200],
            peak_nav_cents=[10_000, 10_500, 10_500],
        )
        assert ec.times_ns == [1_000, 2_000, 3_000]
        assert ec.nav_cents == [10_000, 10_500, 10_200]
        assert ec.peak_nav_cents == [10_000, 10_500, 10_500]

    def test_frozen(self):
        from abides_markets.simulation.result import EquityCurve

        ec = EquityCurve(times_ns=[1], nav_cents=[100], peak_nav_cents=[100])
        with pytest.raises((TypeError, ValidationError)):
            ec.times_ns = [2]

    def test_max_drawdown_cents_no_drawdown(self):
        from abides_markets.simulation.result import EquityCurve

        ec = EquityCurve(
            times_ns=[1, 2, 3],
            nav_cents=[100, 200, 300],
            peak_nav_cents=[100, 200, 300],
        )
        assert ec.max_drawdown_cents == 0

    def test_max_drawdown_cents_with_drawdown(self):
        from abides_markets.simulation.result import EquityCurve

        ec = EquityCurve(
            times_ns=[1, 2, 3, 4],
            nav_cents=[10_000, 10_500, 9_800, 10_300],
            peak_nav_cents=[10_000, 10_500, 10_500, 10_500],
        )
        # drawdown at t=3: 10_500 - 9_800 = 700
        assert ec.max_drawdown_cents == 700

    def test_max_drawdown_cents_empty(self):
        from abides_markets.simulation.result import EquityCurve

        ec = EquityCurve(times_ns=[], nav_cents=[], peak_nav_cents=[])
        assert ec.max_drawdown_cents == 0


# ---------------------------------------------------------------------------
# _extract_equity_curve
# ---------------------------------------------------------------------------
class TestExtractEquityCurve:
    def _make_agent_with_log(self, log_entries):
        """Create a fake agent with a log attribute."""

        class FakeAgent:
            pass

        agent = FakeAgent()
        agent.log = log_entries
        return agent

    def test_extracts_fill_pnl_events(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        agent = self._make_agent_with_log(
            [
                (
                    1_000,
                    "FILL_PNL",
                    {"nav": 10_000, "peak_nav": 10_000, "symbol": "AAPL"},
                ),
                (2_000, "ORDER_EXECUTED", {"order_id": 1}),
                (
                    3_000,
                    "FILL_PNL",
                    {"nav": 10_200, "peak_nav": 10_200, "symbol": "AAPL"},
                ),
            ]
        )
        ec = _extract_equity_curve(agent)
        assert ec is not None
        assert len(ec.times_ns) == 2
        assert ec.times_ns == [1_000, 3_000]
        assert ec.nav_cents == [10_000, 10_200]
        assert ec.peak_nav_cents == [10_000, 10_200]

    def test_no_fill_pnl_returns_none(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        agent = self._make_agent_with_log(
            [
                (1_000, "ORDER_EXECUTED", {"order_id": 1}),
                (2_000, "ORDER_ACCEPTED", {"order_id": 2}),
            ]
        )
        assert _extract_equity_curve(agent) is None

    def test_empty_log_returns_none(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        agent = self._make_agent_with_log([])
        assert _extract_equity_curve(agent) is None

    def test_no_log_attribute_returns_none(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        class NoLogAgent:
            pass

        assert _extract_equity_curve(NoLogAgent()) is None

    def test_fill_pnl_with_missing_nav_skipped(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        agent = self._make_agent_with_log(
            [
                (1_000, "FILL_PNL", {"nav": 10_000, "peak_nav": 10_000}),
                (2_000, "FILL_PNL", {"nav": None, "peak_nav": 10_000}),
                (3_000, "FILL_PNL", {"peak_nav": 10_000}),
            ]
        )
        ec = _extract_equity_curve(agent)
        assert ec is not None
        assert len(ec.times_ns) == 1

    def test_fill_pnl_non_dict_data_skipped(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        agent = self._make_agent_with_log(
            [
                (1_000, "FILL_PNL", "not a dict"),
                (2_000, "FILL_PNL", {"nav": 10_000, "peak_nav": 10_000}),
            ]
        )
        ec = _extract_equity_curve(agent)
        assert ec is not None
        assert len(ec.times_ns) == 1

    def test_drawdown_from_extracted_curve(self):
        from abides_markets.simulation.runner import _extract_equity_curve

        agent = self._make_agent_with_log(
            [
                (1_000, "FILL_PNL", {"nav": 10_000, "peak_nav": 10_000, "symbol": "X"}),
                (2_000, "FILL_PNL", {"nav": 10_500, "peak_nav": 10_500, "symbol": "X"}),
                (3_000, "FILL_PNL", {"nav": 9_800, "peak_nav": 10_500, "symbol": "X"}),
                (4_000, "FILL_PNL", {"nav": 10_100, "peak_nav": 10_500, "symbol": "X"}),
            ]
        )
        ec = _extract_equity_curve(agent)
        assert ec is not None
        assert ec.max_drawdown_cents == 700


# ---------------------------------------------------------------------------
# Profile gating
# ---------------------------------------------------------------------------
class TestEquityCurveProfile:
    def test_equity_curve_in_quant(self):
        from abides_markets.simulation.profiles import ResultProfile

        assert ResultProfile.EQUITY_CURVE in ResultProfile.QUANT

    def test_equity_curve_not_in_summary(self):
        from abides_markets.simulation.profiles import ResultProfile

        assert ResultProfile.EQUITY_CURVE not in ResultProfile.SUMMARY

    def test_equity_curve_in_full(self):
        from abides_markets.simulation.profiles import ResultProfile

        assert ResultProfile.EQUITY_CURVE in ResultProfile.FULL


# ---------------------------------------------------------------------------
# AgentData integration
# ---------------------------------------------------------------------------
class TestAgentDataEquityCurve:
    def test_agent_data_with_equity_curve(self):
        from abides_markets.simulation.result import AgentData, EquityCurve

        ec = EquityCurve(
            times_ns=[1_000],
            nav_cents=[10_000],
            peak_nav_cents=[10_000],
        )
        ad = AgentData(
            agent_id=1,
            agent_type="TestAgent",
            agent_name="agent_1",
            agent_category="background",
            final_holdings={"CASH": 10_000},
            starting_cash_cents=10_000,
            mark_to_market_cents=10_000,
            pnl_cents=0,
            pnl_pct=0.0,
            equity_curve=ec,
        )
        assert ad.equity_curve is not None
        assert ad.equity_curve.nav_cents == [10_000]

    def test_agent_data_without_equity_curve(self):
        from abides_markets.simulation.result import AgentData

        ad = AgentData(
            agent_id=1,
            agent_type="TestAgent",
            agent_name="agent_1",
            agent_category="background",
            final_holdings={"CASH": 10_000},
            starting_cash_cents=10_000,
            mark_to_market_cents=10_000,
            pnl_cents=0,
            pnl_pct=0.0,
        )
        assert ad.equity_curve is None
