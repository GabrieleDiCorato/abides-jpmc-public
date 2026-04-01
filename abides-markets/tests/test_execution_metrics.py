"""Tests for ExecutionMetrics (P0 item 4).

Covers:
- Direct ExecutionMetrics construction and field validation
- VWAP slippage calculation
- Participation rate calculation
- Implementation shortfall calculation
- None / edge-case handling
- Extraction from mock execution agents via _extract_execution_metrics
- AgentData with execution_metrics field
- summary_dict() execution_summary section
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from abides_markets.simulation.profiles import ResultProfile
from abides_markets.simulation.result import (
    AgentData,
    ExecutionMetrics,
    L1Close,
    LiquidityMetrics,
    MarketSummary,
    SimulationMetadata,
    SimulationResult,
)
from abides_markets.simulation.runner import _extract_execution_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeExecutionAgent:
    """Minimal duck-typed execution agent for testing metrics extraction."""

    def __init__(
        self,
        *,
        quantity: int = 1000,
        executed_quantity: int = 800,
        execution_history: list[dict[str, Any]] | None = None,
        symbol: str = "ABM",
        last_bid: int | None = 10000,
        last_ask: int | None = 10100,
    ) -> None:
        self.quantity = quantity
        self.executed_quantity = executed_quantity
        self.execution_history = (
            execution_history if execution_history is not None else []
        )
        self.symbol = symbol
        self.last_bid = last_bid
        self.last_ask = last_ask


class _FakeNonExecutionAgent:
    """Agent that does not have execution_history — should return None."""

    pass


def _make_liquidity(
    vwap_cents: int | None = 10050,
    total_exchanged_volume: int = 5000,
) -> LiquidityMetrics:
    return LiquidityMetrics(
        pct_time_no_bid=0.0,
        pct_time_no_ask=0.0,
        total_exchanged_volume=total_exchanged_volume,
        last_trade_cents=10050,
        vwap_cents=vwap_cents,
    )


# ---------------------------------------------------------------------------
# Direct ExecutionMetrics construction tests
# ---------------------------------------------------------------------------


class TestExecutionMetricsModel:
    def test_basic_construction(self) -> None:
        em = ExecutionMetrics(
            target_quantity=1000,
            filled_quantity=800,
            fill_rate_pct=80.0,
        )
        assert em.target_quantity == 1000
        assert em.filled_quantity == 800
        assert em.fill_rate_pct == 80.0
        assert em.avg_fill_price_cents is None
        assert em.vwap_cents is None
        assert em.vwap_slippage_bps is None
        assert em.participation_rate_pct is None
        assert em.arrival_price_cents is None
        assert em.implementation_shortfall_bps is None

    def test_full_construction(self) -> None:
        em = ExecutionMetrics(
            target_quantity=1000,
            filled_quantity=1000,
            fill_rate_pct=100.0,
            avg_fill_price_cents=10020,
            vwap_cents=10000,
            vwap_slippage_bps=20,
            participation_rate_pct=15.0,
            arrival_price_cents=9950,
            implementation_shortfall_bps=70,
        )
        assert em.avg_fill_price_cents == 10020
        assert em.vwap_slippage_bps == 20
        assert em.participation_rate_pct == 15.0
        assert em.arrival_price_cents == 9950
        assert em.implementation_shortfall_bps == 70

    def test_frozen(self) -> None:
        em = ExecutionMetrics(
            target_quantity=1000, filled_quantity=500, fill_rate_pct=50.0
        )
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            em.filled_quantity = 600


# ---------------------------------------------------------------------------
# _extract_execution_metrics tests
# ---------------------------------------------------------------------------


class TestExtractExecutionMetrics:
    def test_non_execution_agent_returns_none(self) -> None:
        agent = _FakeNonExecutionAgent()
        result = _extract_execution_metrics(agent, {})  # type: ignore[arg-type]
        assert result is None

    def test_no_fills(self) -> None:
        agent = _FakeExecutionAgent(
            quantity=1000,
            executed_quantity=0,
            execution_history=[],
            last_bid=None,
            last_ask=None,
        )
        liquidity = {"ABM": _make_liquidity()}
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None
        assert result.target_quantity == 1000
        assert result.filled_quantity == 0
        assert result.fill_rate_pct == 0.0
        assert result.avg_fill_price_cents is None
        assert result.vwap_slippage_bps is None
        assert result.participation_rate_pct is None
        assert result.arrival_price_cents is None
        assert result.implementation_shortfall_bps is None

    def test_normal_fills(self) -> None:
        history = [
            {"fill_price": 10000, "quantity": 300},
            {"fill_price": 10100, "quantity": 500},
        ]
        agent = _FakeExecutionAgent(
            quantity=1000,
            executed_quantity=800,
            execution_history=history,
            last_bid=10000,
            last_ask=10100,
        )
        liquidity = {
            "ABM": _make_liquidity(vwap_cents=10050, total_exchanged_volume=5000)
        }
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None

        # avg fill = (10000*300 + 10100*500) // 800 = (3000000 + 5050000) // 800 = 10062
        assert result.avg_fill_price_cents == 10062
        assert result.fill_rate_pct == 80.0

        # vwap_slippage = (10062 - 10050) / 10050 * 10000 = 11.94 → 11 (integer division)
        assert result.vwap_slippage_bps == 11

        # participation = 800 / 5000 * 100 = 16.0
        assert result.participation_rate_pct == 16.0

        # arrival = (10000 + 10100) // 2 = 10050
        assert result.arrival_price_cents == 10050

        # impl shortfall = (10062 - 10050) / 10050 * 10000 = 11.94 → 11
        assert result.implementation_shortfall_bps == 11

    def test_negative_slippage(self) -> None:
        """Buyer fills below VWAP — negative slippage (good)."""
        history = [{"fill_price": 9900, "quantity": 500}]
        agent = _FakeExecutionAgent(
            quantity=500,
            executed_quantity=500,
            execution_history=history,
            last_bid=9900,
            last_ask=10000,
        )
        liquidity = {"ABM": _make_liquidity(vwap_cents=10000)}
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None
        assert result.avg_fill_price_cents == 9900
        # (9900 - 10000) / 10000 * 10000 = -100
        assert result.vwap_slippage_bps == -100

    def test_no_vwap_in_liquidity(self) -> None:
        history = [{"fill_price": 10000, "quantity": 100}]
        agent = _FakeExecutionAgent(
            quantity=100,
            executed_quantity=100,
            execution_history=history,
        )
        liquidity = {"ABM": _make_liquidity(vwap_cents=None, total_exchanged_volume=0)}
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None
        assert result.vwap_slippage_bps is None
        assert result.participation_rate_pct is None

    def test_no_arrival_price(self) -> None:
        """Agent has no bid/ask → arrival_price is None."""
        history = [{"fill_price": 10000, "quantity": 100}]
        agent = _FakeExecutionAgent(
            quantity=100,
            executed_quantity=100,
            execution_history=history,
            last_bid=None,
            last_ask=None,
        )
        liquidity = {"ABM": _make_liquidity()}
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None
        assert result.arrival_price_cents is None
        assert result.implementation_shortfall_bps is None

    def test_symbol_not_in_liquidity(self) -> None:
        """Agent symbol not found in liquidity map."""
        history = [{"fill_price": 10000, "quantity": 100}]
        agent = _FakeExecutionAgent(
            quantity=100,
            executed_quantity=100,
            execution_history=history,
            symbol="XYZ",
        )
        # Only "ABM" in liquidity, not "XYZ"
        liquidity = {"ABM": _make_liquidity()}
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None
        assert result.vwap_cents is None
        assert result.participation_rate_pct is None

    def test_zero_target_quantity(self) -> None:
        agent = _FakeExecutionAgent(
            quantity=0, executed_quantity=0, execution_history=[]
        )
        liquidity = {"ABM": _make_liquidity()}
        result = _extract_execution_metrics(agent, liquidity)  # type: ignore[arg-type]
        assert result is not None
        assert result.fill_rate_pct == 0.0


# ---------------------------------------------------------------------------
# AgentData with execution_metrics
# ---------------------------------------------------------------------------


class TestAgentDataExecutionMetrics:
    def test_agent_data_without_metrics(self) -> None:
        ad = AgentData(
            agent_id=1,
            agent_type="NoiseAgent",
            agent_name="noise_1",
            agent_category="background",
            final_holdings={"CASH": 1000000},
            starting_cash_cents=1000000,
            mark_to_market_cents=1000000,
            pnl_cents=0,
            pnl_pct=0.0,
        )
        assert ad.execution_metrics is None

    def test_agent_data_with_metrics(self) -> None:
        em = ExecutionMetrics(
            target_quantity=500,
            filled_quantity=500,
            fill_rate_pct=100.0,
            avg_fill_price_cents=10000,
            vwap_cents=10000,
            vwap_slippage_bps=0,
            participation_rate_pct=10.0,
            arrival_price_cents=10000,
            implementation_shortfall_bps=0,
        )
        ad = AgentData(
            agent_id=5,
            agent_type="POVExecutionAgent",
            agent_name="pov_exec",
            agent_category="execution",
            final_holdings={"CASH": 500000, "ABM": 500},
            starting_cash_cents=1000000,
            mark_to_market_cents=1000000,
            pnl_cents=0,
            pnl_pct=0.0,
            execution_metrics=em,
        )
        assert ad.execution_metrics is not None
        assert ad.execution_metrics.target_quantity == 500


# ---------------------------------------------------------------------------
# summary_dict() with execution_summary
# ---------------------------------------------------------------------------


class TestSummaryDictExecutionSummary:
    def _make_result(
        self, *, exec_metrics: ExecutionMetrics | None = None
    ) -> SimulationResult:
        metadata = SimulationMetadata(
            seed=42,
            tickers=["ABM"],
            sim_start_ns=0,
            sim_end_ns=1_000_000_000,
            wall_clock_elapsed_s=1.0,
            config_snapshot={},
        )
        l1 = L1Close(time_ns=999, bid_price_cents=10000, ask_price_cents=10100)
        liq = _make_liquidity()
        market = MarketSummary(symbol="ABM", l1_close=l1, liquidity=liq)

        agents = [
            AgentData(
                agent_id=1,
                agent_type="POVExecutionAgent",
                agent_name="pov",
                agent_category="execution",
                final_holdings={"CASH": 500000},
                starting_cash_cents=1000000,
                mark_to_market_cents=500000,
                pnl_cents=-500000,
                pnl_pct=-50.0,
                execution_metrics=exec_metrics,
            )
        ]

        return SimulationResult(
            metadata=metadata,
            markets={"ABM": market},
            agents=agents,
            profile=ResultProfile.SUMMARY,
        )

    def test_no_execution_metrics(self) -> None:
        result = self._make_result()
        sd = result.summary_dict()
        assert "execution_summary" in sd
        assert sd["execution_summary"] == []

    def test_with_execution_metrics(self) -> None:
        em = ExecutionMetrics(
            target_quantity=1000,
            filled_quantity=800,
            fill_rate_pct=80.0,
            avg_fill_price_cents=10020,
            vwap_cents=10000,
            vwap_slippage_bps=20,
            participation_rate_pct=16.0,
            arrival_price_cents=10050,
            implementation_shortfall_bps=-29,
        )
        result = self._make_result(exec_metrics=em)
        sd = result.summary_dict()
        assert len(sd["execution_summary"]) == 1
        entry = sd["execution_summary"][0]
        assert entry["agent_id"] == 1
        assert entry["target_quantity"] == 1000
        assert entry["filled_quantity"] == 800
        assert entry["vwap_slippage_bps"] == 20
        assert entry["participation_rate_pct"] == 16.0
        assert entry["arrival_price_cents"] == 10050
        assert entry["implementation_shortfall_bps"] == -29

    def test_summary_narrative_includes_execution(self) -> None:
        em = ExecutionMetrics(
            target_quantity=1000,
            filled_quantity=800,
            fill_rate_pct=80.0,
            avg_fill_price_cents=10020,
            vwap_slippage_bps=20,
            participation_rate_pct=16.0,
        )
        result = self._make_result(exec_metrics=em)
        text = result.summary()
        assert "Execution agents:" in text
        assert "filled=800/1000" in text
        assert "80.0%" in text
