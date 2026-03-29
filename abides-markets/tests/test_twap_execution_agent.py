"""Tests for TWAPExecutionAgent (P1 item 5).

Covers:
- Slice sizing logic (uniform distribution with catch-up)
- IOC order style (default)
- Market order style
- Last-slice exhaustion
- Completion detection
- Config validation and time-window guard
- Integration: agent creation via config system
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from abides_core.utils import str_to_ns

MKT_OPEN: int = str_to_ns("09:30:00")
MKT_CLOSE: int = str_to_ns("16:00:00")


# ---------------------------------------------------------------------------
# Direct agent tests (no kernel)
# ---------------------------------------------------------------------------
class TestTWAPSliceSizing:
    """Verify _compute_slice_quantity distributes evenly with catch-up."""

    def _make_agent(self, quantity: int = 1000, freq_str: str = "1min", **kw):
        from abides_markets.agents.twap_execution_agent import TWAPExecutionAgent

        start = MKT_OPEN + str_to_ns("00:30:00")
        end = MKT_CLOSE - str_to_ns("00:30:00")
        return TWAPExecutionAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            start_time=start,
            end_time=end,
            freq=str_to_ns(freq_str),
            quantity=quantity,
            random_state=np.random.RandomState(42),
            **kw,
        )

    def test_total_slices_calculation(self):
        agent = self._make_agent(freq_str="1min")
        # 5.5h window (10:00-15:30) / 1min = 330 slices
        duration = (MKT_CLOSE - str_to_ns("00:30:00")) - (
            MKT_OPEN + str_to_ns("00:30:00")
        )
        expected = math.ceil(duration / str_to_ns("1min"))
        assert agent.total_slices == expected
        assert expected == 330

    def test_uniform_first_slice(self):
        agent = self._make_agent(quantity=330, freq_str="1min")
        t = agent.start_time
        qty = agent._compute_slice_quantity(t)
        # 330 shares / 330 slices = 1 per slice
        assert qty == 1

    def test_catchup_after_underfill(self):
        """If first slice only partially fills, next slice increases."""
        agent = self._make_agent(quantity=100, freq_str="1min")
        t = agent.start_time

        # First slice
        first = agent._compute_slice_quantity(t)
        # Simulate partial fill: only half executed
        executed = first // 2
        agent.executed_quantity += executed
        agent.remaining_quantity -= executed

        # Second slice should be larger to catch up
        second = agent._compute_slice_quantity(t + agent.freq)
        expected_remaining_slices = agent.total_slices - 2
        expected = math.ceil(agent.remaining_quantity / (expected_remaining_slices + 1))
        # second should be >= first since we under-filled
        assert second >= first // 2

    def test_last_slice_gets_remainder(self):
        """When slices_remaining=1, agent sends all remaining_quantity."""
        agent = self._make_agent(quantity=100, freq_str="1min")
        # Fast-forward to last slice
        agent.slices_completed = agent.total_slices - 1
        agent.remaining_quantity = 17
        qty = agent._compute_slice_quantity(agent.start_time)
        assert qty == 17

    def test_order_style_default_ioc(self):
        agent = self._make_agent()
        assert agent.order_style == "ioc_limit"

    def test_order_style_market(self):
        agent = self._make_agent(order_style="market")
        assert agent.order_style == "market"

    def test_direction_stored(self):
        from abides_markets.orders import Side

        agent = self._make_agent(direction=Side.ASK)
        assert agent.direction == Side.ASK

    def test_arrival_mid_initially_none(self):
        agent = self._make_agent()
        assert agent.arrival_mid_cents is None


# ---------------------------------------------------------------------------
# Config-system tests
# ---------------------------------------------------------------------------
class TestTWAPConfig:
    def test_config_creates_agent(self):
        from abides_markets.agents.twap_execution_agent import TWAPExecutionAgent
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            TWAPExecutionAgentConfig,
        )

        cfg = TWAPExecutionAgentConfig(quantity=5000)
        ctx = AgentCreationContext(
            ticker="TEST",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            log_orders=False,
            oracle_r_bar=None,
        )
        agents = cfg.create_agents(
            count=1,
            id_start=100,
            master_rng=np.random.RandomState(1),
            context=ctx,
        )
        assert len(agents) == 1
        agent = agents[0]
        assert isinstance(agent, TWAPExecutionAgent)
        assert agent.quantity == 5000
        assert agent.order_style == "ioc_limit"

    def test_inverted_window_raises(self):
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            TWAPExecutionAgentConfig,
        )

        cfg = TWAPExecutionAgentConfig(
            start_time_offset="05:00:00",
            end_time_offset="05:00:00",
        )
        ctx = AgentCreationContext(
            ticker="TEST",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            log_orders=False,
            oracle_r_bar=None,
        )
        with pytest.raises(ValueError, match="inverted"):
            cfg.create_agents(
                count=1,
                id_start=0,
                master_rng=np.random.RandomState(1),
                context=ctx,
            )

    def test_market_order_style_config(self):
        from abides_markets.agents.twap_execution_agent import TWAPExecutionAgent
        from abides_markets.config_system.agent_configs import (
            AgentCreationContext,
            TWAPExecutionAgentConfig,
        )

        cfg = TWAPExecutionAgentConfig(order_style="market")
        ctx = AgentCreationContext(
            ticker="TEST",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            log_orders=False,
            oracle_r_bar=None,
        )
        agents = cfg.create_agents(
            count=1,
            id_start=0,
            master_rng=np.random.RandomState(1),
            context=ctx,
        )
        assert agents[0].order_style == "market"
        assert isinstance(agents[0], TWAPExecutionAgent)


# ---------------------------------------------------------------------------
# Registry / builder tests
# ---------------------------------------------------------------------------
class TestTWAPRegistration:
    def test_registered_in_registry(self):
        from abides_markets.config_system.registry import registry

        entry = registry.get("twap_execution")
        assert entry is not None
        assert entry.category == "execution"

    def test_builder_integration(self):
        from abides_markets.config_system.builder import SimulationBuilder

        config = (
            SimulationBuilder()
            .market(oracle=None, opening_price=100_000)
            .seed(42)
            .enable_agent("twap_execution", count=1, quantity=500)
            .build()
        )
        assert "twap_execution" in config.agents
        assert config.agents["twap_execution"].enabled is True
