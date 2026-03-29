"""Tests for AdaptiveMarketMakerAgent end-of-day flatten logic (P1 item 8).

Covers:
- Flatten triggered when wakeup >= mkt_close - flatten_before_close_ns
- Cancel all orders + market order to zero out long/short positions
- _flattened flag prevents multiple flatten attempts
- Disabled when flatten_before_close_ns is None
- Config field mapping (flatten_before_close -> flatten_before_close_ns)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from abides_core.utils import str_to_ns

MKT_OPEN: int = str_to_ns("09:30:00")
MKT_CLOSE: int = str_to_ns("16:00:00")
FIVE_MIN_NS: int = str_to_ns("5min")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_amm(flatten_ns: int | None = FIVE_MIN_NS, **kw):
    from abides_markets.agents import AdaptiveMarketMakerAgent

    agent = AdaptiveMarketMakerAgent(
        id=0,
        symbol="TEST",
        starting_cash=10_000_000,
        random_state=np.random.RandomState(42),
        flatten_before_close_ns=flatten_ns,
        **kw,
    )
    # Simulate that market hours are known (as if MarketHoursMsg was received).
    agent.mkt_open = MKT_OPEN
    agent.mkt_close = MKT_CLOSE
    return agent


# ---------------------------------------------------------------------------
# Unit tests — _flatten_position
# ---------------------------------------------------------------------------
class TestFlattenPosition:
    """Direct tests for _flatten_position logic."""

    def test_flatten_long_position(self):
        """Long position triggers a SELL market order."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": 50}
        agent.exchange_id = 1
        with (
            patch.object(agent, "cancel_all_orders") as mock_cancel,
            patch.object(agent, "place_market_order") as mock_mkt,
            patch.object(agent, "logEvent"),
        ):
            agent._flatten_position(MKT_CLOSE - FIVE_MIN_NS)

        mock_cancel.assert_called_once()
        from abides_markets.orders import Side

        mock_mkt.assert_called_once_with("TEST", 50, Side.ASK)
        assert agent._flattened is True

    def test_flatten_short_position(self):
        """Short position triggers a BUY market order."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": -30}
        agent.exchange_id = 1
        with (
            patch.object(agent, "cancel_all_orders") as mock_cancel,
            patch.object(agent, "place_market_order") as mock_mkt,
            patch.object(agent, "logEvent"),
        ):
            agent._flatten_position(MKT_CLOSE - FIVE_MIN_NS)

        mock_cancel.assert_called_once()
        from abides_markets.orders import Side

        mock_mkt.assert_called_once_with("TEST", 30, Side.BID)
        assert agent._flattened is True

    def test_flatten_zero_position(self):
        """Zero position just cancels orders — no market order placed."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": 0}
        agent.exchange_id = 1
        with (
            patch.object(agent, "cancel_all_orders") as mock_cancel,
            patch.object(agent, "place_market_order") as mock_mkt,
            patch.object(agent, "logEvent"),
        ):
            agent._flatten_position(MKT_CLOSE - FIVE_MIN_NS)

        mock_cancel.assert_called_once()
        mock_mkt.assert_not_called()
        assert agent._flattened is True

    def test_flatten_no_holdings_entry(self):
        """Symbol not in holdings → zero position, no market order."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000}
        agent.exchange_id = 1
        with (
            patch.object(agent, "cancel_all_orders") as mock_cancel,
            patch.object(agent, "place_market_order") as mock_mkt,
            patch.object(agent, "logEvent"),
        ):
            agent._flatten_position(MKT_CLOSE - FIVE_MIN_NS)

        mock_cancel.assert_called_once()
        mock_mkt.assert_not_called()
        assert agent._flattened is True


# ---------------------------------------------------------------------------
# Wakeup integration tests
# ---------------------------------------------------------------------------
class TestFlattenWakeupTrigger:
    """Verify flatten triggers at the right time during wakeup."""

    def _wakeup_with_time(self, agent, current_time):
        """Call wakeup with mocked super().wakeup() returning True."""
        # Skip subscription and polling steps that need a real kernel.
        agent.has_subscribed = True
        agent.subscription_requested = True
        with patch(
            "abides_markets.agents.market_makers.adaptive_market_maker_agent.TradingAgent.wakeup",
            return_value=True,
        ):
            with (
                patch.object(agent, "cancel_all_orders"),
                patch.object(agent, "place_market_order"),
                patch.object(agent, "logEvent"),
                patch.object(agent, "get_current_spread"),
                patch.object(agent, "get_transacted_volume"),
            ):
                agent.wakeup(current_time)

    def test_flatten_triggered_at_boundary(self):
        """Flatten fires exactly at mkt_close - flatten_before_close_ns."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": 10}
        flatten_time = MKT_CLOSE - FIVE_MIN_NS
        self._wakeup_with_time(agent, flatten_time)
        assert agent._flattened is True

    def test_flatten_triggered_after_boundary(self):
        """Flatten fires when past the flatten window."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": 10}
        self._wakeup_with_time(agent, MKT_CLOSE - str_to_ns("1min"))
        assert agent._flattened is True

    def test_no_flatten_before_boundary(self):
        """No flatten when still before the flatten window."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": 10}
        self._wakeup_with_time(agent, MKT_CLOSE - str_to_ns("10min"))
        assert agent._flattened is False

    def test_flatten_only_once(self):
        """Once flattened, subsequent wakeups do not re-flatten."""
        agent = _make_amm()
        agent.holdings = {"CASH": 10_000_000, "TEST": 10}
        agent._flattened = True  # already flattened
        agent.has_subscribed = True
        agent.subscription_requested = True
        with patch(
            "abides_markets.agents.market_makers.adaptive_market_maker_agent.TradingAgent.wakeup",
            return_value=True,
        ):
            with (
                patch.object(agent, "cancel_all_orders") as mock_cancel,
                patch.object(agent, "place_market_order") as mock_mkt,
                patch.object(agent, "logEvent"),
                patch.object(agent, "get_current_spread"),
                patch.object(agent, "get_transacted_volume"),
            ):
                agent.wakeup(MKT_CLOSE - str_to_ns("1min"))

        # Should not have called flatten methods again
        mock_cancel.assert_not_called()
        mock_mkt.assert_not_called()

    def test_flatten_disabled_when_none(self):
        """No flatten when flatten_before_close_ns is None."""
        agent = _make_amm(flatten_ns=None)
        agent.holdings = {"CASH": 10_000_000, "TEST": 100}
        self._wakeup_with_time(agent, MKT_CLOSE - str_to_ns("1min"))
        assert agent._flattened is False


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestAMMFlattenConfig:
    """Config field mapping for flatten_before_close."""

    def test_config_default(self):
        from abides_markets.config_system.agent_configs import (
            AdaptiveMarketMakerConfig,
        )

        cfg = AdaptiveMarketMakerConfig()
        assert cfg.flatten_before_close == "5min"

    def test_config_none_disables(self):
        from abides_markets.config_system.agent_configs import (
            AdaptiveMarketMakerConfig,
        )

        cfg = AdaptiveMarketMakerConfig(flatten_before_close=None)
        assert cfg.flatten_before_close is None

    def test_config_custom_value(self):
        from abides_markets.config_system.agent_configs import (
            AdaptiveMarketMakerConfig,
        )

        cfg = AdaptiveMarketMakerConfig(flatten_before_close="10min")
        assert cfg.flatten_before_close == "10min"

    def test_config_kwargs_mapping(self):
        """Verify flatten_before_close is converted to flatten_before_close_ns."""
        from abides_markets.config_system.agent_configs import (
            AdaptiveMarketMakerConfig,
            AgentCreationContext,
        )

        cfg = AdaptiveMarketMakerConfig(flatten_before_close="10min")
        context = AgentCreationContext(
            ticker="TEST",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            log_orders=False,
            oracle_r_bar=None,
        )
        kwargs: dict = {}
        kwargs = cfg._prepare_constructor_kwargs(
            kwargs, agent_id=0, agent_rng=np.random.RandomState(42), context=context
        )
        assert kwargs["flatten_before_close_ns"] == str_to_ns("10min")

    def test_config_kwargs_mapping_none(self):
        """Verify flatten_before_close=None passes None to constructor."""
        from abides_markets.config_system.agent_configs import (
            AdaptiveMarketMakerConfig,
            AgentCreationContext,
        )

        cfg = AdaptiveMarketMakerConfig(flatten_before_close=None)
        context = AgentCreationContext(
            ticker="TEST",
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            log_orders=False,
            oracle_r_bar=None,
        )
        kwargs: dict = {}
        kwargs = cfg._prepare_constructor_kwargs(
            kwargs, agent_id=0, agent_rng=np.random.RandomState(42), context=context
        )
        assert kwargs["flatten_before_close_ns"] is None
