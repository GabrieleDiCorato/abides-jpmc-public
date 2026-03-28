"""Tests for RiskConfig and its wiring through agents and the config system.

Covers:
- RiskConfig dataclass construction (frozen, defaults, custom values)
- TradingAgent accepts RiskConfig and unpacks into its fields
- RiskConfig takes precedence over flat params
- Per-fill P&L tracking (FILL_PNL events, _peak_nav)
- last_trade seeded from fill only when absent
- Concrete agents forward risk_config to TradingAgent
- Config system builds RiskConfig and injects it into all agent types
"""

from copy import deepcopy

import numpy as np
import pytest

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.models.risk_config import RiskConfig
from abides_markets.orders import LimitOrder, Side

DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + str_to_ns("06:30:00")
SYMBOL = "TEST"


def _make_agent(
    risk_config: RiskConfig | None = None,
    starting_cash: int = 10_000_000,
    **flat_params,
) -> TradingAgent:
    agent = TradingAgent(
        id=0,
        random_state=np.random.RandomState(42),
        starting_cash=starting_cash,
        risk_config=risk_config,
        **flat_params,
    )
    agent.exchange_id = 99
    agent.current_time = MKT_OPEN
    agent.mkt_open = MKT_OPEN
    agent.mkt_close = MKT_CLOSE
    agent.send_message = lambda *a, **kw: None  # type: ignore[method-assign]
    agent.send_message_batch = lambda *a, **kw: None  # type: ignore[method-assign]
    agent.logEvent = lambda *a, **kw: None  # type: ignore[method-assign]
    return agent


# ===================================================================
# RiskConfig dataclass
# ===================================================================


class TestRiskConfigDataclass:
    def test_defaults(self):
        rc = RiskConfig()
        assert rc.position_limit is None
        assert rc.position_limit_clamp is False
        assert rc.max_drawdown is None
        assert rc.max_order_rate is None
        assert rc.order_rate_window_ns == 60_000_000_000

    def test_custom_values(self):
        rc = RiskConfig(
            position_limit=500,
            position_limit_clamp=True,
            max_drawdown=1_000_000,
            max_order_rate=10,
            order_rate_window_ns=30_000_000_000,
        )
        assert rc.position_limit == 500
        assert rc.position_limit_clamp is True
        assert rc.max_drawdown == 1_000_000
        assert rc.max_order_rate == 10
        assert rc.order_rate_window_ns == 30_000_000_000

    def test_frozen(self):
        rc = RiskConfig(position_limit=100)
        with pytest.raises(AttributeError):
            rc.position_limit = 200  # type: ignore[misc]


# ===================================================================
# TradingAgent accepts RiskConfig
# ===================================================================


class TestTradingAgentRiskConfig:
    def test_risk_config_unpacks_into_fields(self):
        rc = RiskConfig(
            position_limit=500,
            position_limit_clamp=True,
            max_drawdown=1_000_000,
            max_order_rate=10,
            order_rate_window_ns=30_000_000_000,
        )
        agent = _make_agent(risk_config=rc)
        assert agent.position_limit == 500
        assert agent.position_limit_clamp is True
        assert agent.max_drawdown == 1_000_000
        assert agent.max_order_rate == 10
        assert agent.order_rate_window_ns == 30_000_000_000

    def test_risk_config_overrides_flat_params(self):
        rc = RiskConfig(position_limit=500, max_drawdown=1_000_000)
        agent = _make_agent(
            risk_config=rc,
            position_limit=999,
            max_drawdown=999_999,
        )
        # RiskConfig wins over flat params
        assert agent.position_limit == 500
        assert agent.max_drawdown == 1_000_000

    def test_flat_params_work_without_risk_config(self):
        agent = _make_agent(position_limit=300, max_drawdown=500_000)
        assert agent.position_limit == 300
        assert agent.max_drawdown == 500_000

    def test_defaults_when_neither_provided(self):
        agent = _make_agent()
        assert agent.position_limit is None
        assert agent.position_limit_clamp is False
        assert agent.max_drawdown is None
        assert agent.max_order_rate is None
        assert agent.order_rate_window_ns == 60_000_000_000

    def test_risk_config_enables_circuit_breaker(self):
        rc = RiskConfig(max_drawdown=500_000)
        agent = _make_agent(risk_config=rc, starting_cash=10_000_000)
        agent.last_trade[SYMBOL] = 4_000
        fill = LimitOrder(0, MKT_OPEN, SYMBOL, 100, Side.BID, 10_000)
        fill.fill_price = 10_000
        agent.orders[fill.order_id] = deepcopy(fill)
        agent.order_executed(fill)
        # mark_to_market = 9M + 100*4000 = 9.4M → loss=600k > 500k
        assert agent._circuit_breaker_tripped is True

    def test_risk_config_enables_position_limit(self):
        rc = RiskConfig(position_limit=50)
        agent = _make_agent(risk_config=rc)
        # Try to place order for 100 shares; limit is 50
        sent: list = []
        agent.send_message = lambda dest, msg: sent.append(msg)
        agent.place_limit_order(SYMBOL, 100, Side.BID, 10_000)
        # Order should be blocked (no message sent)
        assert len(sent) == 0


# ===================================================================
# Per-fill P&L tracking
# ===================================================================


class TestFillPnlTracking:
    def test_peak_nav_initialised_to_starting_cash(self):
        agent = _make_agent(starting_cash=5_000_000)
        assert agent._peak_nav == 5_000_000

    def test_fill_pnl_event_logged(self):
        events: list = []
        agent = _make_agent()
        agent.logEvent = lambda name, data, **kw: events.append((name, data))
        agent.last_trade[SYMBOL] = 10_000
        fill = LimitOrder(0, MKT_OPEN, SYMBOL, 10, Side.BID, 10_000)
        fill.fill_price = 10_000
        agent.orders[fill.order_id] = deepcopy(fill)
        agent.order_executed(fill)
        pnl_events = [(n, d) for n, d in events if n == "FILL_PNL"]
        assert len(pnl_events) == 1
        _, data = pnl_events[0]
        assert "nav" in data
        assert "peak_nav" in data
        assert "symbol" in data
        assert data["symbol"] == SYMBOL

    def test_peak_nav_tracks_high_water_mark(self):
        agent = _make_agent(starting_cash=10_000_000)
        agent.logEvent = lambda *a, **kw: None

        # Fill 1: buy 10 @ 10000; last_trade=10000 → nav=10M (no change)
        agent.last_trade[SYMBOL] = 10_000
        f1 = LimitOrder(0, MKT_OPEN, SYMBOL, 10, Side.BID, 10_000)
        f1.fill_price = 10_000
        agent.orders[f1.order_id] = deepcopy(f1)
        agent.order_executed(f1)
        assert agent._peak_nav == 10_000_000

        # Simulate price increase: last_trade rises to 12000
        agent.last_trade[SYMBOL] = 12_000
        # Fill 2: buy 10 more
        f2 = LimitOrder(1, MKT_OPEN, SYMBOL, 10, Side.BID, 12_000)
        f2.fill_price = 12_000
        agent.orders[f2.order_id] = deepcopy(f2)
        agent.order_executed(f2)
        # holdings: 20 shares, cash = 10M - 100k - 120k = 9_780_000
        # nav = 9_780_000 + 20*12000 = 10_020_000
        assert agent._peak_nav == 10_020_000

    def test_last_trade_seeded_from_fill_when_absent(self):
        agent = _make_agent()
        agent.logEvent = lambda *a, **kw: None
        assert SYMBOL not in agent.last_trade
        fill = LimitOrder(0, MKT_OPEN, SYMBOL, 10, Side.BID, 10_000)
        fill.fill_price = 10_000
        agent.orders[fill.order_id] = deepcopy(fill)
        agent.order_executed(fill)
        assert agent.last_trade[SYMBOL] == 10_000

    def test_last_trade_not_overwritten_when_present(self):
        agent = _make_agent()
        agent.logEvent = lambda *a, **kw: None
        agent.last_trade[SYMBOL] = 9_500
        fill = LimitOrder(0, MKT_OPEN, SYMBOL, 10, Side.BID, 10_000)
        fill.fill_price = 10_000
        agent.orders[fill.order_id] = deepcopy(fill)
        agent.order_executed(fill)
        # Should still be the original market-data price
        assert agent.last_trade[SYMBOL] == 9_500


# ===================================================================
# Concrete agents forward risk_config
# ===================================================================


class TestConcreteAgentsForwardRiskConfig:
    """Verify all five concrete agents accept and forward risk_config."""

    def test_noise_agent(self):
        from abides_markets.agents.noise_agent import NoiseAgent

        rc = RiskConfig(position_limit=100)
        agent = NoiseAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol=SYMBOL,
            starting_cash=10_000_000,
            wakeup_time=MKT_OPEN,
            risk_config=rc,
        )
        assert agent.position_limit == 100

    def test_value_agent(self):
        from abides_markets.agents.value_agent import ValueAgent

        rc = RiskConfig(max_drawdown=500_000)
        agent = ValueAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol=SYMBOL,
            starting_cash=10_000_000,
            r_bar=100_000,
            kappa=1.67e-16,
            sigma_s=1_000_000,
            risk_config=rc,
        )
        assert agent.max_drawdown == 500_000

    def test_momentum_agent(self):
        from abides_markets.agents.examples.momentum_agent import MomentumAgent

        rc = RiskConfig(max_order_rate=5)
        agent = MomentumAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol=SYMBOL,
            starting_cash=10_000_000,
            short_window=5,
            long_window=20,
            risk_config=rc,
        )
        assert agent.max_order_rate == 5

    def test_pov_execution_agent(self):
        from abides_markets.agents.pov_execution_agent import POVExecutionAgent

        rc = RiskConfig(position_limit=1000)
        agent = POVExecutionAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol=SYMBOL,
            starting_cash=10_000_000,
            direction=Side.BID,
            quantity=100,
            pov=0.1,
            start_time=MKT_OPEN,
            end_time=MKT_CLOSE,
            risk_config=rc,
        )
        assert agent.position_limit == 1000

    def test_adaptive_market_maker_agent(self):
        from abides_markets.agents.market_makers.adaptive_market_maker_agent import (
            AdaptiveMarketMakerAgent,
        )

        rc = RiskConfig(max_drawdown=1_000_000, position_limit=200)
        agent = AdaptiveMarketMakerAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol=SYMBOL,
            starting_cash=10_000_000,
            window_size=5,
            num_ticks=10,
            level_spacing=1.0,
            wake_up_freq=1_000_000_000,
            risk_config=rc,
        )
        assert agent.max_drawdown == 1_000_000
        assert agent.position_limit == 200


# ===================================================================
# Config system builds RiskConfig
# ===================================================================


class TestConfigSystemRiskConfig:
    def test_base_agent_config_builds_risk_config(self):
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        config = BaseAgentConfig(
            position_limit=500,
            max_drawdown=1_000_000,
            max_order_rate=10,
            order_rate_window_ns=30_000_000_000,
        )
        context = _stub_context()
        kwargs = config._prepare_constructor_kwargs(
            {}, agent_id=0, agent_rng=np.random.RandomState(42), context=context
        )
        rc = kwargs["risk_config"]
        assert isinstance(rc, RiskConfig)
        assert rc.position_limit == 500
        assert rc.max_drawdown == 1_000_000
        assert rc.max_order_rate == 10
        assert rc.order_rate_window_ns == 30_000_000_000

    def test_risk_fields_excluded_from_auto_kwargs(self):
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        config = BaseAgentConfig()
        excluded = config._EXCLUDE_FROM_KWARGS
        assert "position_limit" in excluded
        assert "position_limit_clamp" in excluded
        assert "max_drawdown" in excluded
        assert "max_order_rate" in excluded
        assert "order_rate_window_ns" in excluded

    def test_noise_config_inherits_risk_config_injection(self):
        from abides_markets.config_system.agent_configs import NoiseAgentConfig

        config = NoiseAgentConfig(position_limit=100)
        context = _stub_context()
        kwargs = config._prepare_constructor_kwargs(
            {}, agent_id=0, agent_rng=np.random.RandomState(42), context=context
        )
        rc = kwargs["risk_config"]
        assert isinstance(rc, RiskConfig)
        assert rc.position_limit == 100


def _stub_context():
    """Minimal AgentCreationContext-like object for testing."""
    from types import SimpleNamespace

    return SimpleNamespace(
        ticker="TEST",
        exchange_id=99,
        mkt_open=MKT_OPEN,
        mkt_close=MKT_CLOSE,
        date_ns=DATE,
        oracle=None,
    )
