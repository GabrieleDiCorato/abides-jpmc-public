"""Tests for agent-level circuit breaker in TradingAgent.

Covers:
- _check_circuit_breaker: disabled by default, drawdown trigger, order-rate trigger
- Drawdown skipped when no last_trade data available
- Order rate tumbling window reset on time advance
- Latching behaviour (permanently halted once tripped)
- Integration with create_limit_order, place_market_order, place_multiple_orders
- Proactive trip inside order_executed on fill
- Config system: BaseAgentConfig fields propagate to TradingAgent
"""

from copy import deepcopy

import numpy as np
import pytest

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.orders import LimitOrder, MarketOrder, Side

DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + str_to_ns("06:30:00")
SYMBOL = "TEST"


def _make_agent(
    max_drawdown: int | None = None,
    max_order_rate: int | None = None,
    order_rate_window_ns: int = 60_000_000_000,
    starting_cash: int = 10_000_000,
    position_limit: int | None = None,
) -> TradingAgent:
    agent = TradingAgent(
        id=0,
        random_state=np.random.RandomState(42),
        starting_cash=starting_cash,
        max_drawdown=max_drawdown,
        max_order_rate=max_order_rate,
        order_rate_window_ns=order_rate_window_ns,
        position_limit=position_limit,
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
# _check_circuit_breaker — basics
# ===================================================================


class TestCheckCircuitBreakerBasics:
    def test_disabled_by_default(self):
        agent = _make_agent()
        assert agent._check_circuit_breaker() is False

    def test_not_tripped_when_within_drawdown(self):
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        # Simulate a small loss: bought 100 shares at 10000, now worth 9900.
        agent.holdings[SYMBOL] = 100
        agent.holdings["CASH"] = 10_000_000 - 100 * 10_000
        agent.last_trade[SYMBOL] = 9_900
        # mark_to_market = CASH + 100*9900 = 9_000_000 + 990_000 = 9_990_000
        # loss = 10_000_000 - 9_990_000 = 10_000 (well under 500k)
        assert agent._check_circuit_breaker() is False

    def test_drawdown_trips_breaker(self):
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        # Simulate a big loss: bought 100 shares at 10000, now worth 4000.
        agent.holdings[SYMBOL] = 100
        agent.holdings["CASH"] = 10_000_000 - 100 * 10_000
        agent.last_trade[SYMBOL] = 4_000
        # mark_to_market = 9_000_000 + 100*4000 = 9_400_000
        # loss = 10_000_000 - 9_400_000 = 600_000 >= 500_000
        assert agent._check_circuit_breaker() is True
        assert agent._circuit_breaker_tripped is True

    def test_drawdown_skipped_when_no_last_trade(self):
        agent = _make_agent(max_drawdown=1)
        # No last_trade data — breaker should NOT trip (can't price).
        assert agent._check_circuit_breaker() is False

    def test_order_rate_trips_breaker(self):
        agent = _make_agent(max_order_rate=3)
        # Simulate 3 orders already recorded in the current window.
        agent._window_start = MKT_OPEN
        agent._order_count_in_window = 3
        assert agent._check_circuit_breaker() is True
        assert agent._circuit_breaker_tripped is True

    def test_order_rate_within_limit(self):
        agent = _make_agent(max_order_rate=5)
        agent._window_start = MKT_OPEN
        agent._order_count_in_window = 4
        assert agent._check_circuit_breaker() is False

    def test_order_rate_window_expired_resets(self):
        agent = _make_agent(max_order_rate=3, order_rate_window_ns=1_000_000_000)
        agent._window_start = MKT_OPEN
        agent._order_count_in_window = 10
        # Advance time past the window.
        agent.current_time = MKT_OPEN + 2_000_000_000
        # Window has expired → the stale count doesn't trigger.
        assert agent._check_circuit_breaker() is False


# ===================================================================
# Latching behaviour
# ===================================================================


class TestCircuitBreakerLatching:
    def test_breaker_stays_tripped(self):
        agent = _make_agent(max_drawdown=100)
        # Force the latch.
        agent._circuit_breaker_tripped = True
        # Even with no loss, it should remain tripped.
        assert agent._check_circuit_breaker() is True

    def test_breaker_latches_after_conditions_improve(self):
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        # Trip via drawdown.
        agent.holdings[SYMBOL] = 100
        agent.holdings["CASH"] = 10_000_000 - 100 * 10_000
        agent.last_trade[SYMBOL] = 4_000  # loss = 600k
        assert agent._check_circuit_breaker() is True
        # Now "fix" the situation.
        agent.last_trade[SYMBOL] = 10_000  # back to even
        # Still tripped — latching.
        assert agent._check_circuit_breaker() is True


# ===================================================================
# _record_order_for_rate_check
# ===================================================================


class TestRecordOrderForRateCheck:
    def test_noop_when_disabled(self):
        agent = _make_agent(max_order_rate=None)
        agent._record_order_for_rate_check()
        assert agent._order_count_in_window == 0

    def test_starts_window_on_first_call(self):
        agent = _make_agent(max_order_rate=10)
        agent._record_order_for_rate_check()
        assert agent._window_start == MKT_OPEN
        assert agent._order_count_in_window == 1

    def test_increments_within_window(self):
        agent = _make_agent(max_order_rate=10)
        agent._record_order_for_rate_check()
        agent._record_order_for_rate_check()
        agent._record_order_for_rate_check()
        assert agent._order_count_in_window == 3

    def test_resets_after_window_expires(self):
        agent = _make_agent(max_order_rate=10, order_rate_window_ns=1_000_000_000)
        agent._record_order_for_rate_check()
        agent._record_order_for_rate_check()
        assert agent._order_count_in_window == 2
        # Advance time past the window.
        agent.current_time = MKT_OPEN + 2_000_000_000
        agent._record_order_for_rate_check()
        assert agent._order_count_in_window == 1
        assert agent._window_start == MKT_OPEN + 2_000_000_000


# ===================================================================
# Logging
# ===================================================================


class TestCircuitBreakerLogging:
    def test_drawdown_logs_event(self):
        logged = []
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        agent.logEvent = lambda event, data, **kw: logged.append((event, data))
        agent.holdings[SYMBOL] = 100
        agent.holdings["CASH"] = 10_000_000 - 100 * 10_000
        agent.last_trade[SYMBOL] = 4_000  # loss = 600k
        agent._check_circuit_breaker()
        assert any(
            ev == "CIRCUIT_BREAKER_TRIPPED" and d["reason"] == "max_drawdown"
            for ev, d in logged
        )

    def test_order_rate_logs_event(self):
        logged = []
        agent = _make_agent(max_order_rate=2)
        agent.logEvent = lambda event, data, **kw: logged.append((event, data))
        agent._window_start = MKT_OPEN
        agent._order_count_in_window = 2
        agent._check_circuit_breaker()
        assert any(
            ev == "CIRCUIT_BREAKER_TRIPPED" and d["reason"] == "max_order_rate"
            for ev, d in logged
        )


# ===================================================================
# create_limit_order integration
# ===================================================================


class TestCreateLimitOrderCircuitBreaker:
    def test_no_breaker_creates_order(self):
        agent = _make_agent()
        order = agent.create_limit_order(SYMBOL, 50, Side.BID, 10_000)
        assert order is not None
        assert order.quantity == 50

    def test_tripped_returns_none(self):
        agent = _make_agent()
        agent._circuit_breaker_tripped = True
        result = agent.create_limit_order(SYMBOL, 50, Side.BID, 10_000)
        assert result is None

    def test_drawdown_blocks_order(self):
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        agent.holdings[SYMBOL] = 100
        agent.holdings["CASH"] = 10_000_000 - 100 * 10_000
        agent.last_trade[SYMBOL] = 4_000  # loss = 600k
        result = agent.create_limit_order(SYMBOL, 10, Side.BID, 10_000)
        assert result is None


# ===================================================================
# place_market_order integration
# ===================================================================


class TestPlaceMarketOrderCircuitBreaker:
    def test_tripped_blocks_market_order(self):
        sent = []
        agent = _make_agent()
        agent.send_message = lambda dest, msg: sent.append(msg)
        agent._circuit_breaker_tripped = True
        agent.place_market_order(SYMBOL, 50, Side.BID)
        assert len(sent) == 0

    def test_no_breaker_sends_market_order(self):
        sent = []
        agent = _make_agent()
        agent.send_message = lambda dest, msg: sent.append(msg)
        agent.place_market_order(SYMBOL, 50, Side.BID)
        assert len(sent) == 1


# ===================================================================
# place_multiple_orders integration
# ===================================================================


class TestPlaceMultipleOrdersCircuitBreaker:
    def test_tripped_blocks_batch(self):
        sent_batches = []
        agent = _make_agent()
        agent.send_message_batch = lambda dest, msgs: sent_batches.append(msgs)
        agent._circuit_breaker_tripped = True
        orders = [
            LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_000),
            LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_100),
        ]
        agent.place_multiple_orders(orders)
        assert len(sent_batches) == 0

    def test_no_breaker_sends_batch(self):
        sent_batches = []
        agent = _make_agent()
        agent.send_message_batch = lambda dest, msgs: sent_batches.append(msgs)
        orders = [
            LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_000),
            LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_100),
        ]
        agent.place_multiple_orders(orders)
        assert len(sent_batches) == 1
        assert len(sent_batches[0]) == 2


# ===================================================================
# Order rate via place_limit_order
# ===================================================================


class TestOrderRateViaPlacement:
    def test_rate_limit_trips_after_n_orders(self):
        sent = []
        agent = _make_agent(max_order_rate=3)
        agent.send_message = lambda dest, msg: sent.append(msg)
        # Place 3 orders — all should succeed.
        for i in range(3):
            agent.place_limit_order(SYMBOL, 10, Side.BID, 10_000)
        assert len(sent) == 3
        # 4th order should be blocked (breaker trips at check time).
        agent.place_limit_order(SYMBOL, 10, Side.BID, 10_000)
        assert len(sent) == 3
        assert agent._circuit_breaker_tripped is True

    def test_rate_limit_window_reset_allows_more(self):
        """After the window expires, the counter resets but the breaker should NOT
        have been tripped yet (it trips only when check finds count >= limit)."""
        sent = []
        agent = _make_agent(max_order_rate=3, order_rate_window_ns=1_000_000_000)
        agent.send_message = lambda dest, msg: sent.append(msg)
        # Place 2 orders.
        agent.place_limit_order(SYMBOL, 10, Side.BID, 10_000)
        agent.place_limit_order(SYMBOL, 10, Side.BID, 10_000)
        assert len(sent) == 2
        # Advance time past the window.
        agent.current_time = MKT_OPEN + 2_000_000_000
        # Place 2 more orders — should succeed (new window).
        agent.place_limit_order(SYMBOL, 10, Side.BID, 10_000)
        agent.place_limit_order(SYMBOL, 10, Side.BID, 10_000)
        assert len(sent) == 4
        assert agent._circuit_breaker_tripped is False


# ===================================================================
# Breaker trips inside order_executed
# ===================================================================


class TestCircuitBreakerTripsOnFill:
    def test_fill_triggers_drawdown_trip(self):
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        agent.last_trade[SYMBOL] = 4_000
        # Simulate a fill: buy 100 shares at 10_000 each.
        fill = LimitOrder(0, MKT_OPEN, SYMBOL, 100, Side.BID, 10_000)
        fill.fill_price = 10_000
        agent.orders[fill.order_id] = deepcopy(fill)
        agent.order_executed(fill)
        # After execution: holdings[SYMBOL]=100, CASH = 10M - 100*10000 = 9M
        # mark_to_market = 9M + 100*4000 = 9_400_000
        # loss = 10M - 9_400_000 = 600k >= 500k
        assert agent._circuit_breaker_tripped is True

    def test_fill_no_trip_when_within_threshold(self):
        agent = _make_agent(max_drawdown=500_000, starting_cash=10_000_000)
        agent.last_trade[SYMBOL] = 9_900
        fill = LimitOrder(0, MKT_OPEN, SYMBOL, 10, Side.BID, 10_000)
        fill.fill_price = 10_000
        agent.orders[fill.order_id] = deepcopy(fill)
        agent.order_executed(fill)
        # loss = 10M - (10M - 10*10000 + 10*9900) = 10M - 9_999_000 = 1_000
        assert agent._circuit_breaker_tripped is False


# ===================================================================
# Config system integration
# ===================================================================


class TestConfigSystemCircuitBreaker:
    def test_base_agent_config_has_fields(self):
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        config = BaseAgentConfig(
            max_drawdown=500_000,
            max_order_rate=100,
            order_rate_window_ns=30_000_000_000,
        )
        assert config.max_drawdown == 500_000
        assert config.max_order_rate == 100
        assert config.order_rate_window_ns == 30_000_000_000

    def test_base_agent_config_defaults(self):
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        config = BaseAgentConfig()
        assert config.max_drawdown is None
        assert config.max_order_rate is None
        assert config.order_rate_window_ns == 60_000_000_000
