"""Tests for market open / close boundary conditions.

Covers:
- TradingAgent.wakeup() gating before MarketHoursMsg, at market open, after close
- ExchangeAgent _cancel_day_orders() behaviour at market close
- ExchangeAgent post-close message rejection (OrderMsg → MarketClosedMsg)
- TradingAgent.mkt_closed flag set from response messages
- DAY vs GTC order survival at close
- ExchangeAgent market close wakeup and MarketClosePriceMsg fan-out
"""

from __future__ import annotations

import numpy as np

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.agents.exchange_agent import ExchangeAgent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.messages.market import (
    MarketClosedMsg,
    MarketClosePriceMsg,
    MarketClosePriceRequestMsg,
    MarketHoursMsg,
    MarketHoursRequestMsg,
)
from abides_markets.messages.order import LimitOrderMsg, MarketOrderMsg
from abides_markets.messages.query import (
    QueryLastTradeMsg,
    QueryLastTradeResponseMsg,
    QuerySpreadResponseMsg,
)
from abides_markets.orders import LimitOrder, MarketOrder, Side, TimeInForce

DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + str_to_ns("06:30:00")
SYMBOL = "TEST"


# ===================================================================
# Helpers
# ===================================================================


def _make_trading_agent(**kwargs) -> TradingAgent:
    defaults = dict(
        id=1,
        random_state=np.random.RandomState(42),
        starting_cash=10_000_000,
    )
    merged = {**defaults, **kwargs}
    agent = TradingAgent(**merged)
    agent.exchange_id = 0
    agent.current_time = MKT_OPEN
    agent.kernel = object()  # type: ignore[assignment]  # stub — Agent.wakeup needs kernel not None
    agent.send_message = lambda *a, **kw: None  # type: ignore[method-assign]
    agent.send_message_batch = lambda *a, **kw: None  # type: ignore[method-assign]
    agent.logEvent = lambda *a, **kw: None  # type: ignore[method-assign]
    return agent


def _make_exchange(**kwargs) -> ExchangeAgent:
    """Create a minimal ExchangeAgent for unit testing."""
    defaults = dict(
        id=0,
        mkt_open=MKT_OPEN,
        mkt_close=MKT_CLOSE,
        symbols=[SYMBOL],
        name="TEST-EXCHANGE",
        random_state=np.random.RandomState(0),
        use_metric_tracker=False,
        log_orders=False,
    )
    merged = {**defaults, **kwargs}
    ex = ExchangeAgent(**merged)
    ex.current_time = MKT_OPEN
    ex.kernel = object()  # type: ignore[assignment]
    # Stub messaging — collect sent messages
    ex._sent: list[tuple[int, object]] = []  # type: ignore[attr-defined, misc]
    ex.send_message = lambda rcpt, msg: ex._sent.append((rcpt, msg))  # type: ignore[method-assign, assignment, attr-defined]
    ex.logEvent = lambda *a, **kw: None  # type: ignore[method-assign]
    ex.set_computation_delay = lambda d: None  # type: ignore[method-assign, assignment]
    ex.set_wakeup = lambda t: None  # type: ignore[method-assign, assignment]
    # Give the book an opening price
    ex.order_books[SYMBOL].last_trade = 10_000
    return ex


def _place_limit(
    exchange: ExchangeAgent,
    agent_id: int,
    side: Side,
    price: int,
    qty: int,
    tif: TimeInForce = TimeInForce.GTC,
) -> LimitOrder:
    """Place a limit order directly into the exchange's order book."""
    order = LimitOrder(
        agent_id=agent_id,
        time_placed=exchange.current_time,
        symbol=SYMBOL,
        quantity=qty,
        side=side,
        limit_price=price,
        time_in_force=tif,
    )
    exchange.order_books[SYMBOL].handle_limit_order(order)
    return order


# ===================================================================
# TradingAgent wakeup gating
# ===================================================================


class TestTradingAgentWakeupGating:
    """Verify wakeup() returns False when market hours unknown or market closed."""

    def test_wakeup_returns_false_before_market_hours_known(self):
        """Before MarketHoursMsg, mkt_open is None → wakeup returns falsy."""
        agent = _make_trading_agent()
        agent.mkt_open = None
        agent.mkt_close = None
        # wakeup should return falsy (None/False) because hours unknown
        result = agent.wakeup(MKT_OPEN)
        assert not result

    def test_wakeup_returns_true_with_hours_set_and_market_open(self):
        """With hours known and market not closed → wakeup returns truthy."""
        agent = _make_trading_agent()
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = MKT_CLOSE
        agent.mkt_closed = False
        agent.first_wake = False  # skip first-wake side effects
        result = agent.wakeup(MKT_OPEN)
        assert result

    def test_wakeup_returns_false_after_market_closed(self):
        """With mkt_closed flag set → wakeup returns falsy."""
        agent = _make_trading_agent()
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = MKT_CLOSE
        agent.mkt_closed = True
        agent.first_wake = False
        result = agent.wakeup(MKT_OPEN + str_to_ns("06:30:01"))
        assert not result

    def test_wakeup_with_only_mkt_open_set(self):
        """If mkt_open known but mkt_close still None → wakeup returns falsy."""
        agent = _make_trading_agent()
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = None
        result = agent.wakeup(MKT_OPEN)
        assert not result


# ===================================================================
# TradingAgent receives MarketHoursMsg
# ===================================================================


class TestMarketHoursReception:
    """Verify TradingAgent records market hours from the exchange response."""

    def test_market_hours_msg_sets_open_close(self):
        agent = _make_trading_agent()
        agent.mkt_open = None
        agent.mkt_close = None
        msg = MarketHoursMsg(mkt_open=MKT_OPEN, mkt_close=MKT_CLOSE)
        agent._handle_market_hours_msg(msg)
        assert agent.mkt_open == MKT_OPEN
        assert agent.mkt_close == MKT_CLOSE

    def test_market_hours_second_update_overwrites(self):
        """If exchange sends hours again, agent updates."""
        agent = _make_trading_agent()
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = MKT_CLOSE
        new_close = MKT_CLOSE + str_to_ns("00:30:00")
        agent._handle_market_hours_msg(MarketHoursMsg(MKT_OPEN, new_close))
        assert agent.mkt_close == new_close


# ===================================================================
# TradingAgent mkt_closed flag from query responses
# ===================================================================


class TestMktClosedFlag:
    """When the exchange sends mkt_closed=True on a query response,
    the agent's mkt_closed flag should be latched."""

    def test_query_last_trade_response_latches_mkt_closed(self):
        agent = _make_trading_agent()
        agent.mkt_closed = False
        msg = QueryLastTradeResponseMsg(
            symbol=SYMBOL, last_trade=10_000, mkt_closed=True
        )
        agent._handle_query_last_trade_response_msg(msg)
        assert agent.mkt_closed is True
        assert agent.last_trade[SYMBOL] == 10_000

    def test_query_spread_response_latches_mkt_closed(self):
        agent = _make_trading_agent()
        agent.mkt_closed = False
        msg = QuerySpreadResponseMsg(
            symbol=SYMBOL, last_trade=10_000, bids=[], asks=[], mkt_closed=True, depth=1
        )
        agent._handle_query_spread_response_msg(msg)
        assert agent.mkt_closed is True

    def test_query_response_mkt_open_does_not_latch(self):
        """mkt_closed=False should not flip the flag if it's already False."""
        agent = _make_trading_agent()
        agent.mkt_closed = False
        msg = QueryLastTradeResponseMsg(
            symbol=SYMBOL, last_trade=10_000, mkt_closed=False
        )
        agent._handle_query_last_trade_response_msg(msg)
        assert agent.mkt_closed is False


# ===================================================================
# ExchangeAgent DAY order cancellation at market close
# ===================================================================


class TestDAYOrderCancellation:
    """_cancel_day_orders() removes DAY orders but leaves GTC orders."""

    def test_day_orders_cancelled_at_close(self):
        ex = _make_exchange()
        _place_limit(ex, 1, Side.BID, 9_900, 100, TimeInForce.DAY)
        _place_limit(ex, 2, Side.ASK, 10_100, 50, TimeInForce.DAY)
        assert len(ex.order_books[SYMBOL].bids) == 1
        assert len(ex.order_books[SYMBOL].asks) == 1

        ex._cancel_day_orders()

        # Both sides should now be empty
        assert len(ex.order_books[SYMBOL].bids) == 0
        assert len(ex.order_books[SYMBOL].asks) == 0

    def test_gtc_orders_survive_close(self):
        ex = _make_exchange()
        gtc_bid = _place_limit(ex, 1, Side.BID, 9_900, 100, TimeInForce.GTC)
        _place_limit(ex, 2, Side.ASK, 10_100, 50, TimeInForce.DAY)

        ex._cancel_day_orders()

        # GTC bid survives, DAY ask is cancelled
        assert len(ex.order_books[SYMBOL].bids) == 1
        assert len(ex.order_books[SYMBOL].asks) == 0
        remaining = ex.order_books[SYMBOL].bids[0].visible_orders[0][0]
        assert remaining.order_id == gtc_bid.order_id

    def test_mixed_day_and_gtc_same_price_level(self):
        """DAY and GTC orders at the same price — only DAY is removed."""
        ex = _make_exchange()
        gtc = _place_limit(ex, 1, Side.BID, 9_900, 100, TimeInForce.GTC)
        _place_limit(ex, 2, Side.BID, 9_900, 50, TimeInForce.DAY)

        assert len(ex.order_books[SYMBOL].bids) == 1  # one price level
        ex._cancel_day_orders()

        # GTC should survive at that price level
        assert len(ex.order_books[SYMBOL].bids) == 1
        remaining = ex.order_books[SYMBOL].bids[0].visible_orders[0][0]
        assert remaining.order_id == gtc.order_id
        assert remaining.quantity == 100

    def test_empty_book_cancel_day_orders_is_noop(self):
        """Calling _cancel_day_orders on empty book should not raise."""
        ex = _make_exchange()
        ex._cancel_day_orders()  # should be a no-op
        assert len(ex.order_books[SYMBOL].bids) == 0
        assert len(ex.order_books[SYMBOL].asks) == 0

    def test_day_orders_multiple_price_levels(self):
        """DAY orders across several price levels are all removed."""
        ex = _make_exchange()
        _place_limit(ex, 1, Side.BID, 9_900, 100, TimeInForce.DAY)
        _place_limit(ex, 2, Side.BID, 9_800, 200, TimeInForce.DAY)
        _place_limit(ex, 3, Side.BID, 9_700, 50, TimeInForce.DAY)
        assert len(ex.order_books[SYMBOL].bids) == 3

        ex._cancel_day_orders()
        assert len(ex.order_books[SYMBOL].bids) == 0


# ===================================================================
# ExchangeAgent post-close message rejection
# ===================================================================


class TestPostCloseRejection:
    """After mkt_close, ExchangeAgent rejects OrderMsgs with MarketClosedMsg."""

    def test_limit_order_after_close_returns_market_closed(self):
        ex = _make_exchange()
        ex.current_time = MKT_CLOSE + 1  # 1ns past close

        order = LimitOrder(
            agent_id=1,
            time_placed=ex.current_time,
            symbol=SYMBOL,
            quantity=100,
            side=Side.BID,
            limit_price=9_900,
        )
        ex.receive_message(ex.current_time, 1, LimitOrderMsg(order))

        # Should have sent MarketClosedMsg
        assert any(isinstance(msg, MarketClosedMsg) for _, msg in ex._sent)

    def test_market_order_after_close_returns_market_closed(self):
        ex = _make_exchange()
        ex.current_time = MKT_CLOSE + 1

        order = MarketOrder(
            agent_id=1,
            time_placed=ex.current_time,
            symbol=SYMBOL,
            quantity=100,
            side=Side.BID,
        )
        ex.receive_message(ex.current_time, 1, MarketOrderMsg(order))

        assert any(isinstance(msg, MarketClosedMsg) for _, msg in ex._sent)

    def test_query_after_close_still_works(self):
        """QueryMsg after close should NOT be rejected — agents can query
        the final trade price after close."""
        ex = _make_exchange()
        ex.current_time = MKT_CLOSE + 1

        ex.receive_message(ex.current_time, 1, QueryLastTradeMsg(symbol=SYMBOL))

        # Should get a QueryLastTradeResponseMsg, not MarketClosedMsg
        assert any(isinstance(msg, QueryLastTradeResponseMsg) for _, msg in ex._sent)
        assert not any(isinstance(msg, MarketClosedMsg) for _, msg in ex._sent)


# ===================================================================
# ExchangeAgent wakeup at mkt_close
# ===================================================================


class TestExchangeCloseWakeup:
    """When ExchangeAgent wakes up at mkt_close, it should cancel DAY orders
    and fan out MarketClosePriceMsg."""

    def test_wakeup_at_close_fans_out_close_prices(self):
        ex = _make_exchange()
        # Register 2 agents for close price notifications
        ex.market_close_price_subscriptions = [1, 2]
        ex.current_time = MKT_CLOSE

        # Need to call wakeup through the chain, but super().wakeup requires kernel
        # Simulate directly: the wakeup logic checks current_time >= mkt_close
        ex.wakeup(MKT_CLOSE)

        # Find MarketClosePriceMsg messages
        close_msgs = [
            (rcpt, msg)
            for rcpt, msg in ex._sent
            if isinstance(msg, MarketClosePriceMsg)
        ]
        assert len(close_msgs) == 2
        recipients = {rcpt for rcpt, _ in close_msgs}
        assert recipients == {1, 2}

    def test_close_price_contains_last_trade(self):
        ex = _make_exchange()
        ex.order_books[SYMBOL].last_trade = 12_345
        ex.market_close_price_subscriptions = [1]
        ex.current_time = MKT_CLOSE

        ex.wakeup(MKT_CLOSE)

        close_msgs = [
            msg for _, msg in ex._sent if isinstance(msg, MarketClosePriceMsg)
        ]
        assert len(close_msgs) == 1
        assert close_msgs[0].close_prices[SYMBOL] == 12_345

    def test_no_subscribers_no_messages(self):
        ex = _make_exchange()
        ex.market_close_price_subscriptions = []
        ex.current_time = MKT_CLOSE

        ex.wakeup(MKT_CLOSE)

        close_msgs = [
            msg for _, msg in ex._sent if isinstance(msg, MarketClosePriceMsg)
        ]
        assert len(close_msgs) == 0


# ===================================================================
# Market hours request handling
# ===================================================================


class TestMarketHoursRequestHandling:
    """ExchangeAgent responds to MarketHoursRequestMsg with correct hours."""

    def test_exchange_responds_with_hours(self):
        ex = _make_exchange()
        ex.receive_message(MKT_OPEN, 1, MarketHoursRequestMsg())

        hours_msgs = [msg for _, msg in ex._sent if isinstance(msg, MarketHoursMsg)]
        assert len(hours_msgs) == 1
        assert hours_msgs[0].mkt_open == MKT_OPEN
        assert hours_msgs[0].mkt_close == MKT_CLOSE

    def test_market_close_price_request_subscription(self):
        ex = _make_exchange()
        assert 1 not in ex.market_close_price_subscriptions

        ex.receive_message(MKT_OPEN, 1, MarketClosePriceRequestMsg())

        assert 1 in ex.market_close_price_subscriptions


# ===================================================================
# kernel_stopping — TradingAgent end-of-day reporting
# ===================================================================


class TestTradingAgentKernelStopping:
    """Verify kernel_stopping() computes final holdings and mark-to-market."""

    def test_kernel_stopping_logs_final_cash(self):
        """kernel_stopping should compute ending cash from mark_to_market."""
        agent = _make_trading_agent()
        agent.holdings["CASH"] = 5_000_000
        agent.holdings[SYMBOL] = 100
        agent.last_trade[SYMBOL] = 10_000  # $100/share → 100 * 10000 = 1_000_000

        # Need kernel mock with mean_result_by_agent_type and agent_count_by_type
        from collections import defaultdict

        class FakeKernel:
            mean_result_by_agent_type = defaultdict(float)
            agent_count_by_type = defaultdict(int)

        agent.kernel = FakeKernel()
        agent.type = "TestAgent"
        agent.starting_cash = 5_000_000

        events: list[tuple] = []
        agent.logEvent = lambda *a, **kw: events.append(a)

        agent.kernel_stopping()

        # Should have logged ENDING_CASH = 5_000_000 + 100*10_000 = 6_000_000
        ending_cash_events = [(e[0], e[1]) for e in events if e[0] == "ENDING_CASH"]
        assert len(ending_cash_events) == 1
        assert ending_cash_events[0][1] == 6_000_000

        # The gain should be recorded
        gain = 6_000_000 - 5_000_000
        assert FakeKernel.mean_result_by_agent_type["TestAgent"] == gain
        assert FakeKernel.agent_count_by_type["TestAgent"] == 1

    def test_kernel_stopping_short_position(self):
        """Short position should subtract from cash in mark-to-market."""
        agent = _make_trading_agent()
        agent.holdings["CASH"] = 10_000_000
        agent.holdings[SYMBOL] = -50
        agent.last_trade[SYMBOL] = 10_000

        from collections import defaultdict

        class FakeKernel:
            mean_result_by_agent_type = defaultdict(float)
            agent_count_by_type = defaultdict(int)

        agent.kernel = FakeKernel()
        agent.type = "TestAgent"
        agent.starting_cash = 10_000_000

        events: list[tuple] = []
        agent.logEvent = lambda *a, **kw: events.append(a)

        agent.kernel_stopping()

        ending_cash = [e[1] for e in events if e[0] == "ENDING_CASH"][0]
        expected = 10_000_000 + (-50 * 10_000)
        assert ending_cash == expected


# ===================================================================
# TradingAgent.market_closed() callback
# ===================================================================


class TestMarketClosedCallback:
    """TradingAgent.market_closed() should log MKT_CLOSED and set mkt_closed."""

    def test_market_closed_sets_flag_via_stop_triggered(self):
        """stop_triggered() has a side effect of setting mkt_closed=True.
        This is how the exchange signals market close to agents who have stops."""
        from abides_markets.orders import StopOrder

        agent = _make_trading_agent()
        agent.mkt_closed = False
        stop = StopOrder(
            agent_id=1,
            time_placed=MKT_CLOSE,
            symbol=SYMBOL,
            quantity=100,
            side=Side.BID,
            stop_price=10_000,
        )
        agent.orders[stop.order_id] = stop
        agent.stop_triggered(stop)
        assert agent.mkt_closed is True
        # Order should be removed
        assert stop.order_id not in agent.orders


# ===================================================================
# Boundary: order placed exactly at mkt_close (edge of time)
# ===================================================================


class TestOrderAtExactClose:
    """Orders arriving exactly at mkt_close vs. mkt_close + 1."""

    def test_order_at_exactly_mkt_close_is_accepted(self):
        """At current_time == mkt_close, the exchange close wakeup fires,
        but incoming orders at exactly mkt_close should still go through
        (the check is current_time > mkt_close, not >=)."""
        ex = _make_exchange()
        ex.current_time = MKT_CLOSE

        order = LimitOrder(
            agent_id=1,
            time_placed=MKT_CLOSE,
            symbol=SYMBOL,
            quantity=100,
            side=Side.BID,
            limit_price=9_900,
        )
        ex.receive_message(MKT_CLOSE, 1, LimitOrderMsg(order))

        # Should NOT get MarketClosedMsg — order is accepted
        closed_msgs = [msg for _, msg in ex._sent if isinstance(msg, MarketClosedMsg)]
        assert len(closed_msgs) == 0

    def test_order_at_mkt_close_plus_one_is_rejected(self):
        """1 nanosecond after close → MarketClosedMsg."""
        ex = _make_exchange()
        t = MKT_CLOSE + 1
        ex.current_time = t

        order = LimitOrder(
            agent_id=1,
            time_placed=t,
            symbol=SYMBOL,
            quantity=100,
            side=Side.BID,
            limit_price=9_900,
        )
        ex.receive_message(t, 1, LimitOrderMsg(order))

        closed_msgs = [msg for _, msg in ex._sent if isinstance(msg, MarketClosedMsg)]
        assert len(closed_msgs) == 1
