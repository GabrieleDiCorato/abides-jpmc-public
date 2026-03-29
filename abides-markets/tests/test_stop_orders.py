"""Tests for stop order support (P1 item 7).

Covers:
- StopOrder construction and deepcopy
- StopOrderMsg / StopTriggeredMsg message types
- Exchange-side stop order storage and trigger logic
- TradingAgent.place_stop_order and stop_triggered callback
- Buy stop triggers on last_trade >= stop_price
- Sell stop triggers on last_trade <= stop_price
- Untriggered stops survive trade events
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from abides_core.utils import str_to_ns

from abides_markets.orders import MarketOrder, Side, StopOrder

MKT_OPEN: int = str_to_ns("09:30:00")
MKT_CLOSE: int = str_to_ns("16:00:00")


# ---------------------------------------------------------------------------
# StopOrder unit tests
# ---------------------------------------------------------------------------
class TestStopOrder:
    """Test StopOrder model."""

    def test_construction(self):
        order = StopOrder(
            agent_id=1,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=100,
            side=Side.BID,
            stop_price=10_000,
        )
        assert order.agent_id == 1
        assert order.symbol == "TEST"
        assert order.quantity == 100
        assert order.side == Side.BID
        assert order.stop_price == 10_000
        assert order.fill_price is None

    def test_deepcopy_preserves_fields(self):
        order = StopOrder(
            agent_id=2,
            time_placed=MKT_OPEN,
            symbol="ABC",
            quantity=50,
            side=Side.ASK,
            stop_price=5_000,
            tag="test-tag",
        )
        copy = deepcopy(order)
        assert copy.agent_id == order.agent_id
        assert copy.symbol == order.symbol
        assert copy.quantity == order.quantity
        assert copy.side == order.side
        assert copy.stop_price == order.stop_price
        assert copy.tag == order.tag
        assert copy.order_id == order.order_id
        assert copy is not order

    def test_str_representation(self):
        order = StopOrder(
            agent_id=1,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=10,
            side=Side.BID,
            stop_price=10_000,
        )
        s = str(order)
        assert "STOP" in s
        assert "BID" in s
        assert "TEST" in s


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------
class TestStopMessages:
    """Test stop-related messages."""

    def test_stop_order_msg(self):
        from abides_markets.messages.order import StopOrderMsg

        order = StopOrder(
            agent_id=1,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=100,
            side=Side.BID,
            stop_price=10_000,
        )
        msg = StopOrderMsg(order=order)
        assert msg.order is order

    def test_stop_triggered_msg(self):
        from abides_markets.messages.orderbook import StopTriggeredMsg

        order = StopOrder(
            agent_id=1,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=100,
            side=Side.ASK,
            stop_price=9_000,
        )
        msg = StopTriggeredMsg(order=order)
        assert msg.order is order


# ---------------------------------------------------------------------------
# Exchange-level stop order trigger tests
# ---------------------------------------------------------------------------
class TestExchangeStopTrigger:
    """Test _check_stop_orders in ExchangeAgent."""

    def _make_exchange(self, symbols=("TEST",)):
        from abides_markets.agents.exchange_agent import ExchangeAgent

        exchange = ExchangeAgent(
            id=0,
            mkt_open=MKT_OPEN,
            mkt_close=MKT_CLOSE,
            symbols=list(symbols),
            name="TestExchange",
            random_state=np.random.RandomState(42),
            log_orders=False,
            use_metric_tracker=False,
        )
        return exchange

    def test_stop_orders_initialised(self):
        exchange = self._make_exchange()
        assert "TEST" in exchange.stop_orders
        assert exchange.stop_orders["TEST"] == []

    def test_buy_stop_triggers_on_price_above(self):
        """Buy stop should fire when last_trade >= stop_price."""
        exchange = self._make_exchange()
        stop = StopOrder(1, MKT_OPEN, "TEST", 100, Side.BID, stop_price=10_000)
        exchange.stop_orders["TEST"].append((stop, 1))

        # Set last trade above stop price
        exchange.order_books["TEST"].last_trade = 10_100

        # Patch send_message and handle_market_order
        sent: list = []
        market_orders: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.order_books["TEST"].handle_market_order = lambda o: market_orders.append(o)
        exchange.current_time = MKT_OPEN + 1
        exchange.publish_order_book_data = lambda s: None

        exchange._check_stop_orders("TEST")

        assert len(exchange.stop_orders["TEST"]) == 0
        assert len(sent) == 1
        assert len(market_orders) == 1

        # Verify the market order has correct parameters
        mkt = market_orders[0]
        assert isinstance(mkt, MarketOrder)
        assert mkt.agent_id == 1
        assert mkt.quantity == 100
        assert mkt.side == Side.BID

    def test_sell_stop_triggers_on_price_below(self):
        """Sell stop should fire when last_trade <= stop_price."""
        exchange = self._make_exchange()
        stop = StopOrder(2, MKT_OPEN, "TEST", 50, Side.ASK, stop_price=9_500)
        exchange.stop_orders["TEST"].append((stop, 2))

        exchange.order_books["TEST"].last_trade = 9_400

        sent: list = []
        market_orders: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.order_books["TEST"].handle_market_order = lambda o: market_orders.append(o)
        exchange.current_time = MKT_OPEN + 1
        exchange.publish_order_book_data = lambda s: None

        exchange._check_stop_orders("TEST")

        assert len(exchange.stop_orders["TEST"]) == 0
        assert len(market_orders) == 1
        mkt = market_orders[0]
        assert mkt.side == Side.ASK
        assert mkt.quantity == 50

    def test_stop_not_triggered_below(self):
        """Buy stop does NOT fire when last_trade < stop_price."""
        exchange = self._make_exchange()
        stop = StopOrder(1, MKT_OPEN, "TEST", 100, Side.BID, stop_price=10_000)
        exchange.stop_orders["TEST"].append((stop, 1))

        exchange.order_books["TEST"].last_trade = 9_900

        sent: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.current_time = MKT_OPEN + 1

        exchange._check_stop_orders("TEST")

        # Stop should still be pending
        assert len(exchange.stop_orders["TEST"]) == 1
        assert len(sent) == 0

    def test_sell_stop_not_triggered_above(self):
        """Sell stop does NOT fire when last_trade > stop_price."""
        exchange = self._make_exchange()
        stop = StopOrder(2, MKT_OPEN, "TEST", 50, Side.ASK, stop_price=9_500)
        exchange.stop_orders["TEST"].append((stop, 2))

        exchange.order_books["TEST"].last_trade = 9_600

        sent: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.current_time = MKT_OPEN + 1

        exchange._check_stop_orders("TEST")

        assert len(exchange.stop_orders["TEST"]) == 1
        assert len(sent) == 0

    def test_no_trigger_without_last_trade(self):
        """No stops fire if no trades have occurred yet."""
        exchange = self._make_exchange()
        stop = StopOrder(1, MKT_OPEN, "TEST", 100, Side.BID, stop_price=10_000)
        exchange.stop_orders["TEST"].append((stop, 1))

        # last_trade should not be set on a fresh order book
        # (it starts as None by default, but might start as 0 with oracle opening)
        exchange.order_books["TEST"].last_trade = None

        sent: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.current_time = MKT_OPEN + 1

        exchange._check_stop_orders("TEST")

        assert len(exchange.stop_orders["TEST"]) == 1
        assert len(sent) == 0

    def test_buy_stop_triggers_at_exact_price(self):
        """Buy stop triggers at exactly the stop price."""
        exchange = self._make_exchange()
        stop = StopOrder(1, MKT_OPEN, "TEST", 100, Side.BID, stop_price=10_000)
        exchange.stop_orders["TEST"].append((stop, 1))

        exchange.order_books["TEST"].last_trade = 10_000

        sent: list = []
        market_orders: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.order_books["TEST"].handle_market_order = lambda o: market_orders.append(o)
        exchange.current_time = MKT_OPEN + 1
        exchange.publish_order_book_data = lambda s: None

        exchange._check_stop_orders("TEST")

        assert len(exchange.stop_orders["TEST"]) == 0
        assert len(market_orders) == 1

    def test_multiple_stops_partial_trigger(self):
        """Only matching stops trigger; others remain pending."""
        exchange = self._make_exchange()
        buy_stop = StopOrder(1, MKT_OPEN, "TEST", 100, Side.BID, stop_price=10_000)
        sell_stop = StopOrder(2, MKT_OPEN, "TEST", 50, Side.ASK, stop_price=9_000)
        exchange.stop_orders["TEST"].append((buy_stop, 1))
        exchange.stop_orders["TEST"].append((sell_stop, 2))

        # Trade at 10_100 — triggers buy stop but not sell stop
        exchange.order_books["TEST"].last_trade = 10_100

        sent: list = []
        market_orders: list = []
        exchange.send_message = lambda rid, msg: sent.append((rid, msg))
        exchange.order_books["TEST"].handle_market_order = lambda o: market_orders.append(o)
        exchange.current_time = MKT_OPEN + 1
        exchange.publish_order_book_data = lambda s: None

        exchange._check_stop_orders("TEST")

        # Buy stop triggered, sell stop remains
        assert len(exchange.stop_orders["TEST"]) == 1
        assert exchange.stop_orders["TEST"][0][0] is sell_stop
        assert len(market_orders) == 1


# ---------------------------------------------------------------------------
# TradingAgent.place_stop_order tests
# ---------------------------------------------------------------------------
class TestPlaceStopOrder:
    """Test that TradingAgent.place_stop_order creates and sends correctly."""

    def test_place_stop_order_records_in_orders(self):
        """place_stop_order should store a deepcopy in self.orders."""
        from unittest.mock import MagicMock

        from abides_markets.agents.trading_agent import TradingAgent

        agent = TradingAgent.__new__(TradingAgent)
        agent.id = 5
        agent.current_time = MKT_OPEN
        agent.exchange_id = 0
        agent.orders = {}
        agent.log_orders = False
        agent.send_message = MagicMock()

        agent.place_stop_order("TEST", 100, Side.BID, 10_000)

        assert len(agent.orders) == 1
        order = list(agent.orders.values())[0]
        assert isinstance(order, StopOrder)
        assert order.stop_price == 10_000
        assert order.quantity == 100
        assert order.side == Side.BID

        # Verify message was sent
        agent.send_message.assert_called_once()
        _, msg = agent.send_message.call_args.args
        from abides_markets.messages.order import StopOrderMsg

        assert isinstance(msg, StopOrderMsg)

    def test_stop_triggered_removes_from_orders(self):
        """stop_triggered should remove the stop order from self.orders."""
        from unittest.mock import MagicMock

        from abides_markets.agents.trading_agent import TradingAgent

        agent = TradingAgent.__new__(TradingAgent)
        agent.log_orders = False
        agent.orders = {}

        stop = StopOrder(5, MKT_OPEN, "TEST", 100, Side.BID, 10_000)
        agent.orders[stop.order_id] = stop

        agent.stop_triggered(stop)

        assert stop.order_id not in agent.orders
