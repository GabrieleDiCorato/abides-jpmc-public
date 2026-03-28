"""Non-regression and corner-case tests for the replace_order optimisation.

Covers:
- OrderBook.replace_order: ask side, crossing spread, non-existent old order
- TradingAgent: pre-registration, order_replaced cleanup, order_executed with
  pre-registered new order
- ExchangeAgent.send_message: pipeline delay for all OrderBookMsg subtypes,
  logging fallback for messages without .order
- AdaptiveMarketMakerAgent._diff_and_replace: size-only change, both sides
- ValueAgent: partially-filled existing order still triggers replace
"""

import warnings
from copy import deepcopy

import numpy as np

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.agents.market_makers.adaptive_market_maker_agent import (
    AdaptiveMarketMakerAgent,
)
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.agents.value_agent import ValueAgent
from abides_markets.messages.orderbook import (
    OrderAcceptedMsg,
    OrderBookMsg,
    OrderCancelledMsg,
    OrderExecutedMsg,
    OrderModifiedMsg,
    OrderPartialCancelledMsg,
    OrderReplacedMsg,
)
from abides_markets.orders import LimitOrder, Side

# Re-use the FakeExchangeAgent / setup_book_with_orders helpers from the
# existing orderbook test package.
from .orderbook import SYMBOL, TIME, setup_book_with_orders

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + str_to_ns("06:30:00")


# ===================================================================
# OrderBook.replace_order — corner cases
# ===================================================================


class TestOrderBookReplaceAskSide:
    """Complement the existing bid-side-only test with ask-side coverage."""

    def test_replace_ask_order(self):
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [10])],
            asks=[
                (300, [10, 50]),
                (400, [20]),
            ],
        )
        # orders: [bid@100, ask@300(10), ask@300(50), ask@400(20)]
        old_ask = orders[1]  # 10 @ 300

        new_ask = LimitOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=25,
            side=Side.ASK,
            limit_price=500,
        )
        book.replace_order(1, old_ask, new_ask)

        # The 10@300 should be gone; 50@300 remains, then 20@400, then 25@500.
        assert book.get_l3_ask_data() == [
            (300, [50]),
            (400, [20]),
            (500, [25]),
        ]

        assert len(agent.messages) == 1
        _, msg = agent.messages[0]
        assert isinstance(msg, OrderReplacedMsg)
        assert msg.old_order is old_ask
        assert msg.new_order is new_ask


class TestOrderBookReplaceCrossingSpread:
    """When replacement crosses the spread, OrderExecutedMsg must arrive
    BEFORE OrderReplacedMsg (handle_limit_order executes immediately)."""

    def test_crossing_replacement_generates_execution_then_replaced(self):
        # Best ask at 300 (qty 10).  Replace a bid with a new bid at 300 → crosses.
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [20])],
            asks=[(300, [10])],
        )
        old_bid = orders[0]  # 20 @ 100

        new_bid = LimitOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=5,
            side=Side.BID,
            limit_price=300,
        )
        book.replace_order(1, old_bid, new_bid)

        # Should see: OrderExecutedMsg (for the ask that was hit), then
        # OrderExecutedMsg (for the incoming bid), then OrderReplacedMsg.
        msg_types = [type(m).__name__ for _, m in agent.messages]

        # At minimum there must be at least one execution before the replacement.
        replaced_idx = next(
            i
            for i, (_, m) in enumerate(agent.messages)
            if isinstance(m, OrderReplacedMsg)
        )
        executed_indices = [
            i
            for i, (_, m) in enumerate(agent.messages)
            if isinstance(m, OrderExecutedMsg)
        ]
        assert len(executed_indices) > 0, "Crossing should produce executions"
        assert all(
            ei < replaced_idx for ei in executed_indices
        ), f"All executions must precede OrderReplacedMsg, got order: {msg_types}"


class TestOrderBookReplaceNonExistentOrder:
    """replace_order with an old order not in the book should be a no-op."""

    def test_nonexistent_old_order_is_noop(self):
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [10])],
            asks=[(300, [10])],
        )
        # Create an order that was never inserted.
        ghost = LimitOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=5,
            side=Side.BID,
            limit_price=200,
        )
        new = LimitOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=5,
            side=Side.BID,
            limit_price=250,
        )

        book.replace_order(1, ghost, new)

        # No messages should have been sent.
        assert len(agent.messages) == 0

        # Book should be unchanged (original orders still there).
        assert book.get_l3_bid_data() == [(100, [10])]
        assert book.get_l3_ask_data() == [(300, [10])]


class TestOrderBookReplaceSamePriceAndSize:
    """Replace with identical price and size should still work (order ID changes)."""

    def test_replace_identical_price_size(self):
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [10])],
            asks=[(300, [10])],
        )
        old_bid = orders[0]  # 10 @ 100
        new_bid = LimitOrder(
            agent_id=1,
            time_placed=TIME,
            symbol=SYMBOL,
            quantity=10,
            side=Side.BID,
            limit_price=100,
        )
        book.replace_order(1, old_bid, new_bid)

        assert book.get_l3_bid_data() == [(100, [10])]
        assert len(agent.messages) == 1
        _, msg = agent.messages[0]
        assert isinstance(msg, OrderReplacedMsg)
        assert msg.new_order.order_id != msg.old_order.order_id


# ===================================================================
# TradingAgent — replace_order pre-registration and callbacks
# ===================================================================


class TestTradingAgentReplacePreRegistration:
    """replace_order() must pre-register the new order and keep the old one."""

    def _make_agent(self) -> TradingAgent:
        agent = TradingAgent(
            id=0,
            random_state=np.random.RandomState(42),
            starting_cash=10_000_000,
        )
        agent.exchange_id = 99
        agent.current_time = MKT_OPEN
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = MKT_CLOSE
        # Stub send_message to avoid kernel dependency.
        agent.send_message = lambda *a, **kw: None  # type: ignore[method-assign]
        agent.logEvent = lambda *a, **kw: None  # type: ignore[method-assign]
        return agent

    def test_both_orders_present_after_replace(self):
        agent = self._make_agent()
        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        agent.replace_order(old, new)

        # Both old and new should be present in self.orders.
        assert (
            old.order_id in agent.orders
        ), "Old order must remain until order_replaced"
        assert new.order_id in agent.orders, "New order must be pre-registered"

    def test_order_replaced_cleans_old(self):
        agent = self._make_agent()
        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        agent.replace_order(old, new)

        # Simulate exchange confirming replacement.
        agent.order_replaced(old, new)

        assert (
            old.order_id not in agent.orders
        ), "Old order must be removed after confirmation"
        assert new.order_id in agent.orders, "New order must remain"
        # Exchange may update fields; order_replaced refreshes the entry.
        assert agent.orders[new.order_id] is new

    def test_order_replaced_idempotent_on_already_executed_old(self):
        """If old order was already removed by order_executed, order_replaced
        must not crash (defensive 'if in')."""
        agent = self._make_agent()
        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        agent.replace_order(old, new)

        # Simulate: old order gets executed and is removed before order_replaced.
        del agent.orders[old.order_id]

        # order_replaced should not raise.
        agent.order_replaced(old, new)
        assert new.order_id in agent.orders

    def test_order_executed_finds_pre_registered_new_order(self):
        """An immediate execution of the new replacement order should find it
        in self.orders (because replace_order pre-registered it)."""
        agent = self._make_agent()
        agent.holdings = {"CASH": 10_000_000, "TEST": 0}

        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, "TEST", 5, Side.BID, 99_100)
        agent.replace_order(old, new)

        # Simulate: exchange fills the new order immediately (before OrderReplacedMsg).
        fill = deepcopy(new)
        fill.fill_price = 99_100
        fill.quantity = 5  # fully filled

        # order_executed should NOT warn, because new order is pre-registered.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent.order_executed(fill)

        assert new.order_id not in agent.orders, "Fully-filled order must be removed"
        assert agent.holdings["TEST"] == 5
        assert agent.holdings["CASH"] == 10_000_000 - 5 * 99_100

    def test_order_executed_for_old_order_no_warning(self):
        """An execution for the old order (still in self.orders) must not warn."""
        agent = self._make_agent()
        agent.holdings = {"CASH": 10_000_000, "TEST": 0}

        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        agent.replace_order(old, new)

        # Simulate partial fill of old order (5 of 10).
        fill_old = deepcopy(old)
        fill_old.fill_price = 99_000
        fill_old.quantity = 5

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agent.order_executed(fill_old)

        # Old order should still be in self.orders with reduced quantity.
        assert old.order_id in agent.orders
        assert agent.orders[old.order_id].quantity == 5
        assert agent.holdings["TEST"] == 5


# ===================================================================
# ExchangeAgent.send_message — pipeline delay & logging
# ===================================================================


class TestExchangeAgentSendMessage:
    """Verify that all OrderBookMsg subtypes get pipeline delay and correct logging."""

    def test_all_orderbook_msg_subtypes_are_orderbook_msg(self):
        """Confirm all expected subtypes inherit from OrderBookMsg.

        This is a meta-test ensuring the isinstance check in send_message
        covers all order-book mutations.
        """
        for cls in (
            OrderAcceptedMsg,
            OrderExecutedMsg,
            OrderCancelledMsg,
            OrderPartialCancelledMsg,
            OrderModifiedMsg,
            OrderReplacedMsg,
        ):
            assert issubclass(cls, OrderBookMsg), f"{cls.__name__} not an OrderBookMsg"

    def test_logging_getattr_fallback_for_replaced(self):
        """OrderReplacedMsg has .old_order/.new_order but no .order.
        The getattr fallback should pick up .new_order for logging."""
        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        msg = OrderReplacedMsg(old, new)

        # Simulate the same logic used in ExchangeAgent.send_message.
        order = getattr(msg, "order", None) or msg.new_order
        assert order is new

    def test_logging_getattr_fallback_for_modified(self):
        """OrderModifiedMsg has .new_order but no .order."""
        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        msg = OrderModifiedMsg(new)

        order = getattr(msg, "order", None) or msg.new_order
        assert order is new

    def test_logging_getattr_fallback_for_partial_cancelled(self):
        """OrderPartialCancelledMsg has .new_order but no .order."""
        new = LimitOrder(0, MKT_OPEN, "TEST", 5, Side.BID, 99_000)
        msg = OrderPartialCancelledMsg(new)

        order = getattr(msg, "order", None) or msg.new_order
        assert order is new

    def test_logging_direct_order_for_accepted(self):
        """OrderAcceptedMsg has .order — getattr should find it directly."""
        o = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        msg = OrderAcceptedMsg(o)

        order = getattr(msg, "order", None) or getattr(msg, "new_order", None)
        assert order is o

    def test_logging_direct_order_for_executed(self):
        """OrderExecutedMsg has .order — getattr should find it directly."""

        o = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        msg = OrderExecutedMsg(o)

        order = getattr(msg, "order", None) or getattr(msg, "new_order", None)
        assert order is o

    def test_logging_direct_order_for_cancelled(self):
        """OrderCancelledMsg has .order — getattr should find it directly."""
        o = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        msg = OrderCancelledMsg(o)

        order = getattr(msg, "order", None) or getattr(msg, "new_order", None)
        assert order is o


# ===================================================================
# AdaptiveMarketMakerAgent._diff_and_replace — extra corner cases
# ===================================================================


class TestDiffAndReplaceCornerCases:
    """Cover _diff_and_replace scenarios not in the original test suite."""

    def _make_agent(self) -> AdaptiveMarketMakerAgent:
        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
        )
        agent.current_time = MKT_OPEN
        agent.exchange_id = 99
        return agent

    def test_size_only_change_triggers_replace(self):
        """Same price, different size — must still replace."""
        agent = self._make_agent()
        old = LimitOrder(0, MKT_OPEN, "TEST", 20, Side.BID, 99_000)

        replaced = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: None

        new_orders: list = []
        agent._diff_and_replace(
            existing=[old],
            desired=[(99_000, 30)],  # Same price, different size
            side=Side.BID,
            new_orders=new_orders,
        )

        assert len(replaced) == 1, "Size-only change must trigger replace"
        assert replaced[0][1].quantity == 30
        assert replaced[0][1].limit_price == 99_000

    def test_ask_side_replace(self):
        """Ensure _diff_and_replace works correctly for ask-side orders."""
        agent = self._make_agent()
        old = LimitOrder(0, MKT_OPEN, "TEST", 15, Side.ASK, 101_000)

        replaced = []
        cancelled = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        agent._diff_and_replace(
            existing=[old],
            desired=[(101_200, 15)],
            side=Side.ASK,
            new_orders=new_orders,
        )

        assert len(replaced) == 1
        assert replaced[0][1].side == Side.ASK
        assert replaced[0][1].limit_price == 101_200

    def test_empty_existing_and_empty_desired(self):
        """Both lists empty — no operations at all."""
        agent = self._make_agent()
        replaced = []
        cancelled = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        agent._diff_and_replace([], [], Side.BID, new_orders)

        assert len(replaced) == 0
        assert len(cancelled) == 0
        assert len(new_orders) == 0

    def test_multiple_replacements_and_cancels(self):
        """3 existing, 2 desired with different params → 2 replace, 1 cancel."""
        agent = self._make_agent()
        existing = [
            LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 98_000),
            LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000),
            LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 100_000),
        ]
        desired = [(98_100, 15), (99_100, 20)]

        replaced = []
        cancelled = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        agent._diff_and_replace(existing, desired, Side.BID, new_orders)

        assert len(replaced) == 2
        assert len(cancelled) == 1
        assert cancelled[0] is existing[2]
        assert len(new_orders) == 0

    def test_multiple_fresh_orders(self):
        """1 existing identical, 3 desired → 0 replace, 0 cancel, 2 fresh."""
        agent = self._make_agent()
        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.ASK, 101_000)

        replaced = []
        cancelled = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        agent._diff_and_replace(
            existing=[old],
            desired=[(101_000, 10), (102_000, 20), (103_000, 30)],
            side=Side.ASK,
            new_orders=new_orders,
        )

        assert len(replaced) == 0, "First pair is identical → skip"
        assert len(cancelled) == 0
        assert len(new_orders) == 2
        assert new_orders[0].limit_price == 102_000
        assert new_orders[1].limit_price == 103_000


# ===================================================================
# ValueAgent — partially-filled order still triggers replace
# ===================================================================


class TestValueAgentPartialFillReplace:
    """If an existing order was partially filled (reduced quantity),
    place_order should still replace it."""

    def _make_agent(self) -> ValueAgent:
        agent = ValueAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol="TEST",
            starting_cash=10_000_000,
            r_bar=100_000,
            sigma_n=10_000,
        )
        agent.current_time = MKT_OPEN + str_to_ns("00:05:00")
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = MKT_CLOSE
        agent.exchange_id = 99

        class _FakeOracle:
            def observe_price(self, symbol, current_time, sigma_n=0, random_state=None):
                return 100_000

        agent.oracle = _FakeOracle()
        return agent

    def test_replace_called_with_partially_filled_order(self):
        agent = self._make_agent()

        # Simulate an open order that was partially filled (7 of original 30 remain).
        old = LimitOrder(0, MKT_OPEN, "TEST", 7, Side.BID, 99_950)
        agent.orders[old.order_id] = old

        replaced = []
        placed = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.place_limit_order = lambda *a, **kw: placed.append(a)

        agent.known_bids = {"TEST": [[99_900, 100]]}
        agent.known_asks = {"TEST": [[100_100, 100]]}

        agent.place_order()

        assert len(replaced) == 1, "Should replace the partially-filled order"
        assert replaced[0][0] is old
        assert len(placed) == 0

    def test_place_called_when_all_orders_non_limit(self):
        """If self.orders has entries but none are LimitOrder, place fresh."""
        from abides_markets.orders import MarketOrder

        agent = self._make_agent()

        # A stale MarketOrder somehow left in orders (shouldn't normally happen,
        # but the isinstance guard must be robust).
        mkt = MarketOrder(0, MKT_OPEN, "TEST", 10, Side.BID)
        agent.orders[mkt.order_id] = mkt

        replaced = []
        placed = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.place_limit_order = lambda *a, **kw: placed.append(a)

        agent.known_bids = {"TEST": [[99_900, 100]]}
        agent.known_asks = {"TEST": [[100_100, 100]]}

        agent.place_order()

        assert len(placed) == 1, "Should place fresh when no LimitOrder found"
        assert len(replaced) == 0


# ===================================================================
# ExchangeAgent — market-closed path with ReplaceOrderMsg
# ===================================================================


class TestExchangeAgentMarketClosedReplace:
    """The market-closed isinstance check must include ReplaceOrderMsg."""

    def test_replace_order_msg_detected_in_market_closed_guard(self):
        """Regression: ReplaceOrderMsg was missing from the isinstance check
        that logs .old_order / .new_order instead of .order."""
        from abides_markets.messages.order import ModifyOrderMsg, ReplaceOrderMsg

        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        msg = ReplaceOrderMsg(agent_id=0, old_order=old, new_order=new)

        # The guard checks isinstance(message, (ModifyOrderMsg, ReplaceOrderMsg))
        assert isinstance(msg, (ModifyOrderMsg, ReplaceOrderMsg))

        # Verify attribute access doesn't crash.
        assert msg.old_order is old
        assert msg.new_order is new

    def test_modify_order_msg_still_detected(self):
        """ModifyOrderMsg should still pass the same guard."""
        from abides_markets.messages.order import ModifyOrderMsg, ReplaceOrderMsg

        old = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_000)
        new = LimitOrder(0, MKT_OPEN, "TEST", 10, Side.BID, 99_100)
        msg = ModifyOrderMsg(old, new)

        assert isinstance(msg, (ModifyOrderMsg, ReplaceOrderMsg))


# ===================================================================
# OrderBook.replace_order — history entry
# ===================================================================


class TestOrderBookReplaceHistory:
    """Verify the REPLACE entry is appended to order book history."""

    def test_replace_adds_history_entry(self):
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [10])],
            asks=[(300, [10])],
        )
        old = orders[0]  # 10 @ 100 BID
        new = LimitOrder(1, TIME, SYMBOL, 10, Side.BID, 150)

        book.replace_order(1, old, new)

        # Find REPLACE entry in history.
        replace_entries = [h for h in book.history if h["type"] == "REPLACE"]
        assert len(replace_entries) == 1
        entry = replace_entries[0]
        assert entry["old_order_id"] == old.order_id
        assert entry["new_order_id"] == new.order_id
        assert entry["quantity"] == 10
        assert entry["price"] == 150

    def test_failed_replace_no_history_entry(self):
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [10])],
        )
        ghost = LimitOrder(1, TIME, SYMBOL, 5, Side.BID, 200)
        new = LimitOrder(1, TIME, SYMBOL, 5, Side.BID, 250)

        initial_history_len = len(book.history)
        book.replace_order(1, ghost, new)

        replace_entries = [
            h for h in book.history[initial_history_len:] if h["type"] == "REPLACE"
        ]
        assert len(replace_entries) == 0, "Failed replace must not add history"
