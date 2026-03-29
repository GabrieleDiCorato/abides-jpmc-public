"""Thin market scenario tests.

Tests for order book behavior under thin/illiquid conditions:
- One-sided books (only bids or only asks)
- Market orders into empty book sides
- Wide spread conditions
- Minimal liquidity fills
- Crossed-book scenarios
"""

from abides_markets.messages.orderbook import OrderExecutedMsg
from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, MarketOrder, Side

from . import SYMBOL, TIME, FakeExchangeAgent, setup_book_with_orders


class TestOneSidedBook:
    """Order book with liquidity on only one side."""

    def test_bid_only_book_ask_order_executes(self):
        """Incoming ask crosses the lone bid."""
        book, agent, orders = setup_book_with_orders(bids=[(100, [10])], asks=[])
        ask = LimitOrder(2, TIME, SYMBOL, 10, Side.ASK, 100)
        book.handle_limit_order(ask)

        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) >= 1
        assert book.bids == []
        assert book.asks == []

    def test_ask_only_book_bid_order_executes(self):
        """Incoming bid crosses the lone ask."""
        book, agent, orders = setup_book_with_orders(bids=[], asks=[(100, [10])])
        bid = LimitOrder(2, TIME, SYMBOL, 10, Side.BID, 100)
        book.handle_limit_order(bid)

        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) >= 1
        assert book.bids == []
        assert book.asks == []

    def test_bid_only_book_bid_order_rests(self):
        """Incoming bid into bid-only book rests at its price."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [10])], asks=[])
        new_bid = LimitOrder(2, TIME, SYMBOL, 5, Side.BID, 99)
        book.handle_limit_order(new_bid)

        assert len(book.bids) == 2
        assert book.bids[0].price == 100
        assert book.bids[1].price == 99
        assert book.asks == []

    def test_ask_only_book_ask_order_rests(self):
        """Incoming ask into ask-only book rests."""
        book, agent, _ = setup_book_with_orders(bids=[], asks=[(100, [10])])
        new_ask = LimitOrder(2, TIME, SYMBOL, 5, Side.ASK, 101)
        book.handle_limit_order(new_ask)

        assert len(book.asks) == 2
        assert book.asks[0].price == 100
        assert book.asks[1].price == 101


class TestMarketOrderIntoEmptySide:
    """Market orders when the opposite side is empty."""

    def test_market_buy_empty_asks(self):
        """Market buy with no asks — nothing to fill against."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)
        # Only bids exist.
        bid = LimitOrder(1, TIME, SYMBOL, 10, Side.BID, 100)
        book.handle_limit_order(bid)
        agent.reset()

        mkt = MarketOrder(2, TIME, SYMBOL, 5, Side.BID)
        book.handle_market_order(mkt)

        # No executions should occur (no asks).
        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) == 0

    def test_market_sell_empty_bids(self):
        """Market sell with no bids — nothing to fill against."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)
        ask = LimitOrder(1, TIME, SYMBOL, 10, Side.ASK, 100)
        book.handle_limit_order(ask)
        agent.reset()

        mkt = MarketOrder(2, TIME, SYMBOL, 5, Side.ASK)
        book.handle_market_order(mkt)

        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) == 0

    def test_market_buy_completely_empty_book(self):
        """Market buy on a 100% empty book."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        mkt = MarketOrder(2, TIME, SYMBOL, 5, Side.BID)
        book.handle_market_order(mkt)

        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) == 0


class TestWideSpreads:
    """Book with large bid-ask spread."""

    def test_wide_spread_resting_order(self):
        """Order within the spread rests without executing."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [10])], asks=[(200, [10])])
        # Place a bid within the spread but below the ask.
        mid_bid = LimitOrder(2, TIME, SYMBOL, 5, Side.BID, 150)
        book.handle_limit_order(mid_bid)

        # Should become the new best bid — no execution.
        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) == 0
        assert book.bids[0].price == 150

    def test_wide_spread_crossing_ask(self):
        """Bid at the ask price crosses despite wide spread."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [10])], asks=[(200, [10])])
        crossing_bid = LimitOrder(2, TIME, SYMBOL, 5, Side.BID, 200)
        book.handle_limit_order(crossing_bid)

        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) >= 1

    def test_spread_narrows_after_mid_order(self):
        """Placing orders within the spread narrows it."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [10])], asks=[(200, [10])])
        # Place a bid at 150 and an ask at 160
        bid = LimitOrder(2, TIME, SYMBOL, 5, Side.BID, 150)
        ask = LimitOrder(3, TIME, SYMBOL, 5, Side.ASK, 160)
        book.handle_limit_order(bid)
        book.handle_limit_order(ask)

        assert book.bids[0].price == 150
        assert book.asks[0].price == 160
        # Spread narrowed from 100 to 10
        spread = book.asks[0].price - book.bids[0].price
        assert spread == 10


class TestMinimalLiquidity:
    """Edge cases with minimum liquidity (1 share)."""

    def test_single_share_full_fill(self):
        """1-share bid + 1-share ask → full execution."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [1])], asks=[])
        ask = LimitOrder(2, TIME, SYMBOL, 1, Side.ASK, 100)
        book.handle_limit_order(ask)

        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) >= 1
        assert book.bids == []
        assert book.asks == []

    def test_large_order_partial_fill_thin_book(self):
        """Large order sweeps all thin liquidity and remainder rests."""
        book, agent, _ = setup_book_with_orders(
            bids=[], asks=[(100, [1]), (101, [1]), (102, [1])]
        )
        # Buy 10 shares — only 3 available.
        big_bid = LimitOrder(2, TIME, SYMBOL, 10, Side.BID, 105)
        book.handle_limit_order(big_bid)

        # 3 shares filled, 7 remaining rests.
        exec_msgs = [m for _, m in agent.messages if isinstance(m, OrderExecutedMsg)]
        assert len(exec_msgs) >= 1  # at least one execution
        # Asks should be swept
        assert book.asks == []
        # Remaining 7 shares rest as a bid
        assert len(book.bids) == 1
        assert book.bids[0].total_quantity == 7

    def test_sweeping_multiple_price_levels(self):
        """Aggressive order sweeps through multiple price levels."""
        book, agent, _ = setup_book_with_orders(
            bids=[],
            asks=[(100, [5]), (101, [5]), (102, [5])],
        )
        # Buy 12 shares — sweeps 100 (5), 101 (5), 102 (2 of 5)
        bid = LimitOrder(2, TIME, SYMBOL, 12, Side.BID, 102)
        book.handle_limit_order(bid)

        # Check remaining
        assert len(book.asks) == 1
        assert book.asks[0].price == 102
        assert book.asks[0].total_quantity == 3  # 5 - 2 remaining


class TestEmptyBookQueries:
    """Query operations on an empty or one-sided book."""

    def test_empty_book_no_bids_no_asks(self):
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)
        assert book.bids == []
        assert book.asks == []
        assert book.last_trade is None

    def test_empty_book_l1_query(self):
        """L1 query on empty book returns empty/None best prices."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)
        # Best bid/ask should be None-like.
        best_bid = book.bids[0].price if book.bids else None
        best_ask = book.asks[0].price if book.asks else None
        assert best_bid is None
        assert best_ask is None

    def test_cancel_nonexistent_order(self):
        """Cancelling an order not in the book should not crash."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)
        fake_order = LimitOrder(1, TIME, SYMBOL, 10, Side.BID, 100)
        # Attempt to cancel — should not raise.
        book.cancel_order(fake_order)
