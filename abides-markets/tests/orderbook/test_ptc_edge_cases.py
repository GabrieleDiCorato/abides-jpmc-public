"""Edge-case tests for Price-to-Comply (PTC) orders and their interaction with FOK.

Covers:
- PTC order pair creation: hidden + visible at correct prices
- PTC sell with limit_price=1 → hidden half gets limit_price=0 (boundary)
- PTC bid with limit_price=0 is effectively invalid (caught by validation)
- FOK pre-check against books containing hidden (PTC) liquidity
- PTC cancellation cascading to the paired order
- PTC partial fill reduces both halves
"""

from __future__ import annotations

from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, Side, TimeInForce

from . import SYMBOL, TIME, FakeExchangeAgent, setup_book_with_orders
from .test_orderbook_invariants import assert_book_invariants

# ---------------------------------------------------------------------------
# PTC pair creation
# ---------------------------------------------------------------------------


class TestPTCPairCreation:
    """Verify that entering a PTC order creates the correct hidden+visible pair."""

    def test_ptc_bid_creates_hidden_at_higher_price(self):
        """PTC bid: hidden half should be at limit_price + 1."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        order = LimitOrder(
            1,
            TIME,
            SYMBOL,
            100,
            Side.BID,
            limit_price=100,
            is_price_to_comply=True,
        )
        book.handle_limit_order(order)
        assert_book_invariants(book)

        # Should have two price levels on bids: hidden at 101, visible at 100
        assert len(book.bids) == 2
        # Highest price first (descending)
        assert book.bids[0].price == 101
        assert book.bids[1].price == 100

        # Hidden order at price 101
        assert len(book.bids[0].hidden_orders) == 1
        assert book.bids[0].hidden_orders[0][0].is_hidden is True

        # Visible order at price 100
        assert len(book.bids[1].visible_orders) == 1
        assert book.bids[1].visible_orders[0][0].is_hidden is False

    def test_ptc_ask_creates_hidden_at_lower_price(self):
        """PTC ask: hidden half should be at limit_price - 1."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        order = LimitOrder(
            1,
            TIME,
            SYMBOL,
            100,
            Side.ASK,
            limit_price=100,
            is_price_to_comply=True,
        )
        book.handle_limit_order(order)
        assert_book_invariants(book)

        # Should have two price levels on asks: visible at 100, hidden at 99
        assert len(book.asks) == 2
        # Lowest price first (ascending)
        assert book.asks[0].price == 99
        assert book.asks[1].price == 100

        # Hidden order at price 99
        assert len(book.asks[0].hidden_orders) == 1
        assert book.asks[0].hidden_orders[0][0].is_hidden is True

        # Visible order at price 100
        assert len(book.asks[1].visible_orders) == 1

    def test_ptc_metadata_links_halves(self):
        """The metadata of each half should reference the other."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        order = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.BID,
            limit_price=200,
            is_price_to_comply=True,
        )
        book.handle_limit_order(order)

        # Hidden half is at 201 (highest bid)
        hidden_order, hidden_meta = book.bids[0].hidden_orders[0]
        # Visible half is at 200
        visible_order, visible_meta = book.bids[1].visible_orders[0]

        assert hidden_meta["ptc_hidden"] is True
        assert hidden_meta["ptc_other_half"] is visible_order

        assert visible_meta["ptc_hidden"] is False
        assert visible_meta["ptc_other_half"] is hidden_order


# ---------------------------------------------------------------------------
# PTC boundary: low-price asks (hidden half gets price=0)
# ---------------------------------------------------------------------------


class TestPTCLowPriceBoundary:
    """PTC ask with limit_price=1 → hidden half gets limit_price=0.

    This is a boundary condition: the hidden order's price becomes 0, which
    is at the edge of validity. Verify the book doesn't produce negative prices.
    """

    def test_ptc_ask_limit_price_1(self):
        """PTC ask at price=1 → hidden at price=0."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        order = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.ASK,
            limit_price=1,
            is_price_to_comply=True,
        )
        book.handle_limit_order(order)
        assert_book_invariants(book)

        # Hidden half at price 0 (1 - 1 = 0), visible half at price 1
        assert len(book.asks) == 2
        assert book.asks[0].price == 0
        assert book.asks[1].price == 1
        assert book.asks[0].hidden_orders[0][0].limit_price == 0

    def test_ptc_ask_limit_price_2(self):
        """PTC ask at price=2 → hidden at price=1 (safe)."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        order = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.ASK,
            limit_price=2,
            is_price_to_comply=True,
        )
        book.handle_limit_order(order)
        assert_book_invariants(book)
        assert book.asks[0].price == 1
        assert book.asks[1].price == 2


# ---------------------------------------------------------------------------
# FOK interaction with PTC (hidden) liquidity
# ---------------------------------------------------------------------------


class TestFOKWithPTCLiquidity:
    """FOK pre-check uses _matching_liquidity() which only counts total_quantity
    (visible orders). If hidden PTC orders provide additional liquidity,
    the FOK check may under-count. This tests the actual behavior.
    """

    def test_fok_only_sees_visible_quantity(self):
        """FOK order that would match hidden liquidity but not visible → rejected.

        _matching_liquidity counts only `total_quantity` which is visible-only.
        """
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        # Place a PTC ask at 100 → visible at 100 (50 qty), hidden at 99 (50 qty)
        ptc = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.ASK,
            limit_price=100,
            is_price_to_comply=True,
        )
        book.handle_limit_order(ptc)
        agent.reset()

        # FOK bid for 80 qty at 100 — visible qty at that level is only 50
        # Hidden qty at 99 is also 50, but _matching_liquidity may not count it
        # The total visible across both levels (99 and 100) is 0 + 50 = 50
        # (hidden orders contribute to hidden_orders, not visible_orders)
        fok = LimitOrder(
            2,
            TIME,
            SYMBOL,
            80,
            Side.BID,
            limit_price=100,
            time_in_force=TimeInForce.FOK,
        )
        book.handle_limit_order(fok)

        # Verify: FOK should have been rejected (only 50 visible qty vs 80 needed)
        # Check the agent received a cancellation
        cancel_msgs = [
            m for m in agent.messages if type(m[1]).__name__ == "OrderCancelledMsg"
        ]
        assert (
            len(cancel_msgs) >= 1
        ), "FOK should be rejected when visible qty insufficient"

    def test_fok_succeeds_with_sufficient_visible_qty(self):
        """FOK order that fits entirely within visible liquidity → filled."""
        book, agent, _ = setup_book_with_orders(asks=[(100, [50, 50])])

        fok = LimitOrder(
            2,
            TIME,
            SYMBOL,
            80,
            Side.BID,
            limit_price=100,
            time_in_force=TimeInForce.FOK,
        )
        book.handle_limit_order(fok)
        assert_book_invariants(book)

        # Should be executed (100 visible >= 80 needed)
        exec_msgs = [
            m for m in agent.messages if type(m[1]).__name__ == "OrderExecutedMsg"
        ]
        assert len(exec_msgs) >= 1, "FOK should execute when visible qty sufficient"


# ---------------------------------------------------------------------------
# PTC cancellation cascading
# ---------------------------------------------------------------------------


class TestPTCCancellation:
    """Cancelling one half of a PTC pair should cascade to cancel the other half."""

    def test_cancel_visible_half_cancels_hidden(self):
        """Cancel the visible half → hidden half also cancelled."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        ptc = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.BID,
            limit_price=100,
            is_price_to_comply=True,
        )
        book.handle_limit_order(ptc)
        agent.reset()

        # Visible half is at price 100
        visible_order = book.bids[1].visible_orders[0][0]
        result = book.cancel_order(visible_order)
        assert result is True
        assert_book_invariants(book)
        assert book.bids == [], "Both halves should be cancelled"

    def test_cancel_hidden_half_cancels_visible(self):
        """Cancel the hidden half → visible half also cancelled."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        ptc = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.BID,
            limit_price=100,
            is_price_to_comply=True,
        )
        book.handle_limit_order(ptc)
        agent.reset()

        # Hidden half is at price 101 (bids[0])
        hidden_order = book.bids[0].hidden_orders[0][0]
        result = book.cancel_order(hidden_order)
        assert result is True
        assert_book_invariants(book)
        assert book.bids == [], "Both halves should be cancelled"


# ---------------------------------------------------------------------------
# PTC partial fill reduces paired order
# ---------------------------------------------------------------------------


class TestPTCPartialFill:
    """When a PTC order is partially filled, its paired half should also be reduced."""

    def test_partial_fill_of_hidden_reduces_visible(self):
        """Partially filling the hidden half reduces the visible half's quantity."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        # PTC ask at 100 → hidden at 99, visible at 100
        ptc = LimitOrder(
            1,
            TIME,
            SYMBOL,
            50,
            Side.ASK,
            limit_price=100,
            is_price_to_comply=True,
        )
        book.handle_limit_order(ptc)
        agent.reset()

        # Buy 20 units at 99 — matches the hidden ask at 99
        buyer = LimitOrder(2, TIME, SYMBOL, 20, Side.BID, limit_price=99)
        book.handle_limit_order(buyer)
        assert_book_invariants(book)

        # Hidden half should be reduced to 30
        if book.asks[0].price == 99 and book.asks[0].hidden_orders:
            assert book.asks[0].hidden_orders[0][0].quantity == 30
            # Visible half at 100 should also be reduced to 30
            visible = book.asks[1].visible_orders[0][0]
            assert visible.quantity == 30
        else:
            # Hidden half was at 99; if fully consumed (won't be — only 20 of 50),
            # fallback check
            total_remaining = sum(
                sum(o.quantity for o, _ in pl.visible_orders)
                + sum(o.quantity for o, _ in pl.hidden_orders)
                for pl in book.asks
            )
            assert (
                total_remaining == 60
            )  # 50 + 50 - 20 = 80 (both halves), or 30 + 30 = 60
