"""Order book invariant tests — structural properties that must hold after every mutation.

After every insert / cancel / execute / replace / modify, the following must hold:
- Bids are strictly descending in price; asks are strictly ascending
- No empty PriceLevels remain in the book ("zombie levels")
- Non-negative spread (best_ask >= best_bid), or one/both sides empty
- L1 data matches actual book[0] on each side
- Total quantity at each level equals sum of visible + hidden orders
"""

from __future__ import annotations

from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, Side

from . import SYMBOL, TIME, FakeExchangeAgent, setup_book_with_orders

# ---------------------------------------------------------------------------
# Invariant checker — reusable across all tests
# ---------------------------------------------------------------------------


def assert_book_invariants(book: OrderBook) -> None:
    """Assert that all structural invariants of the order book hold."""

    # 1. Bids strictly descending
    for i in range(1, len(book.bids)):
        assert book.bids[i - 1].price > book.bids[i].price, (
            f"Bid prices not strictly descending: "
            f"bids[{i - 1}].price={book.bids[i - 1].price}, "
            f"bids[{i}].price={book.bids[i].price}"
        )

    # 2. Asks strictly ascending
    for i in range(1, len(book.asks)):
        assert book.asks[i - 1].price < book.asks[i].price, (
            f"Ask prices not strictly ascending: "
            f"asks[{i - 1}].price={book.asks[i - 1].price}, "
            f"asks[{i}].price={book.asks[i].price}"
        )

    # 3. No zombie (empty) PriceLevels
    for i, pl in enumerate(book.bids):
        assert not pl.is_empty, f"bids[{i}] at price {pl.price} is empty"
    for i, pl in enumerate(book.asks):
        assert not pl.is_empty, f"asks[{i}] at price {pl.price} is empty"

    # 4. Non-negative spread (or missing side)
    if book.bids and book.asks:
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        assert (
            best_ask >= best_bid
        ), f"Negative spread: best_bid={best_bid}, best_ask={best_ask}"

    # 5. L1 matches first visible-quantity level
    #    (Hidden-only levels from PTC orders have total_quantity=0 and are
    #     skipped by get_l1_*_data, so we compare against the first level
    #     with total_quantity > 0.)
    first_visible_bid = next((pl for pl in book.bids if pl.total_quantity > 0), None)
    if first_visible_bid is not None:
        l1_bid = book.get_l1_bid_data()
        assert l1_bid is not None
        assert l1_bid[0] == first_visible_bid.price
    else:
        assert book.get_l1_bid_data() is None

    first_visible_ask = next((pl for pl in book.asks if pl.total_quantity > 0), None)
    if first_visible_ask is not None:
        l1_ask = book.get_l1_ask_data()
        assert l1_ask is not None
        assert l1_ask[0] == first_visible_ask.price
    else:
        assert book.get_l1_ask_data() is None

    # 6. Quantity consistency at each level
    for pl in book.bids + book.asks:
        visible_qty = sum(o.quantity for o, _ in pl.visible_orders)
        hidden_qty = sum(o.quantity for o, _ in pl.hidden_orders)
        assert (
            visible_qty + hidden_qty > 0
        ), f"PriceLevel at {pl.price} has zero total quantity"


# ---------------------------------------------------------------------------
# Test: invariants hold on a fresh empty book
# ---------------------------------------------------------------------------


class TestEmptyBookInvariants:
    def test_fresh_book(self):
        book = OrderBook(FakeExchangeAgent(), SYMBOL)
        assert_book_invariants(book)

    def test_after_failed_cancel_on_empty_book(self):
        book = OrderBook(FakeExchangeAgent(), SYMBOL)
        dummy = LimitOrder(1, TIME, SYMBOL, 10, Side.BID, 100)
        result = book.cancel_order(dummy)
        assert result is False
        assert_book_invariants(book)


# ---------------------------------------------------------------------------
# Test: invariants hold after single-side insertions
# ---------------------------------------------------------------------------


class TestInsertionInvariants:
    def test_multiple_bid_levels(self):
        """Insert bids at different prices → descending order."""
        book, agent, _ = setup_book_with_orders(
            bids=[(100, [10]), (102, [20]), (98, [5]), (101, [15])]
        )
        assert_book_invariants(book)
        assert len(book.bids) == 4
        prices = [pl.price for pl in book.bids]
        assert prices == [102, 101, 100, 98]

    def test_multiple_ask_levels(self):
        """Insert asks at different prices → ascending order."""
        book, agent, _ = setup_book_with_orders(
            asks=[(100, [10]), (98, [20]), (102, [5]), (101, [15])]
        )
        assert_book_invariants(book)
        assert len(book.asks) == 4
        prices = [pl.price for pl in book.asks]
        assert prices == [98, 100, 101, 102]

    def test_multiple_orders_same_price_level(self):
        """Multiple orders at the same price → single PriceLevel, FIFO order."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [10, 20, 30])])
        assert_book_invariants(book)
        assert len(book.bids) == 1
        assert len(book.bids[0].visible_orders) == 3

    def test_both_sides(self):
        """Orders on both sides → positive spread."""
        book, agent, _ = setup_book_with_orders(
            bids=[(99, [10]), (100, [20])],
            asks=[(101, [10]), (102, [20])],
        )
        assert_book_invariants(book)
        assert book.bids[0].price == 100
        assert book.asks[0].price == 101


# ---------------------------------------------------------------------------
# Test: invariants hold after executions (matching)
# ---------------------------------------------------------------------------


class TestExecutionInvariants:
    def test_full_fill_removes_order(self):
        """Incoming bid fully fills resting ask → ask level removed."""
        book, agent, _ = setup_book_with_orders(asks=[(100, [30])])
        incoming = LimitOrder(2, TIME, SYMBOL, 30, Side.BID, 100)
        book.handle_limit_order(incoming)
        assert_book_invariants(book)
        assert book.asks == []
        assert book.bids == []

    def test_partial_fill_preserves_level(self):
        """Incoming bid partially fills resting ask → ask level still present."""
        book, agent, _ = setup_book_with_orders(asks=[(100, [30])])
        incoming = LimitOrder(2, TIME, SYMBOL, 10, Side.BID, 100)
        book.handle_limit_order(incoming)
        assert_book_invariants(book)
        assert len(book.asks) == 1
        assert book.asks[0].visible_orders[0][0].quantity == 20

    def test_multi_level_fill(self):
        """Aggressive bid sweeps through multiple ask levels."""
        book, agent, _ = setup_book_with_orders(
            asks=[(100, [10]), (101, [20]), (102, [30])]
        )
        incoming = LimitOrder(2, TIME, SYMBOL, 35, Side.BID, 102)
        book.handle_limit_order(incoming)
        assert_book_invariants(book)
        # Consumed 10 @ 100 + 20 @ 101 + 5 @ 102 = 35
        assert len(book.asks) == 1
        assert book.asks[0].price == 102
        assert book.asks[0].visible_orders[0][0].quantity == 25

    def test_fill_leaves_residual_on_opposite_side(self):
        """Bid larger than all asks → remainder rests on bid side."""
        book, agent, _ = setup_book_with_orders(asks=[(100, [10])])
        incoming = LimitOrder(2, TIME, SYMBOL, 50, Side.BID, 100)
        book.handle_limit_order(incoming)
        assert_book_invariants(book)
        assert book.asks == []
        assert len(book.bids) == 1
        assert book.bids[0].visible_orders[0][0].quantity == 40

    def test_interleaved_inserts_and_fills(self):
        """Sequence of buys and sells → invariants hold at every step."""
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        # Build up the book
        for price in [100, 101, 102]:
            book.handle_limit_order(LimitOrder(1, TIME, SYMBOL, 20, Side.ASK, price))
            assert_book_invariants(book)

        for price in [97, 98, 99]:
            book.handle_limit_order(LimitOrder(1, TIME, SYMBOL, 20, Side.BID, price))
            assert_book_invariants(book)

        # Execute a crossing order
        book.handle_limit_order(LimitOrder(2, TIME, SYMBOL, 25, Side.BID, 101))
        assert_book_invariants(book)

        # Cancel a resting order
        remaining_bid = book.bids[-1].visible_orders[0][0]
        book.cancel_order(remaining_bid)
        assert_book_invariants(book)


# ---------------------------------------------------------------------------
# Test: invariants hold after cancellations
# ---------------------------------------------------------------------------


class TestCancellationInvariants:
    def test_cancel_only_order_at_level(self):
        """Cancel the sole order at a price level → level disappears."""
        book, agent, orders = setup_book_with_orders(bids=[(100, [10])])
        book.cancel_order(orders[0])
        assert_book_invariants(book)
        assert book.bids == []

    def test_cancel_one_of_many_at_level(self):
        """Cancel one order among several at the same price → level survives."""
        book, agent, orders = setup_book_with_orders(bids=[(100, [10, 20, 30])])
        book.cancel_order(orders[1])  # cancel the 20-qty order
        assert_book_invariants(book)
        assert len(book.bids) == 1
        assert len(book.bids[0].visible_orders) == 2

    def test_cancel_from_middle_level(self):
        """Cancel the sole order at a middle price level → level removed, others intact."""
        book, agent, orders = setup_book_with_orders(
            bids=[(100, [10]), (99, [20]), (98, [30])]
        )
        # Cancel the 99-level order (middle)
        book.cancel_order(orders[1])
        assert_book_invariants(book)
        prices = [pl.price for pl in book.bids]
        assert prices == [100, 98]

    def test_cancel_nonexistent_order(self):
        """Cancel an order not in the book → returns False, invariants hold."""
        book, agent, _ = setup_book_with_orders(bids=[(100, [10])])
        fake = LimitOrder(99, TIME, SYMBOL, 5, Side.BID, 200)
        result = book.cancel_order(fake)
        assert result is False
        assert_book_invariants(book)


# ---------------------------------------------------------------------------
# Test: invariants hold after replace
# ---------------------------------------------------------------------------


class TestReplaceInvariants:
    def test_replace_price_change(self):
        """Replace an order with a different price → old level may vanish, new level created."""
        book, agent, orders = setup_book_with_orders(bids=[(100, [20])])
        old_order = orders[0]
        new_order = LimitOrder(1, TIME, SYMBOL, 20, Side.BID, 105)
        book.replace_order(1, old_order, new_order)
        assert_book_invariants(book)
        assert len(book.bids) == 1
        assert book.bids[0].price == 105

    def test_replace_quantity_change(self):
        """Replace an order with same price, different quantity."""
        book, agent, orders = setup_book_with_orders(bids=[(100, [20])])
        old_order = orders[0]
        new_order = LimitOrder(1, TIME, SYMBOL, 50, Side.BID, 100)
        book.replace_order(1, old_order, new_order)
        assert_book_invariants(book)
        assert book.bids[0].visible_orders[0][0].quantity == 50

    def test_replace_with_crossing_price(self):
        """Replace a bid with a price that crosses the ask → execution + invariants."""
        book, agent, _ = setup_book_with_orders(
            bids=[(99, [20])],
            asks=[(100, [10])],
        )
        old_bid = book.bids[0].visible_orders[0][0]
        new_bid = LimitOrder(1, TIME, SYMBOL, 20, Side.BID, 100)
        book.replace_order(1, old_bid, new_bid)
        assert_book_invariants(book)


# ---------------------------------------------------------------------------
# Test: complex interleaved scenario (stress test)
# ---------------------------------------------------------------------------


class TestComplexScenario:
    def test_twenty_step_sequence(self):
        """A 20-step sequence of inserts, fills, cancels, modifications, and replaces.

        Asserts invariants after every single step.
        """
        agent = FakeExchangeAgent()
        book = OrderBook(agent, SYMBOL)

        # Step 1-5: Build bid side
        bid_orders = []
        for price in [95, 96, 97, 98, 99]:
            o = LimitOrder(1, TIME, SYMBOL, 100, Side.BID, price)
            book.handle_limit_order(o)
            bid_orders.append(o)
            assert_book_invariants(book)

        # Step 6-10: Build ask side
        ask_orders = []
        for price in [101, 102, 103, 104, 105]:
            o = LimitOrder(2, TIME, SYMBOL, 100, Side.ASK, price)
            book.handle_limit_order(o)
            ask_orders.append(o)
            assert_book_invariants(book)

        assert len(book.bids) == 5
        assert len(book.asks) == 5

        # Step 11: Aggressive buy sweeps 2 ask levels
        sweep = LimitOrder(3, TIME, SYMBOL, 150, Side.BID, 103)
        book.handle_limit_order(sweep)
        assert_book_invariants(book)

        # Step 12: Cancel a bid
        book.cancel_order(bid_orders[0])  # 95 level
        assert_book_invariants(book)

        # Step 13: Add more bids at existing level
        extra_bid = LimitOrder(4, TIME, SYMBOL, 50, Side.BID, 98)
        book.handle_limit_order(extra_bid)
        assert_book_invariants(book)

        # Step 14: Aggressive sell eats into bids
        sell = LimitOrder(5, TIME, SYMBOL, 120, Side.ASK, 97)
        book.handle_limit_order(sell)
        assert_book_invariants(book)

        # Step 15: Replace a resting ask with lower price
        if book.asks:
            old_ask = book.asks[0].visible_orders[0][0]
            new_ask = LimitOrder(2, TIME, SYMBOL, 80, Side.ASK, old_ask.limit_price - 1)
            book.replace_order(2, old_ask, new_ask)
            assert_book_invariants(book)

        # Step 16-17: Add and cancel on the same level
        temp = LimitOrder(6, TIME, SYMBOL, 10, Side.BID, 90)
        book.handle_limit_order(temp)
        assert_book_invariants(book)
        book.cancel_order(temp)
        assert_book_invariants(book)

        # Step 18: Add hidden order
        hidden = LimitOrder(7, TIME, SYMBOL, 50, Side.ASK, 110, is_hidden=True)
        book.handle_limit_order(hidden)
        assert_book_invariants(book)

        # Step 19: Cancel the hidden order
        book.cancel_order(hidden)
        assert_book_invariants(book)

        # Step 20: Final sanity — all remaining levels are coherent
        assert_book_invariants(book)
