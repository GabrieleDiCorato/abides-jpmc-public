"""Tests for per-symbol position limit enforcement in TradingAgent.

Covers:
- _pending_order_delta: basic computation, exclude_order_id
- _check_position_limit: no limit, within limit, breach blocked, breach clamped,
  short side, zero room clamp
- create_limit_order: blocked/clamped/allowed
- place_market_order: blocked/clamped/allowed
- place_multiple_orders: cumulative batch enforcement, partial batch clamp
- replace_order: exclude old order, block on breach, clamp on breach
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
    position_limit: int | None = None,
    position_limit_clamp: bool = False,
    starting_cash: int = 10_000_000,
) -> TradingAgent:
    agent = TradingAgent(
        id=0,
        random_state=np.random.RandomState(42),
        starting_cash=starting_cash,
        position_limit=position_limit,
        position_limit_clamp=position_limit_clamp,
    )
    agent.exchange_id = 99
    agent.current_time = MKT_OPEN
    agent.mkt_open = MKT_OPEN
    agent.mkt_close = MKT_CLOSE
    agent.send_message = lambda *a, **kw: None
    agent.send_message_batch = lambda *a, **kw: None
    agent.logEvent = lambda *a, **kw: None
    return agent


# ===================================================================
# _pending_order_delta
# ===================================================================


class TestPendingOrderDelta:
    def test_empty_orders(self):
        agent = _make_agent(position_limit=100)
        assert agent._pending_order_delta(SYMBOL) == 0

    def test_bid_adds_positive(self):
        agent = _make_agent(position_limit=100)
        order = LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_000)
        agent.orders[order.order_id] = order
        assert agent._pending_order_delta(SYMBOL) == 30

    def test_ask_adds_negative(self):
        agent = _make_agent(position_limit=100)
        order = LimitOrder(0, MKT_OPEN, SYMBOL, 20, Side.ASK, 10_000)
        agent.orders[order.order_id] = order
        assert agent._pending_order_delta(SYMBOL) == -20

    def test_mixed_orders(self):
        agent = _make_agent(position_limit=100)
        bid = LimitOrder(0, MKT_OPEN, SYMBOL, 50, Side.BID, 10_000)
        ask = LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.ASK, 10_000)
        agent.orders[bid.order_id] = bid
        agent.orders[ask.order_id] = ask
        assert agent._pending_order_delta(SYMBOL) == 20

    def test_different_symbol_ignored(self):
        agent = _make_agent(position_limit=100)
        order = LimitOrder(0, MKT_OPEN, "OTHER", 40, Side.BID, 10_000)
        agent.orders[order.order_id] = order
        assert agent._pending_order_delta(SYMBOL) == 0

    def test_exclude_order_id(self):
        agent = _make_agent(position_limit=100)
        bid1 = LimitOrder(0, MKT_OPEN, SYMBOL, 50, Side.BID, 10_000)
        bid2 = LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_000)
        agent.orders[bid1.order_id] = bid1
        agent.orders[bid2.order_id] = bid2
        # Excluding bid1 should give only bid2's delta.
        assert agent._pending_order_delta(SYMBOL, exclude_order_id=bid1.order_id) == 30


# ===================================================================
# _check_position_limit
# ===================================================================


class TestCheckPositionLimit:
    def test_no_limit_passes_through(self):
        agent = _make_agent(position_limit=None)
        assert agent._check_position_limit(SYMBOL, 999, Side.BID) == 999

    def test_within_limit_allowed(self):
        agent = _make_agent(position_limit=100)
        assert agent._check_position_limit(SYMBOL, 50, Side.BID) == 50

    def test_exactly_at_limit_allowed(self):
        agent = _make_agent(position_limit=100)
        assert agent._check_position_limit(SYMBOL, 100, Side.BID) == 100

    def test_exceeds_limit_blocked(self):
        agent = _make_agent(position_limit=100)
        assert agent._check_position_limit(SYMBOL, 101, Side.BID) == 0

    def test_exceeds_limit_clamped(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        assert agent._check_position_limit(SYMBOL, 150, Side.BID) == 100

    def test_short_side_within_limit(self):
        agent = _make_agent(position_limit=100)
        assert agent._check_position_limit(SYMBOL, 80, Side.ASK) == 80

    def test_short_side_exceeds_blocked(self):
        agent = _make_agent(position_limit=100)
        assert agent._check_position_limit(SYMBOL, 101, Side.ASK) == 0

    def test_short_side_exceeds_clamped(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        assert agent._check_position_limit(SYMBOL, 150, Side.ASK) == 100

    def test_existing_holdings_reduce_room(self):
        agent = _make_agent(position_limit=100)
        agent.holdings[SYMBOL] = 60  # already long 60
        assert agent._check_position_limit(SYMBOL, 50, Side.BID) == 0  # 60+50=110 > 100

    def test_existing_holdings_reduce_room_clamped(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        agent.holdings[SYMBOL] = 60
        assert agent._check_position_limit(SYMBOL, 50, Side.BID) == 40  # room=100-60=40

    def test_pending_orders_counted(self):
        agent = _make_agent(position_limit=100)
        # 70 shares pending buy
        pending = LimitOrder(0, MKT_OPEN, SYMBOL, 70, Side.BID, 10_000)
        agent.orders[pending.order_id] = pending
        # Trying to buy 40 more: net = 0 + 70 + 40 = 110 > 100
        assert agent._check_position_limit(SYMBOL, 40, Side.BID) == 0

    def test_pending_orders_clamped(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        pending = LimitOrder(0, MKT_OPEN, SYMBOL, 70, Side.BID, 10_000)
        agent.orders[pending.order_id] = pending
        assert agent._check_position_limit(SYMBOL, 40, Side.BID) == 30  # room=100-70=30

    def test_opposite_side_adds_room(self):
        """Long holdings + an ask order should be ok."""
        agent = _make_agent(position_limit=100)
        agent.holdings[SYMBOL] = 80
        # Selling 30 is fine: net goes from 80 to 50
        assert agent._check_position_limit(SYMBOL, 30, Side.ASK) == 30

    def test_exclude_order_id_in_check(self):
        """exclude_order_id should skip the given order in pending delta."""
        agent = _make_agent(position_limit=100)
        old = LimitOrder(0, MKT_OPEN, SYMBOL, 80, Side.BID, 10_000)
        agent.orders[old.order_id] = old
        # Without exclusion: net = 0+80 = already 80 pending.
        # Buying 90 more → 170 > 100 → blocked.
        assert agent._check_position_limit(SYMBOL, 90, Side.BID) == 0
        # With exclusion: net = 0+0 = 0. Buying 90 → 90 ≤ 100 → allowed.
        assert agent._check_position_limit(
            SYMBOL, 90, Side.BID, exclude_order_id=old.order_id
        ) == 90

    def test_zero_room_clamp_returns_zero(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        agent.holdings[SYMBOL] = 100
        assert agent._check_position_limit(SYMBOL, 10, Side.BID) == 0


# ===================================================================
# create_limit_order integration
# ===================================================================


class TestCreateLimitOrderPositionLimit:
    def test_no_limit_creates_order(self):
        agent = _make_agent(position_limit=None)
        order = agent.create_limit_order(SYMBOL, 50, Side.BID, 10_000)
        assert order is not None
        assert order.quantity == 50

    def test_within_limit_creates_order(self):
        agent = _make_agent(position_limit=100)
        order = agent.create_limit_order(SYMBOL, 80, Side.BID, 10_000)
        assert order is not None
        assert order.quantity == 80

    def test_exceeds_limit_returns_none(self):
        agent = _make_agent(position_limit=100)
        result = agent.create_limit_order(SYMBOL, 150, Side.BID, 10_000)
        assert result is None

    def test_exceeds_limit_clamped(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        order = agent.create_limit_order(SYMBOL, 150, Side.BID, 10_000)
        assert order is not None
        assert order.quantity == 100

    def test_short_exceeds_returns_none(self):
        agent = _make_agent(position_limit=50)
        result = agent.create_limit_order(SYMBOL, 60, Side.ASK, 10_000)
        assert result is None

    def test_holdings_affect_limit(self):
        agent = _make_agent(position_limit=100)
        agent.holdings[SYMBOL] = 70
        result = agent.create_limit_order(SYMBOL, 40, Side.BID, 10_000)
        assert result is None  # 70+40=110 > 100

    def test_holdings_clamped(self):
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        agent.holdings[SYMBOL] = 70
        order = agent.create_limit_order(SYMBOL, 40, Side.BID, 10_000)
        assert order is not None
        assert order.quantity == 30  # room=100-70=30

    def test_zero_quantity_still_warns(self):
        """Quantity 0 should trigger the existing warning, not the position check."""
        agent = _make_agent(position_limit=100)
        with pytest.warns(UserWarning, match="quantity zero"):
            result = agent.create_limit_order(SYMBOL, 0, Side.BID, 10_000)
        assert result is None


# ===================================================================
# place_market_order integration
# ===================================================================


class TestPlaceMarketOrderPositionLimit:
    def test_within_limit_sends(self):
        sent = []
        agent = _make_agent(position_limit=100)
        agent.send_message = lambda dest, msg: sent.append(msg)
        agent.place_market_order(SYMBOL, 50, Side.BID)
        assert len(sent) == 1

    def test_exceeds_limit_blocked(self):
        sent = []
        agent = _make_agent(position_limit=100)
        agent.send_message = lambda dest, msg: sent.append(msg)
        agent.place_market_order(SYMBOL, 150, Side.BID)
        assert len(sent) == 0

    def test_exceeds_limit_clamped(self):
        sent = []
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        agent.send_message = lambda dest, msg: sent.append(msg)
        agent.place_market_order(SYMBOL, 150, Side.BID)
        assert len(sent) == 1
        # The order that was sent should have clamped quantity.
        order_id = list(agent.orders.keys())[0]
        assert agent.orders[order_id].quantity == 100


# ===================================================================
# place_multiple_orders integration
# ===================================================================


class TestPlaceMultipleOrdersPositionLimit:
    def test_all_within_limit(self):
        sent_batches = []
        agent = _make_agent(position_limit=100)
        agent.send_message_batch = lambda dest, msgs: sent_batches.append(msgs)
        orders = [
            LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_000),
            LimitOrder(0, MKT_OPEN, SYMBOL, 30, Side.BID, 10_100),
        ]
        agent.place_multiple_orders(orders)
        assert len(sent_batches) == 1
        assert len(sent_batches[0]) == 2

    def test_cumulative_breach_blocks_later_orders(self):
        """Second order in batch is blocked because the first already filled the limit."""
        sent_batches = []
        agent = _make_agent(position_limit=100)
        agent.send_message_batch = lambda dest, msgs: sent_batches.append(msgs)
        orders = [
            LimitOrder(0, MKT_OPEN, SYMBOL, 80, Side.BID, 10_000),
            LimitOrder(0, MKT_OPEN, SYMBOL, 80, Side.BID, 10_100),
        ]
        agent.place_multiple_orders(orders)
        # First order fits (0+80=80 ≤ 100), second would make it 160 → blocked.
        assert len(sent_batches) == 1
        assert len(sent_batches[0]) == 1
        assert len(agent.orders) == 1

    def test_cumulative_breach_clamps_later_orders(self):
        sent_batches = []
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        agent.send_message_batch = lambda dest, msgs: sent_batches.append(msgs)
        orders = [
            LimitOrder(0, MKT_OPEN, SYMBOL, 80, Side.BID, 10_000),
            LimitOrder(0, MKT_OPEN, SYMBOL, 80, Side.BID, 10_100),
        ]
        agent.place_multiple_orders(orders)
        assert len(sent_batches) == 1
        assert len(sent_batches[0]) == 2
        # Second order should be clamped to 20 (100-80=20).
        second_oid = sent_batches[0][1].order.order_id
        assert agent.orders[second_oid].quantity == 20

    def test_no_limit_sends_all(self):
        sent_batches = []
        agent = _make_agent(position_limit=None)
        agent.send_message_batch = lambda dest, msgs: sent_batches.append(msgs)
        orders = [
            LimitOrder(0, MKT_OPEN, SYMBOL, 500, Side.BID, 10_000),
            LimitOrder(0, MKT_OPEN, SYMBOL, 500, Side.BID, 10_100),
        ]
        agent.place_multiple_orders(orders)
        assert len(sent_batches[0]) == 2


# ===================================================================
# replace_order integration
# ===================================================================


class TestReplaceOrderPositionLimit:
    def test_replace_within_limit_succeeds(self):
        sent = []
        agent = _make_agent(position_limit=100)
        agent.send_message = lambda dest, msg: sent.append(msg)

        old = LimitOrder(0, MKT_OPEN, SYMBOL, 50, Side.BID, 10_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, SYMBOL, 80, Side.BID, 10_000)
        agent.replace_order(old, new)
        assert len(sent) == 1
        assert new.order_id in agent.orders

    def test_replace_exceeds_limit_blocked(self):
        sent = []
        agent = _make_agent(position_limit=100)
        agent.send_message = lambda dest, msg: sent.append(msg)

        old = LimitOrder(0, MKT_OPEN, SYMBOL, 50, Side.BID, 10_000)
        agent.orders[old.order_id] = deepcopy(old)

        # Replacing with 150 → net (excluding old) = 0, + 150 = 150 > 100 → blocked.
        new = LimitOrder(0, MKT_OPEN, SYMBOL, 150, Side.BID, 10_000)
        agent.replace_order(old, new)
        assert len(sent) == 0
        # New order should NOT be pre-registered since it was blocked.
        assert new.order_id not in agent.orders

    def test_replace_exceeds_limit_clamped(self):
        sent = []
        agent = _make_agent(position_limit=100, position_limit_clamp=True)
        agent.send_message = lambda dest, msg: sent.append(msg)

        old = LimitOrder(0, MKT_OPEN, SYMBOL, 50, Side.BID, 10_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, SYMBOL, 150, Side.BID, 10_000)
        agent.replace_order(old, new)
        assert len(sent) == 1
        # Clamped to 100 (room = 100 - 0 excluding old = 100).
        assert agent.orders[new.order_id].quantity == 100

    def test_replace_excludes_old_order_from_delta(self):
        """Old order's exposure should not count against the new order."""
        sent = []
        agent = _make_agent(position_limit=100)
        agent.send_message = lambda dest, msg: sent.append(msg)

        old = LimitOrder(0, MKT_OPEN, SYMBOL, 90, Side.BID, 10_000)
        agent.orders[old.order_id] = deepcopy(old)

        # Without exclusion: net = 0 + 90 (pending old) + 95 = 185 > 100 → blocked.
        # With exclusion: net = 0 + 0 + 95 = 95 ≤ 100 → allowed.
        new = LimitOrder(0, MKT_OPEN, SYMBOL, 95, Side.BID, 10_000)
        agent.replace_order(old, new)
        assert len(sent) == 1

    def test_replace_side_change_checked(self):
        """Replacing a BID with an ASK should check the new ASK exposure."""
        sent = []
        agent = _make_agent(position_limit=50)
        agent.send_message = lambda dest, msg: sent.append(msg)

        old = LimitOrder(0, MKT_OPEN, SYMBOL, 40, Side.BID, 10_000)
        agent.orders[old.order_id] = deepcopy(old)

        # Replacing with ASK 60: net (excl old) = 0, +(-60) = -60, abs > 50 → blocked.
        new = LimitOrder(0, MKT_OPEN, SYMBOL, 60, Side.ASK, 10_000)
        agent.replace_order(old, new)
        assert len(sent) == 0

    def test_replace_no_limit_always_succeeds(self):
        sent = []
        agent = _make_agent(position_limit=None)
        agent.send_message = lambda dest, msg: sent.append(msg)

        old = LimitOrder(0, MKT_OPEN, SYMBOL, 50, Side.BID, 10_000)
        agent.orders[old.order_id] = deepcopy(old)

        new = LimitOrder(0, MKT_OPEN, SYMBOL, 99999, Side.BID, 10_000)
        agent.replace_order(old, new)
        assert len(sent) == 1


# ===================================================================
# Config system integration
# ===================================================================


class TestConfigSystemPositionLimit:
    def test_base_agent_config_has_fields(self):
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        config = BaseAgentConfig(position_limit=200, position_limit_clamp=True)
        assert config.position_limit == 200
        assert config.position_limit_clamp is True

    def test_base_agent_config_defaults(self):
        from abides_markets.config_system.agent_configs import BaseAgentConfig

        config = BaseAgentConfig()
        assert config.position_limit is None
        assert config.position_limit_clamp is False
