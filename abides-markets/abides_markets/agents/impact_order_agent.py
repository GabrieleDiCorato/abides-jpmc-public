"""Impact-order agent — places a single configurable order at a fixed time.

This agent is designed for **price-impact studies**: it sits dormant until
a pre-configured absolute time, fires exactly one order, then becomes
permanently inactive.  The order time, side, quantity, and order type are
all configurable.

Three order types are supported:

``MARKET``
    A plain market order.  Guaranteed to fill immediately, but may walk
    several price levels in a thin book.

``LIMIT``
    A limit order at a caller-supplied explicit price (``limit_price``).
    Rests in the book unfilled if the price is never touched.

``AGGRESSIVE_LIMIT``
    Queries the current spread at execution time, then places a limit order
    at the *best opposite-side price* (i.e. buys at the ask, sells at the
    bid).  Fills immediately at the near touch like a market order, but
    avoids walking deeper into the book.  If the book side is empty at
    execution time, falls back to a plain market order and logs a warning.

Usage via config system::

    from abides_markets.config_system import SimulationBuilder

    config = (SimulationBuilder()
        .from_template("rmsc04")
        .enable_agent(
            "impact_order",
            order_time_offset="01:00:00",   # 1 hour after open
            side="BID",
            quantity=10_000,
            order_type="AGGRESSIVE_LIMIT",
        )
        .seed(42)
        .build())
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from abides_core import Message, NanosecondTime
from abides_markets.models.risk_config import RiskConfig

from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)

# Valid order-type constants (avoid raw-string comparisons elsewhere).
_ORDER_TYPE_MARKET = "MARKET"
_ORDER_TYPE_LIMIT = "LIMIT"
_ORDER_TYPE_AGGRESSIVE_LIMIT = "AGGRESSIVE_LIMIT"
_VALID_ORDER_TYPES: frozenset[str] = frozenset(
    {_ORDER_TYPE_MARKET, _ORDER_TYPE_LIMIT, _ORDER_TYPE_AGGRESSIVE_LIMIT}
)


class ImpactOrderAgent(TradingAgent):
    """Single-shot agent that places one configurable order at a fixed time.

    The agent is dormant until ``order_time``, fires the order, then enters
    state ``"DONE"`` for the rest of the simulation.

    State machine::

        AWAITING_WAKEUP
            │  wakeup() called before order_time → reschedule
            │  wakeup() called at/after order_time:
            │    MARKET / LIMIT → submit immediately
            │    AGGRESSIVE_LIMIT → query spread
            ▼
        AWAITING_SPREAD     (only for AGGRESSIVE_LIMIT)
            │  QuerySpreadResponseMsg arrives → submit
            ▼
        DONE                (permanent; all subsequent messages ignored)
    """

    VALID_STATES = frozenset({"AWAITING_WAKEUP", "AWAITING_SPREAD", "DONE"})

    def __init__(
        self,
        id: int,
        order_time: NanosecondTime,
        quantity: int,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        symbol: str = "IBM",
        starting_cash: int = 100_000_000,
        log_orders: bool = False,
        risk_config: RiskConfig | None = None,
        side: Side = Side.BID,
        order_type: str = _ORDER_TYPE_MARKET,
        limit_price: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        id:
            Agent id — auto-injected by the config system.
        order_time:
            Absolute simulation time (nanoseconds) at which the order is fired.
            Use ``ImpactOrderAgentConfig.order_time_offset`` to express this as a
            duration from market open; the config computes the absolute value.
        quantity:
            Number of shares in the order.  Must be ≥ 1.
        name:
            Human-readable label — auto-set by the config system.
        type:
            Agent type tag for logging — auto-set by the config system.
        random_state:
            Seeded RNG — auto-injected by the config system.
        symbol:
            Ticker symbol to trade.  Must match the exchange symbol.
        starting_cash:
            Initial cash in **integer cents** (e.g. $1 000 000 = 100_000_000).
            Defaults to $1 000 000, which is sufficient for most impact studies.
        log_orders:
            When True, emits a ``ORDER_SUBMITTED`` log event for every order.
        risk_config:
            Optional risk guards (position limit, drawdown circuit breaker, …).
            Bundled by the config system from the ``position_limit`` /
            ``max_drawdown`` / … fields on ``ImpactOrderAgentConfig``.
        side:
            ``Side.BID`` (buy) or ``Side.ASK`` (sell).
        order_type:
            One of ``"MARKET"``, ``"LIMIT"``, or ``"AGGRESSIVE_LIMIT"``.
        limit_price:
            Limit price in **integer cents** (e.g. $100.00 = 10_000).
            Required when ``order_type == "LIMIT"``; ignored otherwise.
        """
        if order_type not in _VALID_ORDER_TYPES:
            raise ValueError(
                f"ImpactOrderAgent: invalid order_type {order_type!r}. "
                f"Must be one of {sorted(_VALID_ORDER_TYPES)}."
            )
        if quantity < 1:
            raise ValueError(
                f"ImpactOrderAgent: quantity must be ≥ 1, got {quantity}."
            )
        if order_type == _ORDER_TYPE_LIMIT and limit_price is None:
            raise ValueError(
                "ImpactOrderAgent: limit_price is required when order_type is 'LIMIT'."
            )

        super().__init__(
            id,
            name,
            type,
            random_state,
            starting_cash,
            log_orders,
            risk_config=risk_config,
        )

        self.symbol: str = symbol
        self.order_time: NanosecondTime = order_time
        self.quantity: int = quantity
        self.side: Side = side
        self.order_type: str = order_type
        self.limit_price: int | None = limit_price

        self.state: str = "AWAITING_WAKEUP"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        # Schedule the first (and typically only) wakeup at order_time.
        self.set_wakeup(self.order_time)

    def kernel_stopping(self) -> None:
        super().kernel_stopping()

    # ------------------------------------------------------------------
    # wakeup()
    # ------------------------------------------------------------------

    def wakeup(self, current_time: NanosecondTime) -> None:
        # Guard: returns False when market hours are unknown or market is closed.
        if not super().wakeup(current_time):
            return

        if self.state == "DONE":
            return

        # Not yet time to fire — reschedule precisely at order_time.
        if current_time < self.order_time:
            self.set_wakeup(self.order_time)
            return

        if self.order_type == _ORDER_TYPE_AGGRESSIVE_LIMIT:
            # Need the current spread before we can set the price.
            # get_current_spread() sends a message; the reply arrives
            # asynchronously in receive_message() as QuerySpreadResponseMsg.
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self._submit_order()
            self.state = "DONE"

    # ------------------------------------------------------------------
    # receive_message()
    # ------------------------------------------------------------------

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # CRITICAL: always call super() first — it updates cached bids/asks
        # and portfolio state.
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREAD" and isinstance(
            message, QuerySpreadResponseMsg
        ):
            self._submit_order()
            self.state = "DONE"

    # ------------------------------------------------------------------
    # _submit_order()
    # ------------------------------------------------------------------

    def _submit_order(self) -> None:
        """Place the configured order.  Called at most once per simulation."""
        if self.order_type == _ORDER_TYPE_MARKET:
            self.place_market_order(self.symbol, self.quantity, self.side)

        elif self.order_type == _ORDER_TYPE_LIMIT:
            # limit_price is validated non-None at construction time.
            assert self.limit_price is not None
            self.place_limit_order(
                self.symbol, self.quantity, self.side, self.limit_price
            )

        elif self.order_type == _ORDER_TYPE_AGGRESSIVE_LIMIT:
            # For an aggressive limit we cross the spread: buys execute at
            # the current ask; sells at the current bid.
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

            price = ask if self.side.is_bid() else bid

            if price is None:
                # Book side is empty — fall back to market order so the
                # agent still participates.
                warnings.warn(
                    f"ImpactOrderAgent {self.name}: book side is empty at "
                    f"order_time; falling back from AGGRESSIVE_LIMIT to MARKET.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.place_market_order(self.symbol, self.quantity, self.side)
            else:
                self.place_limit_order(
                    self.symbol, self.quantity, self.side, price
                )

        logger.debug(
            "{} submitted {} {} {} shares (order_type={}).",
            self.name,
            self.side,
            self.quantity,
            self.symbol,
            self.order_type,
        )
