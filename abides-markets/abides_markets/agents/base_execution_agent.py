"""
Base Slicing Execution Agent

Shared infrastructure for execution agents that divide a parent order into
time-sliced child orders (TWAP, VWAP, POV, etc.).

Subclasses override ``_compute_slice_quantity()`` to define the per-wakeup
slice-sizing logic.  Everything else — the state machine, fill tracking,
arrival-price capture, and order placement — lives here.
"""

from __future__ import annotations

import abc
import logging
from typing import Literal

import numpy as np

from abides_core import Message, NanosecondTime

from ..messages.query import QuerySpreadResponseMsg
from ..models.risk_config import RiskConfig
from ..orders import Side, TimeInForce
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)


class BaseSlicingExecutionAgent(TradingAgent, abc.ABC):
    """Abstract base for time-sliced execution agents.

    The agent follows a fixed state machine:

        AWAITING_WAKEUP → AWAITING_MARKET_DATA → EXECUTING → COMPLETE

    On each wakeup it:
    1. Queries the current spread (for arrival price / IOC pricing).
    2. Calls ``_compute_slice_quantity()`` to decide how many shares to send.
    3. Places either a market order or an IOC limit at the near touch.
    4. Schedules the next wakeup.
    """

    VALID_STATES = frozenset(
        {"AWAITING_WAKEUP", "AWAITING_MARKET_DATA", "EXECUTING", "COMPLETE"}
    )

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        start_time: NanosecondTime,
        end_time: NanosecondTime,
        freq: NanosecondTime,
        direction: Side = Side.BID,
        quantity: int = 1000,
        trade: bool = True,
        order_style: Literal["market", "ioc_limit"] = "market",
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        log_orders: bool = False,
        risk_config: RiskConfig | None = None,
    ) -> None:
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
        self.start_time: NanosecondTime = start_time
        self.end_time: NanosecondTime = end_time
        self.freq: NanosecondTime = freq
        self.direction: Side = direction
        self.quantity: int = quantity
        self.trade: bool = trade
        self.order_style: Literal["market", "ioc_limit"] = order_style

        # Execution tracking
        self.executed_quantity: int = 0
        self.remaining_quantity: int = quantity

        # State management
        self.state: str = "AWAITING_WAKEUP"
        self.execution_started: bool = False
        self.execution_complete: bool = False

        # Market data
        self.last_bid: int | None = None
        self.last_ask: int | None = None

        # Arrival price: mid at first wakeup (integer cents or None)
        self.arrival_mid_cents: int | None = None

        # Logging and statistics
        self.execution_history: list[dict] = []

    # ------------------------------------------------------------------
    # Abstract method
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _compute_slice_quantity(self, current_time: NanosecondTime) -> int:
        """Return the number of shares to execute in this slice.

        Implementations may inspect ``self.remaining_quantity``,
        ``self.executed_quantity``, ``self.freq``, book data, etc.
        Must return a non-negative integer ≤ ``self.remaining_quantity``.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        logger.info(
            f"{self.name}: Execution agent starting. "
            f"Target: {self.quantity} shares, "
            f"Direction: {self.direction.value}"
        )

    def kernel_stopping(self) -> None:
        super().kernel_stopping()
        pct = self.executed_quantity / self.quantity * 100 if self.quantity > 0 else 0
        logger.info(
            f"{self.name}: Execution complete. "
            f"Executed: {self.executed_quantity}/{self.quantity} ({pct:.1f}%)"
        )
        self.logEvent(
            "EXECUTION_SUMMARY",
            {
                "executed_quantity": self.executed_quantity,
                "target_quantity": self.quantity,
                "remaining_quantity": self.remaining_quantity,
                "execution_rate": pct,
            },
        )

    # ------------------------------------------------------------------
    # Wakeup / message handling
    # ------------------------------------------------------------------
    def wakeup(self, current_time: NanosecondTime) -> None:  # type: ignore[override]
        super().wakeup(current_time)

        if not self._should_be_trading(current_time):
            return

        if not self.execution_started:
            self.execution_started = True
            logger.info(f"{self.name}: Starting execution at {current_time}")

        if self.execution_complete or self.remaining_quantity <= 0:
            return

        # Query spread (needed for IOC pricing + arrival price)
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_MARKET_DATA"

    def _should_be_trading(self, current_time: NanosecondTime) -> bool:
        if not self.mkt_open or not self.mkt_close:
            # Market hours not yet known — re-wakeup shortly so we retry
            # once the MarketHoursMsg arrives.
            self.set_wakeup(current_time + self.freq)
            return False
        if current_time < self.start_time:
            self.set_wakeup(self.start_time)
            return False
        if current_time >= self.end_time:
            if not self.execution_complete:
                self.execution_complete = True
                logger.info(
                    f"{self.name}: Execution period ended. "
                    f"Executed {self.executed_quantity}/{self.quantity}"
                )
            return False
        if not self.mkt_open or not self.mkt_close:
            return False
        return not self.mkt_closed

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        super().receive_message(current_time, sender_id, message)

        if self.state != "AWAITING_MARKET_DATA":
            return

        if isinstance(message, QuerySpreadResponseMsg):
            result = self.get_known_bid_ask(self.symbol)
            if len(result) == 4:
                bid, _bv, ask, _av = result
                self.last_bid = bid
                self.last_ask = ask

                # Capture arrival price once
                if (
                    self.arrival_mid_cents is None
                    and bid is not None
                    and ask is not None
                ):
                    self.arrival_mid_cents = (bid + ask) // 2

            self._on_spread_received(current_time)

    def _on_spread_received(self, current_time: NanosecondTime) -> None:
        """Called once the spread response arrives.  Subclasses that need
        additional market data (e.g. transacted volume) should override this
        to issue another query and defer ``_execute_slice`` until that
        response arrives.  The default implementation executes immediately.
        """
        self._execute_slice(current_time)

    # ------------------------------------------------------------------
    # Slice execution
    # ------------------------------------------------------------------
    def _execute_slice(self, current_time: NanosecondTime) -> None:
        self.state = "EXECUTING"

        order_size = self._compute_slice_quantity(current_time)
        order_size = min(max(order_size, 0), self.remaining_quantity)

        if order_size <= 0:
            self._schedule_next_wakeup(current_time)
            return

        self.logEvent(
            "SLICE_DECISION",
            {
                "time": current_time,
                "order_size": order_size,
                "remaining_quantity": self.remaining_quantity,
                "direction": self.direction.value,
            },
        )

        if self.trade:
            self._place_execution_order(order_size)

        self._schedule_next_wakeup(current_time)

    def _place_execution_order(self, order_size: int) -> None:
        if order_size <= 0:
            return

        if self.order_style == "ioc_limit":
            # IOC limit at the near touch
            price = self.last_ask if self.direction == Side.BID else self.last_bid

            if price is None:
                logger.warning(
                    f"{self.name}: No price for IOC limit, falling back to market"
                )
                self.place_market_order(self.symbol, order_size, self.direction)
                return

            self.place_limit_order(
                symbol=self.symbol,
                quantity=order_size,
                side=self.direction,
                limit_price=price,
                time_in_force=TimeInForce.IOC,
            )
        else:
            # Market order
            if self.direction == Side.BID and self.last_ask is None:
                logger.warning(f"{self.name}: No ask price, skipping order")
                return
            if self.direction == Side.ASK and self.last_bid is None:
                logger.warning(f"{self.name}: No bid price, skipping order")
                return
            self.place_market_order(self.symbol, order_size, self.direction)

    def _schedule_next_wakeup(self, current_time: NanosecondTime) -> None:
        next_wakeup = current_time + self.freq
        if next_wakeup < self.end_time:
            self.set_wakeup(next_wakeup)
            self.state = "AWAITING_WAKEUP"
        else:
            self.state = "COMPLETE"

    # ------------------------------------------------------------------
    # Fill tracking
    # ------------------------------------------------------------------
    def order_executed(self, order) -> None:
        super().order_executed(order)

        if order.agent_id == self.id:
            qty = order.quantity
            self.executed_quantity += qty
            self.remaining_quantity -= qty
            self.execution_history.append(
                {
                    "time": self.current_time,
                    "quantity": qty,
                    "side": order.side.value,
                    "fill_price": order.fill_price,
                }
            )
            logger.info(
                f"{self.name}: Filled {qty} @ {order.fill_price}. "
                f"Progress: {self.executed_quantity}/{self.quantity}"
            )
            if self.remaining_quantity <= 0:
                self.execution_complete = True

    def get_wake_frequency(self) -> NanosecondTime:
        return self.freq
