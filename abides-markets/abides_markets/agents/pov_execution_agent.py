"""
POV (Percentage of Volume) Execution Agent

This agent implements a Percentage of Volume (POV) execution strategy, commonly used
in institutional trading to execute large orders while minimizing market impact.

The agent targets a specified percentage of market volume, placing orders that are
proportional to the observed transacted volume over a lookback period.
"""

import logging
from typing import Union

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ..messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from ..models.risk_config import RiskConfig
from ..orders import Side
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)

_DEFAULT_FREQ: int = str_to_ns("1min")


class POVExecutionAgent(TradingAgent):
    """
    Percentage of Volume (POV) Execution Agent.

    This agent executes a target quantity of shares by participating at a specified
    percentage of the market's transacted volume. The agent:

    1. Periodically wakes up and queries the transacted volume over a lookback period
    2. Calculates its target execution size as: pov * transacted_volume
    3. Places market or limit orders to achieve that target
    4. Continues until the total target quantity is executed or time expires

    This is a common execution algorithm used to minimize market impact while
    maintaining a consistent participation rate in the market.

    Attributes:
        symbol: The trading symbol.
        starting_cash: Initial cash holdings.
        start_time: Time when the agent starts executing.
        end_time: Time when the agent stops executing.
        freq: How often the agent wakes up to place orders.
        lookback_period: Time period to look back for volume calculation.
        pov: Target percentage of volume (0.0 to 1.0).
        direction: Side.BID for buying, Side.ASK for selling.
        quantity: Total target quantity to execute.
        trade: Whether to actually place trades (if False, just logs).
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
        freq: NanosecondTime = _DEFAULT_FREQ,
        lookback_period: Union[NanosecondTime, str] | None = None,
        pov: float = 0.1,
        direction: Side = Side.BID,
        quantity: int = 1000,
        trade: bool = True,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        log_orders: bool = False,
        risk_config: RiskConfig | None = None,
    ) -> None:
        """
        Initialize the POV Execution Agent.

        Args:
            id: Unique agent identifier.
            symbol: Trading symbol (e.g., "ABM").
            starting_cash: Initial cash in cents.
            start_time: Absolute time to start execution (nanoseconds).
            end_time: Absolute time to stop execution (nanoseconds).
            freq: Wake-up frequency in nanoseconds. Default 1 minute.
            lookback_period: Period for volume calculation. Can be a string like "1min"
            or nanoseconds. Defaults to freq converted to string.
            pov: Percentage of volume to target (0.0 to 1.0). Default 0.1 (10%).
            direction: Side.BID to buy, Side.ASK to sell.
            quantity: Total target quantity to execute.
            trade: If True, places actual orders. If False, only logs.
            name: Agent name.
            type: Agent type string.
            random_state: NumPy random state for reproducibility.
            log_orders: Whether to log order activity.
            risk_config: Optional RiskConfig for position/circuit-breaker limits.
        """
        super().__init__(
            id,
            name,
            type,
            random_state,
            starting_cash,
            log_orders,
            risk_config=risk_config,
        )
        self.start_time: NanosecondTime = start_time
        self.end_time: NanosecondTime = end_time
        self.freq: NanosecondTime = freq

        # Handle lookback_period - convert to string format for get_transacted_volume
        if lookback_period is None:
            self.lookback_period: str = self._ns_to_period_str(freq)
        elif isinstance(lookback_period, str):
            self.lookback_period: str = lookback_period
        else:
            self.lookback_period: str = self._ns_to_period_str(lookback_period)

        self.pov: float = pov
        self.direction: Side = direction
        self.quantity: int = quantity
        self.trade: bool = trade

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
        self.last_transacted_volume: int = 0

        # Logging and statistics
        self.execution_history: list = []
        self.total_market_volume_observed: int = 0

    @staticmethod
    def _ns_to_period_str(ns: NanosecondTime) -> str:
        """
        Convert nanoseconds to a period string format used by get_transacted_volume.

        Args:
            ns: Time in nanoseconds.

        Returns:
            Period string like "1min", "30s", "1h", etc.
        """
        # Convert to seconds
        seconds = ns / 1e9

        if seconds >= 3600 and seconds % 3600 == 0:
            return f"{int(seconds // 3600)}h"
        elif seconds >= 60 and seconds % 60 == 0:
            return f"{int(seconds // 60)}min"
        else:
            return f"{int(seconds)}s"

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        """Called when the simulation kernel starts."""
        super().kernel_starting(start_time)

        logger.info(
            f"{self.name}: POV Execution Agent starting. "
            f"Target: {self.quantity} shares at {self.pov * 100:.1f}% POV, "
            f"Direction: {self.direction.value}"
        )

    def kernel_stopping(self) -> None:
        """Called when the simulation kernel stops."""
        super().kernel_stopping()

        # Log final execution statistics
        execution_rate = (
            self.executed_quantity / self.quantity * 100 if self.quantity > 0 else 0
        )
        effective_pov = (
            self.executed_quantity / self.total_market_volume_observed * 100
            if self.total_market_volume_observed > 0
            else 0
        )

        logger.info(
            f"{self.name}: Execution complete. "
            f"Executed: {self.executed_quantity}/{self.quantity} shares "
            f"({execution_rate:.1f}% of target). "
            f"Effective POV: {effective_pov:.2f}%"
        )

        self.logEvent(
            "EXECUTION_SUMMARY",
            {
                "executed_quantity": self.executed_quantity,
                "target_quantity": self.quantity,
                "remaining_quantity": self.remaining_quantity,
                "execution_rate": execution_rate,
                "effective_pov": effective_pov,
                "total_market_volume": self.total_market_volume_observed,
            },
        )

    def wakeup(self, current_time: NanosecondTime) -> bool:
        """
        Handle agent wakeup.

        The agent queries market data and transacted volume, then places orders
        based on the POV strategy.

        Returns:
            True if the agent is ready to trade, False otherwise.
        """
        can_trade = super().wakeup(current_time)

        # Check if we should be trading
        if not self._should_be_trading(current_time):
            return can_trade

        # Mark execution as started
        if not self.execution_started:
            self.execution_started = True
            logger.info(f"{self.name}: Starting execution at {current_time}")

        # Check if execution is complete
        if self.execution_complete or self.remaining_quantity <= 0:
            logger.debug(f"{self.name}: Execution already complete")
            return can_trade

        # Query market data: get spread and transacted volume
        self.get_current_spread(self.symbol)
        self.get_transacted_volume(self.symbol, lookback_period=self.lookback_period)
        self.state = "AWAITING_MARKET_DATA"

        return can_trade

    def _should_be_trading(self, current_time: NanosecondTime) -> bool:
        """Check if the agent should be actively trading."""
        # Not yet time to start
        if current_time < self.start_time:
            self.set_wakeup(self.start_time)
            return False

        # Past end time
        if current_time >= self.end_time:
            if not self.execution_complete:
                self.execution_complete = True
                logger.info(
                    f"{self.name}: Execution period ended. "
                    f"Executed {self.executed_quantity}/{self.quantity}"
                )
            return False

        # Market hours check (handled by parent)
        if not self.mkt_open or not self.mkt_close:
            return False

        return not self.mkt_closed

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Process incoming messages from the exchange."""
        super().receive_message(current_time, sender_id, message)

        if self.state != "AWAITING_MARKET_DATA":
            return

        # Process spread response
        if isinstance(message, QuerySpreadResponseMsg):
            result = self.get_known_bid_ask(self.symbol)
            if len(result) == 4:
                bid, bid_vol, ask, ask_vol = result
                self.last_bid = bid
                self.last_ask = ask

                logger.debug(f"{self.name}: Spread update - Bid: {bid}, Ask: {ask}")

        # Process transacted volume response
        elif isinstance(message, QueryTransactedVolResponseMsg):
            bid_volume = message.bid_volume
            ask_volume = message.ask_volume
            total_volume = bid_volume + ask_volume

            self.last_transacted_volume = total_volume
            self.total_market_volume_observed += total_volume

            logger.debug(
                f"{self.name}: Volume update - "
                f"Bid Vol: {bid_volume}, Ask Vol: {ask_volume}, Total: {total_volume}"
            )

            # Now we have both spread and volume, execute the POV strategy
            self._execute_pov_strategy(current_time, total_volume)

    def _execute_pov_strategy(
        self, current_time: NanosecondTime, transacted_volume: int
    ) -> None:
        """
        Execute the POV strategy based on observed market volume.

        Calculates target order size as: min(pov * transacted_volume, remaining_quantity)
        """
        self.state = "EXECUTING"

        # Calculate target order size
        target_size = int(self.pov * transacted_volume)

        # Don't exceed remaining quantity
        order_size = min(target_size, self.remaining_quantity)

        # Minimum order size check
        if order_size <= 0:
            logger.debug(
                f"{self.name}: No order placed - "
                f"target_size={target_size}, remaining={self.remaining_quantity}"
            )
            self._schedule_next_wakeup(current_time)
            return

        # Log the execution decision
        self.logEvent(
            "POV_EXECUTION_DECISION",
            {
                "time": current_time,
                "transacted_volume": transacted_volume,
                "target_size": target_size,
                "order_size": order_size,
                "remaining_quantity": self.remaining_quantity,
                "direction": self.direction.value,
            },
        )

        # Place the order if trading is enabled
        if self.trade:
            self._place_execution_order(order_size)
        else:
            logger.info(
                f"{self.name}: Would place {self.direction.value} order "
                f"for {order_size} shares (trade=False)"
            )

        # Schedule next wakeup
        self._schedule_next_wakeup(current_time)

    def _place_execution_order(self, order_size: int) -> None:
        """
        Place an execution order.

        Uses market orders for aggressive execution to ensure fills.
        Could be extended to use limit orders for less market impact.
        """
        if order_size <= 0:
            return

        # Check if we have price data
        if self.direction == Side.BID and self.last_ask is None:
            logger.warning(f"{self.name}: No ask price available, skipping order")
            return
        if self.direction == Side.ASK and self.last_bid is None:
            logger.warning(f"{self.name}: No bid price available, skipping order")
            return

        # Place market order for aggressive execution
        # This ensures we participate at the target POV rate
        self.place_market_order(
            symbol=self.symbol,
            quantity=order_size,
            side=self.direction,
        )

        logger.info(
            f"{self.name}: Placed {self.direction.value} market order "
            f"for {order_size} shares"
        )

    def _schedule_next_wakeup(self, current_time: NanosecondTime) -> None:
        """Schedule the next wakeup time."""
        next_wakeup = current_time + self.freq

        # Don't schedule past end time
        if next_wakeup < self.end_time:
            self.set_wakeup(next_wakeup)
            self.state = "AWAITING_WAKEUP"
        else:
            self.state = "COMPLETE"

    def order_executed(self, order) -> None:
        """
        Handle order execution notification.

        Updates execution tracking when our orders are filled.
        """
        super().order_executed(order)

        # Track execution for our orders
        if order.agent_id == self.id:
            executed_qty = order.quantity
            self.executed_quantity += executed_qty
            self.remaining_quantity -= executed_qty

            self.execution_history.append(
                {
                    "time": self.current_time,
                    "quantity": executed_qty,
                    "side": order.side.value,
                    "fill_price": order.fill_price,
                }
            )

            logger.info(
                f"{self.name}: Order executed - {executed_qty} shares at "
                f"{order.fill_price}. Progress: {self.executed_quantity}/{self.quantity}"
            )

            # Check if execution is complete
            if self.remaining_quantity <= 0:
                self.execution_complete = True
                logger.info(f"{self.name}: Target quantity fully executed!")

    def get_wake_frequency(self) -> NanosecondTime:
        """Return the wake-up frequency for this agent."""
        return self.freq
