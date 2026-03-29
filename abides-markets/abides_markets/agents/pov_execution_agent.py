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

from ..messages.query import QueryTransactedVolResponseMsg
from ..models.risk_config import RiskConfig
from ..orders import Side
from .base_execution_agent import BaseSlicingExecutionAgent

logger = logging.getLogger(__name__)

_DEFAULT_FREQ: int = str_to_ns("1min")


class POVExecutionAgent(BaseSlicingExecutionAgent):
    """
    Percentage of Volume (POV) Execution Agent.

    This agent executes a target quantity of shares by participating at a specified
    percentage of the market's transacted volume. The agent:

    1. Periodically wakes up and queries the transacted volume over a lookback period
    2. Calculates its target execution size as: pov * transacted_volume
    3. Places market orders to achieve that target
    4. Continues until the total target quantity is executed or time expires

    This is a common execution algorithm used to minimize market impact while
    maintaining a consistent participation rate in the market.
    """

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
        super().__init__(
            id=id,
            symbol=symbol,
            starting_cash=starting_cash,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            direction=direction,
            quantity=quantity,
            trade=trade,
            order_style="market",
            name=name,
            type=type,
            random_state=random_state,
            log_orders=log_orders,
            risk_config=risk_config,
        )

        # Handle lookback_period - convert to string format for get_transacted_volume
        if lookback_period is None:
            self.lookback_period: str = self._ns_to_period_str(freq)
        elif isinstance(lookback_period, str):
            self.lookback_period = lookback_period
        else:
            self.lookback_period = self._ns_to_period_str(lookback_period)

        self.pov: float = pov

        # POV-specific tracking
        self.last_transacted_volume: int = 0
        self.total_market_volume_observed: int = 0

    # ------------------------------------------------------------------
    # POV helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ns_to_period_str(ns: NanosecondTime) -> str:
        """Convert nanoseconds to a period string for get_transacted_volume."""
        seconds = ns / 1e9
        if seconds >= 3600 and seconds % 3600 == 0:
            return f"{int(seconds // 3600)}h"
        elif seconds >= 60 and seconds % 60 == 0:
            return f"{int(seconds // 60)}min"
        else:
            return f"{int(seconds)}s"

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------
    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        logger.info(f"{self.name}: POV target {self.pov * 100:.1f}%")

    def kernel_stopping(self) -> None:
        effective_pov = (
            self.executed_quantity / self.total_market_volume_observed * 100
            if self.total_market_volume_observed > 0
            else 0
        )
        self.logEvent(
            "POV_SUMMARY",
            {
                "effective_pov": effective_pov,
                "total_market_volume": self.total_market_volume_observed,
            },
        )
        super().kernel_stopping()

    # ------------------------------------------------------------------
    # Market data: override to also query transacted volume
    # ------------------------------------------------------------------
    def _on_spread_received(self, current_time: NanosecondTime) -> None:
        """After getting spread, also query transacted volume."""
        self.get_transacted_volume(self.symbol, lookback_period=self.lookback_period)
        # Stay in AWAITING_MARKET_DATA until volume response arrives

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # Let the base class handle spread responses
        super().receive_message(current_time, sender_id, message)

        if self.state != "AWAITING_MARKET_DATA":
            return

        if isinstance(message, QueryTransactedVolResponseMsg):
            bid_volume = message.bid_volume
            ask_volume = message.ask_volume
            total_volume = bid_volume + ask_volume

            self.last_transacted_volume = total_volume
            self.total_market_volume_observed += total_volume

            # Now execute the slice
            self._execute_slice(current_time)

    # ------------------------------------------------------------------
    # Slice sizing: POV strategy
    # ------------------------------------------------------------------
    def _compute_slice_quantity(self, current_time: NanosecondTime) -> int:
        """Compute shares as pov * recent transacted volume."""
        return int(self.pov * self.last_transacted_volume)
