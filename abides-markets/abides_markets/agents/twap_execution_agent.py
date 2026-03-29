"""
TWAP (Time-Weighted Average Price) Execution Agent

Divides a parent order into uniform time slices and executes an equal share
in each slice.  Partial fills from earlier slices are redistributed across
remaining slices (catch-up logic).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from abides_core import NanosecondTime

from ..models.risk_config import RiskConfig
from ..orders import Side
from .base_execution_agent import BaseSlicingExecutionAgent

logger = logging.getLogger(__name__)


class TWAPExecutionAgent(BaseSlicingExecutionAgent):
    """Time-Weighted Average Price execution agent.

    At each wakeup the agent computes::

        slice_qty = remaining_quantity / slices_remaining

    so earlier under-fills are automatically caught up.  The last slice
    gets whatever is left.  Default ``order_style`` is ``"ioc_limit"``
    to minimise market impact.
    """

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
        order_style: str = "ioc_limit",
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
            order_style=order_style,  # type: ignore[arg-type]
            name=name,
            type=type,
            random_state=random_state,
            log_orders=log_orders,
            risk_config=risk_config,
        )
        # Total number of slices = ceil((end - start) / freq)
        duration = self.end_time - self.start_time
        self.total_slices: int = max(1, math.ceil(duration / self.freq))
        self.slices_completed: int = 0

    # ------------------------------------------------------------------
    # Slice sizing: uniform (with catch-up)
    # ------------------------------------------------------------------
    def _compute_slice_quantity(self, current_time: NanosecondTime) -> int:
        slices_remaining = self.total_slices - self.slices_completed
        self.slices_completed += 1

        if slices_remaining <= 1:
            return self.remaining_quantity

        return max(1, math.ceil(self.remaining_quantity / slices_remaining))
