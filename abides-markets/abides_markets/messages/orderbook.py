from abc import ABC
from dataclasses import dataclass

from abides_core import Message

from ..orders import LimitOrder, Order, StopOrder


@dataclass
class OrderBookMsg(Message, ABC):
    pass


@dataclass
class OrderAcceptedMsg(OrderBookMsg):
    order: LimitOrder


@dataclass
class OrderExecutedMsg(OrderBookMsg):
    order: Order


@dataclass
class OrderCancelledMsg(OrderBookMsg):
    order: LimitOrder


@dataclass
class OrderPartialCancelledMsg(OrderBookMsg):
    new_order: LimitOrder


@dataclass
class OrderModifiedMsg(OrderBookMsg):
    new_order: LimitOrder


@dataclass
class OrderReplacedMsg(OrderBookMsg):
    old_order: LimitOrder
    new_order: LimitOrder


@dataclass
class StopTriggeredMsg(OrderBookMsg):
    """Sent to the agent when their stop order has been triggered.

    The ``order`` field carries the original ``StopOrder``.  The
    resulting market order is submitted automatically by the exchange.
    """

    order: StopOrder
