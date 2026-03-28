from collections import deque

import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns

from ...messages.marketdata import L2SubReqMsg, MarketDataMsg
from ...messages.query import QuerySpreadResponseMsg
from ...models.risk_config import RiskConfig
from ...orders import Side
from ..trading_agent import TradingAgent

_DEFAULT_WAKE_UP_FREQ: int = str_to_ns("60s")


class MomentumAgent(TradingAgent):
    """
    Simple Trading Agent that compares the short-window past mid-price observations with the long-window past
    observations and places a buy limit order if the short MA >= long MA or a
    sell limit order if the short MA < long MA.
    """

    VALID_STATES = frozenset(
        {"AWAITING_WAKEUP", "AWAITING_SPREAD", "AWAITING_MARKET_DATA"}
    )

    def __init__(
        self,
        id: int,
        symbol,
        starting_cash,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        min_size=20,
        max_size=50,
        wake_up_freq: NanosecondTime = _DEFAULT_WAKE_UP_FREQ,
        poisson_arrival=True,
        order_size_model=None,
        subscribe=False,
        log_orders=False,
        short_window: int = 20,
        long_window: int = 50,
        risk_config: RiskConfig | None = None,
    ) -> None:
        if short_window < 1 or long_window < 1:
            raise ValueError(
                f"short_window ({short_window}) and long_window ({long_window}) must be >= 1."
            )
        if short_window > long_window:
            raise ValueError(
                f"short_window ({short_window}) must be <= long_window ({long_window})."
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
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = (
            self.random_state.randint(self.min_size, self.max_size)
            if order_size_model is None
            else None
        )
        self.order_size_model = order_size_model  # Probabilistic model for order size
        self.wake_up_freq = wake_up_freq
        self.poisson_arrival = poisson_arrival  # Whether to arrive as a Poisson process
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.short_window: int = short_window
        self.long_window: int = long_window
        self.mid_list: deque[float] = deque(maxlen=long_window)
        self.avg_short_list: list[float] = []
        self.avg_long_list: list[float] = []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Agent wakeup is determined by self.wake_up_freq"""
        can_trade = super().wakeup(current_time)
        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=int(10e9),
                    depth=1,
                )
            )
            self.subscription_requested = True
            self.state = "AWAITING_MARKET_DATA"
        elif can_trade and not self.subscribe:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Momentum agent actions are determined after obtaining the best bid and ask in the LOB"""
        super().receive_message(current_time, sender_id, message)
        if (
            not self.subscribe
            and self.state == "AWAITING_SPREAD"
            and isinstance(message, QuerySpreadResponseMsg)
        ):
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            self.place_orders(bid, ask)
            self.set_wakeup(current_time + self.get_wake_frequency())
            self.state = "AWAITING_WAKEUP"
        elif (
            self.subscribe
            and self.state == "AWAITING_MARKET_DATA"
            and isinstance(message, MarketDataMsg)
        ):
            bids, asks = self.known_bids.get(self.symbol, []), self.known_asks.get(
                self.symbol, []
            )
            if bids and asks:
                self.place_orders(bids[0][0], asks[0][0])
            self.state = "AWAITING_MARKET_DATA"

    def place_orders(self, bid: int, ask: int) -> None:
        """Momentum Agent actions logic"""
        if bid is not None and ask is not None:
            self.mid_list.append((bid + ask) // 2)
            if len(self.mid_list) >= self.short_window:
                self.avg_short_list.append(
                    int(round(MomentumAgent.ma(self.mid_list, n=self.short_window)[-1]))
                )
            if len(self.mid_list) >= self.long_window:
                self.avg_long_list.append(
                    int(round(MomentumAgent.ma(self.mid_list, n=self.long_window)[-1]))
                )
            if len(self.avg_short_list) > 0 and len(self.avg_long_list) > 0:
                if self.order_size_model is not None:
                    self.size = self.order_size_model.sample(
                        random_state=self.random_state
                    )

                if self.size > 0:
                    if self.avg_short_list[-1] >= self.avg_long_list[-1]:
                        self.place_limit_order(
                            self.symbol,
                            quantity=self.size,
                            side=Side.BID,
                            limit_price=ask,
                        )
                    else:
                        self.place_limit_order(
                            self.symbol,
                            quantity=self.size,
                            side=Side.ASK,
                            limit_price=bid,
                        )

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n
