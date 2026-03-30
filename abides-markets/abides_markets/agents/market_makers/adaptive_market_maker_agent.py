from __future__ import annotations

import logging
from math import ceil, floor

import numpy as np

from abides_core import Message, NanosecondTime

from ...messages.marketdata import (
    BookImbalanceDataMsg,
    BookImbalanceSubReqMsg,
    L2SubReqMsg,
    MarketDataEventMsg,
    MarketDataMsg,
)
from ...messages.query import QuerySpreadResponseMsg, QueryTransactedVolResponseMsg
from ...models.risk_config import RiskConfig
from ...orders import LimitOrder, Side
from ...utils import sigmoid
from ..trading_agent import TradingAgent

ANCHOR_TOP_STR = "top"
ANCHOR_BOTTOM_STR = "bottom"
ANCHOR_MIDDLE_STR = "middle"

ADAPTIVE_SPREAD_STR = "adaptive"
INITIAL_SPREAD_VALUE = 50


logger = logging.getLogger(__name__)


class AdaptiveMarketMakerAgent(TradingAgent):
    """This class implements a modification of the Chakraborty-Kearns `ladder` market-making strategy, wherein the
    the size of order placed at each level is set as a fraction of measured transacted volume in the previous time
    period.

    Can skew orders to size of current inventory using beta parameter, whence beta == 0 represents inventory being
    ignored and beta == infinity represents all liquidity placed on one side of book.

    ADAPTIVE SPREAD: the market maker's spread can be set either as a fixed or value or can be adaptive,
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        pov: float = 0.025,
        min_order_size: int = 1,
        window_size: (
            int | str
        ) = "adaptive",  # size in ticks or 'adaptive'. Will be converted in internal int | None
        anchor: str = ANCHOR_MIDDLE_STR,
        num_ticks: int = 10,
        level_spacing: float = 5.0,
        wake_up_freq: NanosecondTime = 1_000_000_000,  # 1 second
        poisson_arrival: bool = True,
        subscribe: bool = False,
        subscribe_freq: int = 10_000_000_000,
        subscribe_num_levels: int = 1,
        cancel_limit_delay: int = 50,
        skew_beta=0,
        price_skew_param=4,
        spread_alpha: float = 0.75,
        backstop_quantity: int = 0,
        log_orders: bool = False,
        min_imbalance=0.9,
        risk_config: RiskConfig | None = None,
        flatten_before_close_ns: NanosecondTime | None = 300_000_000_000,
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
        self.is_adaptive: bool = False
        self.symbol: str = symbol  # Symbol traded
        self.pov: float = (
            pov  # fraction of transacted volume placed at each price level
        )
        self.min_order_size: int = (
            min_order_size  # minimum size order to place at each level, if pov <= min
        )
        self.anchor: str = self.validate_anchor(
            anchor
        )  # anchor either top of window or bottom of window to mid-price
        self.window_size: int | None = self.validate_window_size(
            window_size
        )  # Size in ticks (cents) of how wide the window around mid price is. If equal to
        # string 'adaptive' then ladder starts at best bid and ask
        self.num_ticks: int = (
            num_ticks  # number of ticks on each side of window in which to place liquidity
        )
        self.level_spacing: float = (
            level_spacing  #  level spacing as a fraction of the spread
        )
        self.wake_up_freq: NanosecondTime = wake_up_freq  # Frequency of agent wake up
        self.poisson_arrival: bool = (
            poisson_arrival  # Whether to arrive as a Poisson process
        )
        if self.poisson_arrival:
            self.arrival_rate = self.wake_up_freq

        self.subscribe: bool = (
            subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        )
        self.subscribe_freq: int = (
            subscribe_freq  # Frequency in nanoseconds at which to receive market updates
        )
        # in subscribe mode
        self.min_imbalance = min_imbalance
        self.subscribe_num_levels: int = (
            subscribe_num_levels  # Number of orderbook levels in subscription mode
        )
        self.cancel_limit_delay: int = (
            cancel_limit_delay  # delay in nanoseconds between order cancellations and new limit order placements
        )

        self.skew_beta = (
            skew_beta  # parameter for determining order placement imbalance
        )
        self.price_skew_param = (
            price_skew_param  # parameter determining how much to skew price level.
        )
        self.spread_alpha: float = (
            spread_alpha  # parameter for exponentially weighted moving average of spread. 1 corresponds to ignoring old values, 0 corresponds to no updates
        )
        self.backstop_quantity: int = (
            backstop_quantity  # how many orders to place at outside order level, to prevent liquidity dropouts. If None then place same as at other levels.
        )

        self.has_subscribed = False

        ## Internal variables

        self.subscription_requested: bool = False
        self.state: dict[str, bool] = self.initialise_state()
        self.buy_order_size: int = self.min_order_size
        self.sell_order_size: int = self.min_order_size

        self.last_mid: int | None = None  # last observed mid price
        self.last_spread: float = (
            INITIAL_SPREAD_VALUE  # last observed spread moving average
        )
        self.tick_size: int | None = (
            None if self.is_adaptive else ceil(self.last_spread * self.level_spacing)
        )
        self.LIQUIDITY_DROPOUT_WARNING: str = (
            f"Liquidity dropout for agent {self.name}."
        )

        self.two_side: bool = (
            self.price_skew_param is not None
        )  # switch to control self.get_transacted_volume
        # method

        self.flatten_before_close_ns: NanosecondTime | None = flatten_before_close_ns
        self._flattened: bool = False

    def initialise_state(self) -> dict[str, bool]:
        """Returns variables that keep track of whether spread and transacted volume have been observed."""

        if self.subscribe:
            return {"AWAITING_MARKET_DATA": True, "AWAITING_TRANSACTED_VOLUME": True}
        else:
            return {"AWAITING_SPREAD": True, "AWAITING_TRANSACTED_VOLUME": True}

    def validate_anchor(self, anchor: str) -> str:
        """Checks that input parameter anchor takes allowed value, raises ``ValueError`` if not.

        Arguments:
            anchor:

        Returns:
            The anchor if validated.
        """

        if anchor not in [ANCHOR_TOP_STR, ANCHOR_BOTTOM_STR, ANCHOR_MIDDLE_STR]:
            raise ValueError(
                f"Variable anchor must take the value `{ANCHOR_BOTTOM_STR}`, `{ANCHOR_MIDDLE_STR}` or "
                f"`{ANCHOR_TOP_STR}`"
            )
        else:
            return anchor

    def validate_window_size(self, window_size: int | str) -> int | None:
        """Checks that input parameter window_size takes allowed value, raises ``ValueError`` if not.

        Arguments:
            window_size:

        Returns:
            The window_size if validated
        """

        if isinstance(window_size, int):
            return window_size
        elif isinstance(window_size, str):
            if window_size.lower() == ADAPTIVE_SPREAD_STR:
                self.is_adaptive = True
                self.anchor = ANCHOR_MIDDLE_STR
                return None
        else:
            raise ValueError(
                f"Variable window_size must be of type int or string {ADAPTIVE_SPREAD_STR}."
            )

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)

    def wakeup(self, current_time: NanosecondTime):
        """Agent wakeup is determined by self.wake_up_freq."""

        can_trade = super().wakeup(current_time)

        # --- End-of-day flatten ---
        if (
            not self._flattened
            and self.flatten_before_close_ns is not None
            and self.mkt_close is not None
            and current_time >= self.mkt_close - self.flatten_before_close_ns
        ):
            self._flatten_position(current_time)
            return

        if not self.has_subscribed:
            super().request_data_subscription(
                BookImbalanceSubReqMsg(
                    symbol=self.symbol,
                    min_imbalance=self.min_imbalance,
                )
            )
            self.last_time_book_order = current_time
            self.has_subscribed = True

        if self.subscribe and not self.subscription_requested:
            super().request_data_subscription(
                L2SubReqMsg(
                    symbol=self.symbol,
                    freq=self.subscribe_freq,
                    depth=self.subscribe_num_levels,
                )
            )
            self.subscription_requested = True
            self.get_transacted_volume(self.symbol, lookback_period=self.subscribe_freq)
            self.state = self.initialise_state()

        elif can_trade and not self.subscribe:
            self.get_current_spread(self.symbol, depth=self.subscribe_num_levels)
            self.get_transacted_volume(self.symbol, lookback_period=self.wake_up_freq)
            self.state = self.initialise_state()

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """Processes message from exchange.

        Main function is to update orders in orderbook relative to mid-price.

        Arguments:
            current_time: Simulation current time.
            message: Message received by self from ExchangeAgent.
        """

        super().receive_message(current_time, sender_id, message)

        mid = None
        if self.last_mid is not None:
            mid = self.last_mid

        if self.last_spread is not None and self.is_adaptive:
            self._adaptive_update_window_and_tick_size()

        if (
            isinstance(message, QueryTransactedVolResponseMsg)
            and self.state["AWAITING_TRANSACTED_VOLUME"] is True
        ):
            self.update_order_size()
            self.state["AWAITING_TRANSACTED_VOLUME"] = False

        if (
            isinstance(message, BookImbalanceDataMsg)
            and message.stage == MarketDataEventMsg.Stage.START
            and mid is not None
        ):
            self.place_orders(mid)
            self.last_time_book_order = current_time

        if not self.subscribe:
            if (
                isinstance(message, QuerySpreadResponseMsg)
                and self.state["AWAITING_SPREAD"] is True
            ):
                bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
                if bid and ask:
                    mid = (ask + bid) // 2
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = ask - bid
                        self._adaptive_update_spread(spread)

                    self.state["AWAITING_SPREAD"] = False
                else:
                    logger.debug("SPREAD MISSING at time {}", current_time)
                    self.state["AWAITING_SPREAD"] = (
                        False  # use last mid price and spread
                    )

            if (
                self.state["AWAITING_SPREAD"] is False
                and self.state["AWAITING_TRANSACTED_VOLUME"] is False
                and mid is not None
            ):
                self.place_orders(mid)
                self.state = self.initialise_state()
                self.set_wakeup(current_time + self.get_wake_frequency())

        else:  # subscription mode
            if (
                isinstance(message, MarketDataMsg)
                and self.state["AWAITING_MARKET_DATA"] is True
            ):
                bids = self.known_bids.get(self.symbol, [])
                asks = self.known_asks.get(self.symbol, [])
                bid = bids[0][0] if bids else None
                ask = asks[0][0] if asks else None
                if bid and ask:
                    mid = (ask + bid) // 2
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = ask - bid
                        self._adaptive_update_spread(spread)

                    self.state["AWAITING_MARKET_DATA"] = False
                else:
                    logger.debug("SPREAD MISSING at time {}", current_time)
                    self.state["AWAITING_MARKET_DATA"] = False

            if (
                self.state["AWAITING_MARKET_DATA"] is False
                and self.state["AWAITING_TRANSACTED_VOLUME"] is False
            ):
                self.place_orders(mid)
                self.state = self.initialise_state()

    def _adaptive_update_spread(self, spread) -> None:
        """Update internal spread estimate with exponentially weighted moving average.

        Arguments:
            spread
        """

        spread_ewma = (
            self.spread_alpha * spread + (1 - self.spread_alpha) * self.last_spread
        )
        self.window_size = spread_ewma
        self.last_spread = spread_ewma

    def _adaptive_update_window_and_tick_size(self) -> None:
        """Update window size and tick size relative to internal spread estimate."""

        self.window_size = self.last_spread
        self.tick_size = round(self.level_spacing * self.window_size)
        if self.tick_size == 0:
            self.tick_size = 1

    def update_order_size(self) -> None:
        """Updates size of order to be placed."""

        buy_transacted_volume = self.transacted_volume[self.symbol][0]
        sell_transacted_volume = self.transacted_volume[self.symbol][1]
        total_transacted_volume = buy_transacted_volume + sell_transacted_volume

        qty = round(self.pov * total_transacted_volume)

        if self.skew_beta == 0:  # ignore inventory
            self.buy_order_size = (
                qty if qty >= self.min_order_size else self.min_order_size
            )
            self.sell_order_size = (
                qty if qty >= self.min_order_size else self.min_order_size
            )
        else:
            holdings = self.get_holdings(self.symbol)
            proportion_sell = sigmoid(holdings, self.skew_beta)
            sell_size = ceil(proportion_sell * qty)
            buy_size = floor((1 - proportion_sell) * qty)

            self.buy_order_size = (
                buy_size if buy_size >= self.min_order_size else self.min_order_size
            )
            self.sell_order_size = (
                sell_size if sell_size >= self.min_order_size else self.min_order_size
            )

    def compute_orders_to_place(self, mid: int) -> tuple[list[int], list[int]]:
        """Given a mid price, computes the orders that need to be removed from
        orderbook, and adds these orders to bid and ask deques.

        Arguments:
            mid: Mid price.
        """

        if self.price_skew_param is None:
            mid_point = mid
        else:
            buy_transacted_volume = self.transacted_volume[self.symbol][0]
            sell_transacted_volume = self.transacted_volume[self.symbol][1]

            if (buy_transacted_volume == 0) and (sell_transacted_volume == 0):
                mid_point = mid
            else:
                # trade imbalance, +1 means all transactions are buy, -1 means all transactions are sell
                trade_imbalance = (
                    2
                    * buy_transacted_volume
                    / (buy_transacted_volume + sell_transacted_volume)
                ) - 1
                mid_point = int(mid + (trade_imbalance * self.price_skew_param))

        if self.anchor == ANCHOR_MIDDLE_STR:
            highest_bid = int(mid_point) - floor(0.5 * self.window_size)
            lowest_ask = int(mid_point) + ceil(0.5 * self.window_size)
        elif self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid_point - 1)
            lowest_ask = int(mid_point + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid_point - self.window_size)
            lowest_ask = int(mid_point + 1)

        lowest_bid = highest_bid - ((self.num_ticks - 1) * self.tick_size)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * self.tick_size)

        # Clamp to positive prices — a wide spread or large tick_size can
        # push lowest_bid below zero, which the exchange rejects.
        bids_to_place = [
            price
            for price in range(lowest_bid, highest_bid + self.tick_size, self.tick_size)
            if price >= 1
        ]
        asks_to_place = [
            price
            for price in range(lowest_ask, highest_ask + self.tick_size, self.tick_size)
            if price >= 1
        ]

        return bids_to_place, asks_to_place

    def place_orders(self, mid: int) -> None:
        """Given a mid-price, compute new orders that need to be placed, then
        send the orders to the Exchange.

        Uses replace_order for existing open orders to halve exchange message
        traffic compared to the previous cancel-all + place-new pattern.

        Arguments:
            mid: Mid price.
        """

        bid_prices, ask_prices = self.compute_orders_to_place(mid)

        # Build list of desired (side, price, size) tuples.
        desired: list[tuple[Side, int, int]] = []

        if self.backstop_quantity != 0:
            desired.append((Side.BID, bid_prices[0], self.backstop_quantity))
            bid_prices = bid_prices[1:]
            desired.append((Side.ASK, ask_prices[-1], self.backstop_quantity))
            ask_prices = ask_prices[:-1]

        for bp in bid_prices:
            desired.append((Side.BID, bp, self.buy_order_size))
        for ap in ask_prices:
            desired.append((Side.ASK, ap, self.sell_order_size))

        # Partition existing open LimitOrders by side, sorted by price.
        existing_bids = sorted(
            (
                o
                for o in self.orders.values()
                if isinstance(o, LimitOrder) and o.side.is_bid()
            ),
            key=lambda o: o.limit_price,
        )
        existing_asks = sorted(
            (
                o
                for o in self.orders.values()
                if isinstance(o, LimitOrder) and not o.side.is_bid()
            ),
            key=lambda o: o.limit_price,
        )

        desired_bids = [(p, sz) for s, p, sz in desired if s.is_bid()]
        desired_asks = [(p, sz) for s, p, sz in desired if not s.is_bid()]

        new_orders: list[LimitOrder] = []

        # Diff existing vs desired per side.
        self._diff_and_replace(existing_bids, desired_bids, Side.BID, new_orders)
        self._diff_and_replace(existing_asks, desired_asks, Side.ASK, new_orders)

        if new_orders:
            self.place_multiple_orders(new_orders)

    def _diff_and_replace(
        self,
        existing: list[LimitOrder],
        desired: list[tuple[int, int]],
        side: Side,
        new_orders: list[LimitOrder],
    ) -> None:
        """Pair existing orders with desired orders positionally.

        - Matching pairs with different price/size: replace_order
        - Matching pairs with identical price and size: skip (no message)
        - Surplus existing: cancel_order
        - Surplus desired: accumulate into new_orders for batch placement
        """

        n_common = min(len(existing), len(desired))

        for i in range(n_common):
            old = existing[i]
            want_price, want_size = desired[i]
            if old.limit_price == want_price and old.quantity == want_size:
                continue  # identical — no message needed
            new = self.create_limit_order(self.symbol, want_size, side, want_price)
            self.replace_order(old, new)

        # Cancel surplus existing orders.
        for i in range(n_common, len(existing)):
            self.cancel_order(existing[i])

        # Place surplus desired orders (will be batched).
        for i in range(n_common, len(desired)):
            want_price, want_size = desired[i]
            new_orders.append(
                self.create_limit_order(self.symbol, want_size, side, want_price)
            )

    def _flatten_position(self, current_time: NanosecondTime) -> None:
        """Cancel all outstanding quotes and place market orders to zero out
        the position in ``self.symbol``.  Called once when the flatten window
        is reached; sets ``self._flattened = True`` so it cannot fire again.
        """
        self._flattened = True
        self.cancel_all_orders()

        position = self.get_holdings(self.symbol)
        if position > 0:
            self.place_market_order(self.symbol, abs(position), Side.ASK)
        elif position < 0:
            self.place_market_order(self.symbol, abs(position), Side.BID)

        self.logEvent(
            "AMM_FLATTEN",
            {"symbol": self.symbol, "position_closed": position},
            deepcopy_event=False,
        )
        logger.info(
            "%s flattened position: %d shares of %s at %s",
            self.name,
            position,
            self.symbol,
            current_time,
        )

    def get_wake_frequency(self) -> NanosecondTime:
        if not self.poisson_arrival:
            return self.wake_up_freq
        else:
            delta_time = self.random_state.exponential(scale=self.arrival_rate)
            return int(round(delta_time))
