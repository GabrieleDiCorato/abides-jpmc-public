import logging
import warnings
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from abides_core import Kernel, Message, NanosecondTime

from ..messages.market import (
    MarketClosedMsg,
    MarketClosePriceMsg,
    MarketClosePriceRequestMsg,
    MarketHoursMsg,
    MarketHoursRequestMsg,
)
from ..messages.marketdata import (
    BookImbalanceDataMsg,
    BookImbalanceSubReqMsg,
    L1DataMsg,
    L1SubReqMsg,
    L2DataMsg,
    L2SubReqMsg,
    L3DataMsg,
    L3SubReqMsg,
    MarketDataEventMsg,
    MarketDataSubReqMsg,
    TransactedVolDataMsg,
    TransactedVolSubReqMsg,
)
from ..messages.order import (
    CancelOrderMsg,
    LimitOrderMsg,
    MarketOrderMsg,
    ModifyOrderMsg,
    OrderMsg,
    PartialCancelOrderMsg,
    ReplaceOrderMsg,
)
from ..messages.orderbook import OrderAcceptedMsg, OrderCancelledMsg, OrderExecutedMsg
from ..messages.query import (
    QueryLastTradeMsg,
    QueryLastTradeResponseMsg,
    QueryMsg,
    QueryOrderStreamMsg,
    QueryOrderStreamResponseMsg,
    QuerySpreadMsg,
    QuerySpreadResponseMsg,
    QueryTransactedVolMsg,
    QueryTransactedVolResponseMsg,
)
from ..order_book import OrderBook
from ..orders import Side
from .financial_agent import FinancialAgent

logger = logging.getLogger(__name__)


class ExchangeAgent(FinancialAgent):
    """
    The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp
    to open and close trading, a list of equity symbols for which it should create order
    books, a frequency at which to archive snapshots of its order books, a pipeline
    delay (in ns) for order activity, the exchange computation delay (in ns), the levels
    of order stream history to maintain per symbol (maintains all orders that led to the
    last N trades), whether to log all order activity to the agent log, and a random
    state object (already seeded) to use for stochasticity.
    """

    @dataclass
    class MetricTracker(ABC):  # noqa: B024
        # droupout metrics
        total_time_no_liquidity_asks: int = 0
        total_time_no_liquidity_bids: int = 0
        pct_time_no_liquidity_asks: float = 0
        pct_time_no_liquidity_bids: float = 0

        # exchanged volume
        total_exchanged_volume: int = 0

        # last trade
        last_trade: int | None = 0
        # can be extended

    @dataclass
    class BaseDataSubscription(ABC):
        """
        Base class for all types of data subscription registered with this agent.
        """

        agent_id: int
        last_update_ts: int

    @dataclass
    class FrequencyBasedSubscription(BaseDataSubscription, ABC):
        """
        Base class for all types of data subscription that are sent from this agent
        at a fixed, regular frequency.
        """

        freq: int

    @dataclass
    class L1DataSubscription(FrequencyBasedSubscription):
        pass

    @dataclass
    class L2DataSubscription(FrequencyBasedSubscription):
        depth: int

    @dataclass
    class L3DataSubscription(FrequencyBasedSubscription):
        depth: int

    @dataclass
    class TransactedVolDataSubscription(FrequencyBasedSubscription):
        lookback: str

    @dataclass
    class EventBasedSubscription(BaseDataSubscription, ABC):
        """
        Base class for all types of data subscription that are sent from this agent
        when triggered by an event or specific circumstance.
        """

        event_in_progress: bool

    @dataclass
    class BookImbalanceDataSubscription(EventBasedSubscription):
        # Properties:
        min_imbalance: float
        # State:
        imbalance: float | None = None
        side: Side | None = None

    # ---- dispatch table: message type → handler method name ----
    _MESSAGE_HANDLERS: dict[type, str] = {
        MarketHoursRequestMsg: "_handle_market_hours_request",
        MarketClosePriceRequestMsg: "_handle_market_close_price_request",
        QueryLastTradeMsg: "_handle_query_last_trade",
        QuerySpreadMsg: "_handle_query_spread",
        QueryOrderStreamMsg: "_handle_query_order_stream",
        QueryTransactedVolMsg: "_handle_query_transacted_vol",
        LimitOrderMsg: "_handle_limit_order",
        MarketOrderMsg: "_handle_market_order",
        CancelOrderMsg: "_handle_cancel_order",
        PartialCancelOrderMsg: "_handle_partial_cancel_order",
        ModifyOrderMsg: "_handle_modify_order",
        ReplaceOrderMsg: "_handle_replace_order",
    }

    def __init__(
        self,
        id: int,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: list[str],
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        book_logging: bool = True,
        book_log_depth: int = 10,
        pipeline_delay: int = 40000,
        computation_delay: int = 1,
        stream_history: int = 0,
        log_orders: bool = False,
        use_metric_tracker: bool = True,
    ) -> None:
        super().__init__(id, name, type, random_state)

        # symbols
        self.symbols = symbols

        # Do not request repeated wakeup calls.
        self.reschedule: bool = False

        # Store this exchange's open and close times.
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close

        # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
        # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
        # hours, etc).
        self.pipeline_delay: int = pipeline_delay

        # Computation delay is applied on every wakeup call or message received.
        self.computation_delay: int = computation_delay

        # The exchange maintains an order stream of all orders leading to the last L trades
        # to support certain agents from the auction literature (GD, HBL, etc).
        self.stream_history: int = stream_history

        self.book_logging: bool = book_logging
        self.book_log_depth: int = book_log_depth

        # Log all order activity?
        self.log_orders: bool = log_orders

        # Create an order book for each symbol.
        self.order_books: dict[str, OrderBook] = {
            symbol: OrderBook(self, symbol) for symbol in symbols
        }

        if use_metric_tracker:
            # Create a metric tracker for each symbol.
            self.metric_trackers: dict[str, ExchangeAgent.MetricTracker] = {
                symbol: self.MetricTracker() for symbol in symbols
            }
        else:
            self.metric_trackers: dict[str, ExchangeAgent.MetricTracker] = {}

        # The subscription dict is a dictionary with the key = agent ID,
        # value = dict (key = symbol, value = list [levels (no of levels to recieve updates for),
        # frequency (min number of ns between messages), last agent update timestamp]
        # e.g. {101 : {'AAPL' : [1, 10, NanosecondTime(10:00:00)}}
        self.data_subscriptions: defaultdict[
            str, list[ExchangeAgent.BaseDataSubscription]
        ] = defaultdict(list)

        # Store a list of agents who have requested market close price information.
        # (this is most likely all agents)
        self.market_close_price_subscriptions: list[int] = []

    def kernel_initializing(self, kernel: "Kernel") -> None:
        """
        The exchange agent overrides this to obtain a reference to an oracle.

        This is needed to establish a "last trade price" at open (i.e. an opening
        price) in case agents query last trade before any simulated trades are made.
        This can probably go away once we code the opening cross auction.

        Arguments:
          kernel: The ABIDES kernel that this agent instance belongs to.
        """

        super().kernel_initializing(kernel)

        assert self.kernel is not None

        self.oracle = self.kernel.oracle

        # Obtain opening prices (in integer cents).  These are not noisy right now.
        for symbol in self.order_books:
            try:
                self.order_books[symbol].last_trade = self.oracle.get_daily_open_price(
                    symbol, self.mkt_open
                )
                logger.debug(
                    f"Opening price for {symbol} is {self.order_books[symbol].last_trade}"
                )
            except AttributeError as e:
                logger.debug(str(e))

        # Set a wakeup for the market close so we can send market close price messages.
        self.set_wakeup(self.mkt_close)

    def kernel_terminating(self) -> None:
        """
        The exchange agent overrides this to additionally log the full depth of its
        order books for the entire day.
        """

        super().kernel_terminating()

        # symbols, write them to disk.
        for symbol in self.symbols:
            self.analyse_order_book(symbol)

        self.total_exchanged_volume = 0
        for symbol in self.symbols:
            bid_volume, ask_volume = self.order_books[symbol].get_transacted_volume(
                self.current_time - self.mkt_open
            )
            self.metric_trackers[symbol].total_exchanged_volume = (
                bid_volume + ask_volume
            )
            self.total_exchanged_volume += bid_volume + ask_volume
            self.metric_trackers[symbol].last_trade = self.order_books[
                symbol
            ].last_trade

        if not self.log_orders:
            return

        # If the oracle supports writing the fundamental value series for its
        # symbols, write them to disk.
        if self.oracle.f_log:
            for symbol in self.oracle.f_log:
                dfFund = pd.DataFrame(self.oracle.f_log[symbol])
                if not dfFund.empty:
                    dfFund.set_index("FundamentalTime", inplace=True)
                    self.write_log(dfFund, filename=f"fundamental_{symbol}")
                    logger.debug("Fundamental archival complete.")

    def wakeup(self, current_time: NanosecondTime):
        super().wakeup(current_time)

        # If we have reached market close, send market close price messages to all agents
        # that requested them.
        if current_time >= self.mkt_close:
            message = MarketClosePriceMsg(
                {symbol: book.last_trade for symbol, book in self.order_books.items()}
            )

            for agent in self.market_close_price_subscriptions:
                self.send_message(agent, message)

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """
        Arguments:
            current_time:
            sender_id:
            message:
        """

        super().receive_message(current_time, sender_id, message)

        # Unless the intent of an experiment is to examine computational issues
        # within an Exchange, it will typically have either 1 ns delay (near
        # instant but cannot process multiple orders in the same atomic time unit)
        # or 0 ns delay (can process any number of orders, always in the atomic
        # time unit in which they are received).  This is separate from, and
        # additional to, any parallel pipeline delay imposed for order book
        # activity.

        # Note that computation delay MUST be updated before any calls to send_message.
        self.set_computation_delay(self.computation_delay)

        # Is the exchange closed?  (This block only affects post-close, not pre-open.)
        if current_time > self.mkt_close:
            # Most messages after close will receive a 'MKT_CLOSED' message in
            # response.  A few things might still be processed, like requests
            # for final trade prices or such.
            if isinstance(message, OrderMsg):
                if isinstance(message, ModifyOrderMsg):
                    logger.debug(
                        f"{self.name} received {message.type()}: OLD: {message.old_order} NEW: {message.new_order}"
                    )
                else:
                    logger.debug(
                        # All the other order messages have an 'order' attribute. The hierarchy is not well designed.
                        f"{self.name} received {message.type()}: {message.order}"  # type: ignore
                    )

                self.send_message(sender_id, MarketClosedMsg())

                # Don't do any further processing on these messages!
                return
            elif isinstance(message, QueryMsg):
                # Specifically do allow querying after market close, so agents can get the
                # final trade of the day as their "daily close" price for a symbol.
                pass
            else:
                logger.debug(
                    f"{self.name} received {message.type()}, discarded: market is closed."
                )
                self.send_message(sender_id, MarketClosedMsg())

                # Don't do any further processing on these messages!
                return

        if isinstance(message, OrderMsg):
            # Log order messages only if that option is configured.  Log all other messages.
            if self.log_orders:
                if isinstance(message, (ModifyOrderMsg, ReplaceOrderMsg)):
                    self.logEvent(
                        message.type(),
                        message.new_order.to_dict(),
                        deepcopy_event=False,
                    )
                else:
                    self.logEvent(
                        # TODO All other message types have an 'order' attribute.
                        message.type(),
                        message.order.to_dict(),
                        deepcopy_event=False,  # type: ignore
                    )
        else:
            self.logEvent(message.type(), message)

        if isinstance(message, MarketDataSubReqMsg):
            # Handle the DATA SUBSCRIPTION request and cancellation messages from the agents.
            if message.symbol not in self.order_books:
                return

            if message.cancel:
                logger.debug(
                    f"{self.name} received MarketDataSubscriptionCancellation request from agent {sender_id}"
                )

                # Map message types to subscription types for correct matching
                _msg_sub_map = {
                    L1SubReqMsg: self.L1DataSubscription,
                    L2SubReqMsg: self.L2DataSubscription,
                    L3SubReqMsg: self.L3DataSubscription,
                    TransactedVolSubReqMsg: self.TransactedVolDataSubscription,
                    BookImbalanceSubReqMsg: self.BookImbalanceDataSubscription,
                }
                expected_sub_type = _msg_sub_map.get(type(message))

                for data_sub in self.data_subscriptions[message.symbol]:
                    if (
                        data_sub.agent_id == sender_id
                        and isinstance(data_sub, expected_sub_type)
                        and getattr(data_sub, "freq", None)
                        == getattr(message, "freq", None)
                        and getattr(data_sub, "depth", None)
                        == getattr(message, "depth", None)
                    ):
                        self.data_subscriptions[message.symbol].remove(data_sub)
                        break

            else:
                logger.debug(
                    f"{self.name} received MarketDataSubscriptionRequest request from agent {sender_id}"
                )

                if isinstance(message, L1SubReqMsg):
                    sub = self.L1DataSubscription(sender_id, current_time, message.freq)
                elif isinstance(message, L2SubReqMsg):
                    sub = self.L2DataSubscription(
                        sender_id, current_time, message.freq, message.depth
                    )
                elif isinstance(message, L3SubReqMsg):
                    sub = self.L3DataSubscription(
                        sender_id, current_time, message.freq, message.depth
                    )
                elif isinstance(message, TransactedVolSubReqMsg):
                    sub = self.TransactedVolDataSubscription(
                        sender_id, current_time, message.freq, message.lookback
                    )
                elif isinstance(message, BookImbalanceSubReqMsg):
                    sub = self.BookImbalanceDataSubscription(
                        sender_id, current_time, False, message.min_imbalance
                    )
                else:
                    raise Exception

                self.data_subscriptions[message.symbol].append(sub)

        # O(1) dispatch for the main message-handling chain.
        handler = self._MESSAGE_HANDLERS.get(type(message))
        if handler is not None:
            getattr(self, handler)(sender_id, current_time, message)

    def _handle_market_hours_request(
        self,
        sender_id: int,
        current_time: NanosecondTime,
        message: MarketHoursRequestMsg,
    ) -> None:
        logger.debug(
            f"{self.name} received market hours request from agent {sender_id}"
        )
        self.set_computation_delay(0)
        self.send_message(sender_id, MarketHoursMsg(self.mkt_open, self.mkt_close))

    def _handle_market_close_price_request(
        self,
        sender_id: int,
        current_time: NanosecondTime,
        message: MarketClosePriceRequestMsg,
    ) -> None:
        self.market_close_price_subscriptions.append(sender_id)

    def _handle_query_last_trade(
        self, sender_id: int, current_time: NanosecondTime, message: QueryLastTradeMsg
    ) -> None:
        symbol = message.symbol
        if symbol not in self.order_books:
            warnings.warn(f"Last trade request discarded. Unknown symbol: {symbol}")
        else:
            logger.debug(
                f"{self.name} received QUERY_LAST_TRADE ({symbol}) request from agent {sender_id}"
            )
            self.send_message(
                sender_id,
                QueryLastTradeResponseMsg(
                    symbol=symbol,
                    last_trade=self.order_books[symbol].last_trade,
                    mkt_closed=current_time > self.mkt_close,
                ),
            )

    def _handle_query_spread(
        self, sender_id: int, current_time: NanosecondTime, message: QuerySpreadMsg
    ) -> None:
        symbol = message.symbol
        depth = message.depth
        if symbol not in self.order_books:
            warnings.warn(f"Bid-ask spread request discarded. Unknown symbol: {symbol}")
        else:
            logger.debug(
                f"{self.name} received QUERY_SPREAD ({symbol}:{depth}) request from agent {sender_id}"
            )
            self.send_message(
                sender_id,
                QuerySpreadResponseMsg(
                    symbol=symbol,
                    depth=depth,
                    bids=self.order_books[symbol].get_l2_bid_data(depth),
                    asks=self.order_books[symbol].get_l2_ask_data(depth),
                    last_trade=self.order_books[symbol].last_trade,
                    mkt_closed=current_time > self.mkt_close,
                ),
            )

    def _handle_query_order_stream(
        self, sender_id: int, current_time: NanosecondTime, message: QueryOrderStreamMsg
    ) -> None:
        symbol = message.symbol
        length = message.length
        if symbol not in self.order_books:
            warnings.warn(f"Order stream request discarded. Unknown symbol: {symbol}")
        else:
            logger.debug(
                f"{self.name} received QUERY_ORDER_STREAM ({symbol}:{length}) request from agent {sender_id}"
            )
            self.send_message(
                sender_id,
                QueryOrderStreamResponseMsg(
                    symbol=symbol,
                    length=length,
                    orders=self.order_books[symbol].history[1 : length + 1],
                    mkt_closed=current_time > self.mkt_close,
                ),
            )

    def _handle_query_transacted_vol(
        self,
        sender_id: int,
        current_time: NanosecondTime,
        message: QueryTransactedVolMsg,
    ) -> None:
        symbol = message.symbol
        lookback_period = message.lookback_period
        if symbol not in self.order_books:
            warnings.warn(
                f"Transacted volume request discarded. Unknown symbol: {symbol}"
            )
        else:
            logger.debug(
                f"{self.name} received QUERY_TRANSACTED_VOLUME ({symbol}:{lookback_period}) request from agent {sender_id}"
            )
            bid_volume, ask_volume = self.order_books[symbol].get_transacted_volume(
                lookback_period
            )
            self.send_message(
                sender_id,
                QueryTransactedVolResponseMsg(
                    symbol=symbol,
                    bid_volume=bid_volume,
                    ask_volume=ask_volume,
                    mkt_closed=current_time > self.mkt_close,
                ),
            )

    def _handle_limit_order(
        self, sender_id: int, current_time: NanosecondTime, message: LimitOrderMsg
    ) -> None:
        logger.debug(f"{self.name} received LIMIT_ORDER: {message.order}")
        if message.order.symbol not in self.order_books:
            warnings.warn(
                f"Limit Order discarded. Unknown symbol: {message.order.symbol}"
            )
        else:
            # The TradingAgent already stores its own deepcopy; the message's
            # order reference is not retained elsewhere, so a second deepcopy
            # is unnecessary.
            self.order_books[message.order.symbol].handle_limit_order(message.order)
            self.publish_order_book_data(message.order.symbol)

    def _handle_market_order(
        self, sender_id: int, current_time: NanosecondTime, message: MarketOrderMsg
    ) -> None:
        logger.debug(f"{self.name} received MARKET_ORDER: {message.order}")
        if message.order.symbol not in self.order_books:
            warnings.warn(
                f"Market Order discarded. Unknown symbol: {message.order.symbol}"
            )
        else:
            self.order_books[message.order.symbol].handle_market_order(message.order)
            self.publish_order_book_data(message.order.symbol)

    def _handle_cancel_order(
        self, sender_id: int, current_time: NanosecondTime, message: CancelOrderMsg
    ) -> None:
        tag = message.tag
        metadata = message.metadata
        logger.debug(f"{self.name} received CANCEL_ORDER: {message.order}")
        if message.order.symbol not in self.order_books:
            warnings.warn(
                f"Cancellation request discarded. Unknown symbol: {message.order.symbol}"
            )
        else:
            self.order_books[message.order.symbol].cancel_order(
                message.order, tag, metadata
            )
            self.publish_order_book_data(message.order.symbol)

    def _handle_partial_cancel_order(
        self,
        sender_id: int,
        current_time: NanosecondTime,
        message: PartialCancelOrderMsg,
    ) -> None:
        tag = message.tag
        metadata = message.metadata
        logger.debug(
            f"{self.name} received PARTIAL_CANCEL_ORDER: {message.order}, new order: {message.quantity}"
        )
        if message.order.symbol not in self.order_books:
            warnings.warn(
                f"Modification request discarded. Unknown symbol: {message.order.symbol}"
            )
        else:
            self.order_books[message.order.symbol].partial_cancel_order(
                message.order, message.quantity, tag, metadata
            )
            self.publish_order_book_data(message.order.symbol)

    def _handle_modify_order(
        self, sender_id: int, current_time: NanosecondTime, message: ModifyOrderMsg
    ) -> None:
        old_order = message.old_order
        new_order = message.new_order
        logger.debug(
            f"{self.name} received MODIFY_ORDER: {old_order}, new order: {new_order}"
        )
        if old_order.symbol not in self.order_books:
            warnings.warn(
                f"Modification request discarded. Unknown symbol: {old_order.symbol}"
            )
        else:
            self.order_books[old_order.symbol].modify_order(old_order, new_order)
            self.publish_order_book_data(old_order.symbol)

    def _handle_replace_order(
        self, sender_id: int, current_time: NanosecondTime, message: ReplaceOrderMsg
    ) -> None:
        agent_id = message.agent_id
        order = message.old_order
        new_order = message.new_order
        logger.debug(
            f"{self.name} received REPLACE_ORDER: {order}, new order: {new_order}"
        )
        if order.symbol not in self.order_books:
            warnings.warn(
                f"Replacement request discarded. Unknown symbol: {order.symbol}"
            )
        else:
            self.order_books[order.symbol].replace_order(agent_id, order, new_order)
            self.publish_order_book_data(order.symbol)

    def publish_order_book_data(self, symbol: str) -> None:
        """
        The exchange agents sends an order book update to the agents using the
        subscription API if one of the following conditions are met:

        1) agent requests ALL order book updates (freq == 0)
        2) order book update timestamp > last time agent was updated AND the orderbook
           update time stamp is greater than the last agent update time stamp by a
           period more than that specified in the freq parameter.

        Arguments:
            symbol: Only publish updates for this symbol's subscriptions.
        """

        data_subs = self.data_subscriptions.get(symbol, [])
        if not data_subs:
            return

        book = self.order_books[symbol]

        for data_sub in data_subs:
            if isinstance(data_sub, self.FrequencyBasedSubscription):
                messages = self.handle_frequency_based_data_subscription(
                    symbol, data_sub
                )
            elif isinstance(data_sub, self.EventBasedSubscription):
                messages = self.handle_event_based_data_subscription(symbol, data_sub)
            else:
                raise Exception("Got invalid data subscription object")

            for message in messages:
                self.send_message(data_sub.agent_id, message)

            if len(messages) > 0:
                data_sub.last_update_ts = book.last_update_ts

    def handle_frequency_based_data_subscription(
        self, symbol: str, data_sub: "ExchangeAgent.FrequencyBasedSubscription"
    ) -> list[Message]:
        book = self.order_books[symbol]

        if (book.last_update_ts - data_sub.last_update_ts) < data_sub.freq:
            return []

        messages = []

        if isinstance(data_sub, self.L1DataSubscription):
            bid = book.get_l1_bid_data()
            ask = book.get_l1_ask_data()
            messages.append(
                L1DataMsg(symbol, book.last_trade, self.current_time, bid, ask)
            )

        elif isinstance(data_sub, self.L2DataSubscription):
            bids = book.get_l2_bid_data(data_sub.depth)
            asks = book.get_l2_ask_data(data_sub.depth)
            messages.append(
                L2DataMsg(
                    symbol,
                    book.last_trade,
                    self.current_time,
                    bids,
                    asks,
                )
            )

        elif isinstance(data_sub, self.L3DataSubscription):
            bids = book.get_l3_bid_data(data_sub.depth)
            asks = book.get_l3_ask_data(data_sub.depth)
            messages.append(
                L3DataMsg(
                    symbol,
                    book.last_trade,
                    self.current_time,
                    bids,
                    asks,
                )
            )

        elif isinstance(data_sub, self.TransactedVolDataSubscription):
            bid_volume, ask_volume = book.get_transacted_volume(data_sub.lookback)
            messages.append(
                TransactedVolDataMsg(
                    symbol,
                    book.last_trade,
                    self.current_time,
                    bid_volume,
                    ask_volume,
                )
            )

        else:
            raise Exception("Got invalid data subscription object")

        return messages

    def handle_event_based_data_subscription(
        self, symbol: str, data_sub: "ExchangeAgent.EventBasedSubscription"
    ) -> list[Message]:
        book = self.order_books[symbol]
        messages = []

        if isinstance(data_sub, self.BookImbalanceDataSubscription):
            imbalance, side = book.get_imbalance()

            event_in_progress = imbalance > data_sub.min_imbalance

            # 4 different combinations of current state vs. new state to consider:
            if data_sub.event_in_progress and event_in_progress:
                # Event in progress --> Event in progress
                if side != data_sub.side:
                    # If imbalance flips from one side of the market to the other in one step

                    # Close current event
                    messages.append(
                        BookImbalanceDataMsg(
                            symbol,
                            book.last_trade,
                            self.current_time,
                            MarketDataEventMsg.Stage.FINISH,
                            data_sub.imbalance,
                            data_sub.side,
                        )
                    )

                    # Start new event
                    data_sub.event_in_progress = True
                    data_sub.side = side
                    data_sub.imbalance = imbalance
                    messages.append(
                        BookImbalanceDataMsg(
                            symbol,
                            book.last_trade,
                            self.current_time,
                            MarketDataEventMsg.Stage.START,
                            imbalance,
                            side,
                        )
                    )

            elif data_sub.event_in_progress and not event_in_progress:
                # Event in progress --> Event not in progress
                data_sub.event_in_progress = False
                data_sub.side = None
                data_sub.imbalance = None
                messages.append(
                    BookImbalanceDataMsg(
                        symbol,
                        book.last_trade,
                        self.current_time,
                        MarketDataEventMsg.Stage.FINISH,
                        imbalance,
                        side,
                    )
                )

            elif not data_sub.event_in_progress and event_in_progress:
                # Event not in progress --> Event in progress
                data_sub.event_in_progress = True
                data_sub.side = side
                data_sub.imbalance = imbalance
                messages.append(
                    BookImbalanceDataMsg(
                        symbol,
                        book.last_trade,
                        self.current_time,
                        MarketDataEventMsg.Stage.START,
                        imbalance,
                        side,
                    )
                )

            elif not data_sub.event_in_progress and not event_in_progress:
                # Event not in progress --> Event not in progress
                pass

        else:
            raise Exception("Got invalid data subscription object")

        return messages

    def logL2style(self, symbol: str) -> tuple[list, list] | None:
        book = self.order_books[symbol]
        if not book.book_log2:
            return None
        tmp = book.book_log2
        times = []
        booktop = []
        for t in tmp:
            times.append(t["QuoteTime"])
            booktop.append([t["bids"], t["asks"]])
        return (times, booktop)

    def send_message(self, recipient_id: int, message: Message) -> None:
        """
        Arguments:
            recipient_id:
            message:
        """

        # The ExchangeAgent automatically applies appropriate parallel processing pipeline delay
        # to those message types which require it.
        # TODO: probably organize the order types into categories once there are more, so we can
        # take action by category (e.g. ORDER-related messages) instead of enumerating all message
        # types to be affected.
        if isinstance(message, (OrderAcceptedMsg, OrderCancelledMsg, OrderExecutedMsg)):
            # Messages that require order book modification (not simple queries) incur the additional
            # parallel processing delay as configured.
            super().send_message(recipient_id, message, delay=self.pipeline_delay)
            if self.log_orders:
                self.logEvent(message.type(), message.order.to_dict())
        else:
            # Other message types incur only the currently-configured computation delay for this agent.
            super().send_message(recipient_id, message)

    def analyse_order_book(self, symbol: str):
        # will grow with time
        book = self.order_books[symbol].book_log2
        self.get_time_dropout(book, symbol)

    def get_time_dropout(self, book: list[dict[str, Any]], symbol: str):
        # Calculate simulation duration
        start_time = self.mkt_open
        end_time = self.mkt_close
        total_time = end_time - start_time

        T_null_bids = 0
        T_null_asks = 0

        # Handle edge cases or bad config
        if total_time <= 0:
            total_time = 1  # Avoid division by zero
        elif len(book) == 0:
            # No updates means no liquidity for the entire duration (assuming empty start)
            T_null_bids = total_time
            T_null_asks = total_time
        else:
            df = pd.DataFrame(book)
            times = df["QuoteTime"].values

            # Initial period (mkt_open to first update)
            # Assumption: Book starts empty
            if times[0] > start_time:
                T_null_bids += times[0] - start_time
                T_null_asks += times[0] - start_time

            # Iterate through intervals between updates
            # During interval [t_i, t_{i+1}), the state is determined by updates at t_i
            # We access book[i] directly to avoid DataFrame overhead/issues with list cells
            for i in range(len(times) - 1):
                duration = times[i + 1] - times[i]
                if duration > 0:
                    if len(book[i]["bids"]) == 0:
                        T_null_bids += duration
                    if len(book[i]["asks"]) == 0:
                        T_null_asks += duration

            # Final period (last update to mkt_close)
            if end_time > times[-1]:
                duration = end_time - times[-1]
                if len(book[-1]["bids"]) == 0:
                    T_null_bids += duration
                if len(book[-1]["asks"]) == 0:
                    T_null_asks += duration

        self.metric_trackers[symbol] = self.MetricTracker(
            total_time_no_liquidity_asks=T_null_asks / 1e9,
            total_time_no_liquidity_bids=T_null_bids / 1e9,
            pct_time_no_liquidity_asks=100 * T_null_asks / total_time,
            pct_time_no_liquidity_bids=100 * T_null_bids / total_time,
        )
