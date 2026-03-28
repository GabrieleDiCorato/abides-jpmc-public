import logging

import numpy as np

from abides_core import Message, NanosecondTime
from abides_markets.models.order_size_model import OrderSizeModel
from abides_markets.models.risk_config import RiskConfig

from ..messages.query import QuerySpreadResponseMsg
from ..orders import Side
from .trading_agent import TradingAgent

logger = logging.getLogger(__name__)


class NoiseAgent(TradingAgent):
    """Simplest trading agent — reference implementation for new agent authors.

    Wakes once at a random time, requests the current spread from the exchange,
    and places a single random limit order at the bid or ask.

    **Architecture notes for agent developers:**

    1. Agents are event-driven: the ONLY entry points are ``wakeup()`` and
       ``receive_message()``.  There are no loops.  To do something later,
       call ``self.set_wakeup(future_time)``.

    2. Market data (bid/ask/trades) is NOT available until asynchronously
       received.  ``get_current_spread()`` sends a message to the exchange;
       the reply arrives later via ``receive_message()`` as a
       ``QuerySpreadResponseMsg``.

    3. Every agent needs a state machine to track what it's waiting for.
       This agent uses a simple string state (validated by ``VALID_STATES``
       on TradingAgent).  On each ``wakeup()`` or ``receive_message()``
       the agent checks its state to decide what to do next.
    """

    # Declare the set of valid states.  Assigning any other string to
    # self.state will raise ValueError immediately (catches typos).
    VALID_STATES = frozenset(
        {"AWAITING_WAKEUP", "INACTIVE", "AWAITING_SPREAD", "ACTIVE"}
    )

    def __init__(
        self,
        id: int,
        wakeup_time: NanosecondTime,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        symbol: str = "IBM",
        starting_cash: int = 100000,
        log_orders: bool = False,
        order_size_model: OrderSizeModel | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        # --- Call super().__init__ with the standard set of base-class params.
        # id, name, type, random_state are auto-injected by the config system;
        # starting_cash and log_orders come from BaseAgentConfig fields;
        # risk_config is bundled from position_limit / max_drawdown / ... fields.
        # You do NOT need to declare these in your own config model.
        super().__init__(
            id,
            name,
            type,
            random_state,
            starting_cash,
            log_orders,
            risk_config=risk_config,
        )

        # --- Strategy-specific state (YOUR config model declares these) ---

        self.wakeup_time: NanosecondTime = wakeup_time
        self.symbol: str = symbol

        # Tracks whether we've passed the pre-market phase (exchange discovery).
        self.trading: bool = False

        # State machine — starts waiting for the first wakeup from the kernel.
        self.state: str = "AWAITING_WAKEUP"

        self.prev_wake_time: NanosecondTime | None = None

        self.size: int | None = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model

    # -----------------------------------------------------------------
    # Lifecycle hooks — called by the kernel in order
    # -----------------------------------------------------------------

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        # TradingAgent.kernel_starting() discovers the ExchangeAgent and
        # subscribes to market hours.  Always call super().
        super().kernel_starting(start_time)

    def kernel_stopping(self) -> None:
        # Called when the simulation ends.  Log final P&L.
        super().kernel_stopping()

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if bid is None and ask is None and self.symbol not in self.last_trade:
            # Agent never received any market data — log starting cash as-is.
            self.logEvent("FINAL_VALUATION", self.starting_cash, True)
        else:
            H = self.get_holdings(self.symbol)
            rT = (bid + ask) // 2 if bid and ask else self.last_trade[self.symbol]

            surplus_cents = rT * H + self.holdings["CASH"] - self.starting_cash
            surplus_frac = surplus_cents / self.starting_cash

            self.logEvent("FINAL_VALUATION", surplus_frac, True)

            logger.debug(
                "{} final report.  Holdings: {}, end cash: {}, start cash: {}, final fundamental: {}, surplus: {}",
                self.name,
                H,
                self.holdings["CASH"],
                self.starting_cash,
                rT,
                surplus_frac,
            )

    # -----------------------------------------------------------------
    # wakeup() — the main entry point, triggered by the kernel scheduler
    # -----------------------------------------------------------------

    def wakeup(self, current_time: NanosecondTime) -> None:
        # CRITICAL: super().wakeup() returns False when market hours are
        # unknown or the market is closed.  Always guard on the return value.
        if not super().wakeup(current_time):
            return

        self.state = "INACTIVE"

        if not self.trading:
            self.trading = True
            logger.debug("{} is ready to start trading now.", self.name)

        # --- Market closed: request final price then stop ---
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            return

        # --- Not yet time to trade: reschedule ---
        if self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            return

        # --- Market just closed, need closing price ---
        if self.mkt_closed and self.symbol not in self.daily_close_price:
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        # --- Normal path: request current spread ---
        # get_current_spread() sends a QuerySpreadMsg to the exchange.
        # The response will arrive in receive_message() as a
        # QuerySpreadResponseMsg.  We CANNOT use the data yet — we
        # must transition to AWAITING_SPREAD and return.
        if isinstance(self, NoiseAgent):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
        else:
            self.state = "ACTIVE"

    # -----------------------------------------------------------------
    # place_order() — trading logic (called when data is ready)
    # -----------------------------------------------------------------

    def place_order(self) -> None:
        buy = bool(self.random_state.randint(0, 2))

        # get_known_bid_ask() reads from the cache populated by the last
        # QuerySpreadResponseMsg.  Returns (bid, bid_vol, ask, ask_vol)
        # — any of which may be None if the book side is empty.
        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        # Guard: bid or ask may be None (empty book side).
        if self.size > 0:
            if buy and ask:
                self.place_limit_order(self.symbol, self.size, Side.BID, ask)
            elif not buy and bid:
                self.place_limit_order(self.symbol, self.size, Side.ASK, bid)

    # -----------------------------------------------------------------
    # receive_message() — asynchronous data arrival
    # -----------------------------------------------------------------

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        # CRITICAL: super().receive_message() updates portfolio state, cached
        # bids/asks, and processes market-hours messages.  Always call first.
        super().receive_message(current_time, sender_id, message)

        # State-machine dispatch: check if this message is the one we were
        # waiting for.  Use isinstance() to match the message type.
        if self.state == "AWAITING_SPREAD" and isinstance(
            message, QuerySpreadResponseMsg
        ):
            if self.mkt_closed:
                return

            # NOW the spread data is available — safe to place an order.
            self.place_order()
            self.state = "AWAITING_WAKEUP"

    # -----------------------------------------------------------------
    # get_wake_frequency() — used by TradingAgent for default scheduling
    # -----------------------------------------------------------------

    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=0, high=100)
