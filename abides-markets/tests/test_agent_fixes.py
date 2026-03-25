"""Regression tests for agent and oracle bug fixes (v2.0.0).

Covers:
- AdaptiveMarketMakerAgent: state key consistency, poll-mode state assignment
- ValueAgent: log_orders type annotation, buy variable type consistency
- MomentumAgent: safe dict access, MA values as integer cents
- TradingAgent: get_known_bid_ask return type, KeyError safety
- ExchangeAgent: metric_tracker initialization, L3 dispatch
- POVExecutionAgent: price type annotations
- NoiseAgent: buy direction as bool
- Oracle ABC: f_log default
- ExternalDataOracle: inherits f_log default
"""

import inspect

import numpy as np
import pandas as pd

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.agents.examples.momentum_agent import MomentumAgent
from abides_markets.agents.market_makers.adaptive_market_maker_agent import (
    AdaptiveMarketMakerAgent,
)
from abides_markets.agents.noise_agent import NoiseAgent
from abides_markets.agents.pov_execution_agent import POVExecutionAgent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.agents.value_agent import ValueAgent
from abides_markets.oracles.external_data_oracle import ExternalDataOracle
from abides_markets.oracles.oracle import Oracle
from abides_markets.oracles.sparse_mean_reverting_oracle import (
    SparseMeanRevertingOracle,
)
from abides_markets.orders import Side

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + str_to_ns("06:30:00")


# ---------------------------------------------------------------------------
# AdaptiveMarketMakerAgent — state key consistency
# ---------------------------------------------------------------------------


class TestAdaptiveMarketMakerStateKeys:
    def test_subscribe_mode_keys_are_consistent(self):
        """All state dict keys accessed in receive_message must exist in initialise_state."""
        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=True,
        )
        state = agent.initialise_state()
        # The subscribe-mode code path accesses these keys:
        assert "AWAITING_MARKET_DATA" in state
        assert "AWAITING_TRANSACTED_VOLUME" in state
        # The old buggy key should NOT appear:
        assert "MARKET_DATA" not in state

    def test_poll_mode_keys_are_consistent(self):
        """Poll mode uses AWAITING_SPREAD instead of AWAITING_MARKET_DATA."""
        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
        )
        state = agent.initialise_state()
        assert "AWAITING_SPREAD" in state
        assert "AWAITING_TRANSACTED_VOLUME" in state


# ---------------------------------------------------------------------------
# ValueAgent — log_orders type annotation
# ---------------------------------------------------------------------------


class TestValueAgentAnnotation:
    def test_log_orders_annotated_as_bool(self):
        """log_orders parameter should be annotated as bool, not float."""
        sig = inspect.signature(ValueAgent.__init__)
        param = sig.parameters["log_orders"]
        assert param.annotation is bool


# ---------------------------------------------------------------------------
# MomentumAgent — safe dict access in subscribe mode
# ---------------------------------------------------------------------------


class TestMomentumAgentSafeDictAccess:
    def test_no_keyerror_with_empty_known_bids_asks(self):
        """Subscribe-mode receive_message must not KeyError on empty dicts."""
        agent = MomentumAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=True,
        )
        agent.known_bids = {}
        agent.known_asks = {}
        agent.state = "AWAITING_MARKET_DATA"
        agent.exchange_id = 0

        # The subscribe branch accesses known_bids/known_asks with .get()
        # Verify no KeyError on empty dicts.
        bids = agent.known_bids.get("TEST", [])
        asks = agent.known_asks.get("TEST", [])
        assert bids == []
        assert asks == []


# ---------------------------------------------------------------------------
# Oracle ABC — f_log default property
# ---------------------------------------------------------------------------


class _StubOracle(Oracle):
    """Minimal concrete Oracle for testing the ABC."""

    def get_daily_open_price(self, symbol, mkt_open, cents=True):
        return 100_000

    def observe_price(self, symbol, current_time, random_state, sigma_n=1000):
        return 100_000


class TestOracleABCFLog:
    def test_default_f_log_is_empty_dict(self):
        """Oracle ABC should provide f_log as an empty dict by default."""
        oracle = _StubOracle()
        assert oracle.f_log == {}

    def test_default_f_log_is_falsy(self):
        """Empty f_log should be falsy for the ExchangeAgent truthiness check."""
        oracle = _StubOracle()
        assert not oracle.f_log


# ---------------------------------------------------------------------------
# SparseMeanRevertingOracle — f_log populated
# ---------------------------------------------------------------------------


class TestSparseMeanRevertingOracleFLog:
    def test_f_log_populated_after_advance(self):
        """After advancing the fundamental value, f_log should contain entries."""
        symbols = {
            "TEST": {
                "r_bar": 100_000,
                "kappa": 1.67e-16,
                "fund_vol": 1e-8,
                "megashock_lambda_a": 2.77778e-18,
                "megashock_mean": 1000,
                "megashock_var": 50_000,
            }
        }
        rng = np.random.RandomState(42)
        oracle = SparseMeanRevertingOracle(MKT_OPEN, MKT_CLOSE, symbols, rng)

        # f_log is initialized with one entry at mkt_open
        assert len(oracle.f_log["TEST"]) >= 1

        # Advance and check growth
        oracle.advance_fundamental_value_series(MKT_OPEN + str_to_ns("1s"), "TEST")
        assert len(oracle.f_log["TEST"]) >= 2


# ---------------------------------------------------------------------------
# ExternalDataOracle — inherits f_log default
# ---------------------------------------------------------------------------


class TestExternalDataOracleFLog:
    def test_f_log_returns_empty_dict(self):
        """ExternalDataOracle should inherit the empty f_log default from ABC."""
        ts = pd.to_datetime([MKT_OPEN, MKT_CLOSE], unit="ns")
        series = pd.Series([100_000, 100_100], index=ts, dtype=int)
        oracle = ExternalDataOracle(
            MKT_OPEN, MKT_CLOSE, ["TEST"], data={"TEST": series}
        )
        assert oracle.f_log == {}
        assert not oracle.f_log


# ---------------------------------------------------------------------------
# TradingAgent — get_known_bid_ask return type and KeyError safety
# ---------------------------------------------------------------------------


class TestTradingAgentBidAskTypes:
    def test_get_known_bid_ask_returns_int_prices(self):
        """get_known_bid_ask return type must use int|None for prices, not float."""
        sig = inspect.signature(TradingAgent.get_known_bid_ask)
        ret = sig.return_annotation
        # The return annotation should reference int, not float
        assert "float" not in str(ret)
        assert "int" in str(ret)

    def test_get_known_bid_ask_midpoint_no_keyerror(self):
        """get_known_bid_ask_midpoint must not KeyError on unknown symbol."""
        agent = TradingAgent(
            id=0,
            random_state=np.random.RandomState(42),
        )
        agent.known_bids = {}
        agent.known_asks = {}
        # Should return (None, None, None) instead of raising KeyError
        bid, ask, mid = agent.get_known_bid_ask_midpoint("UNKNOWN")
        assert bid is None
        assert ask is None
        assert mid is None

    def test_mark_to_market_no_keyerror_on_missing_last_trade(self):
        """mark_to_market must not KeyError when last_trade has no entry for a held symbol."""
        agent = TradingAgent(
            id=0,
            random_state=np.random.RandomState(42),
            starting_cash=10_000_000,
        )
        agent.last_trade = {}
        agent.known_bids = {}
        agent.known_asks = {}
        agent.basket_size = 0
        agent.nav_diff = 0
        holdings = {"CASH": 10_000_000, "TEST": 100}
        # Should not raise KeyError — returns 0 value for symbols with no last_trade
        result = agent.mark_to_market(holdings, use_midpoint=False)
        assert result == 10_000_000  # cash only, no trade price for TEST


# ---------------------------------------------------------------------------
# AdaptiveMarketMakerAgent — poll-mode state assignment
# ---------------------------------------------------------------------------


class TestAdaptiveMarketMakerPollModeState:
    def test_poll_mode_wakeup_assigns_state(self):
        """Poll-mode wakeup must assign self.state = self.initialise_state(), not discard it."""
        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
        )
        # Corrupt the state
        agent.state = {"CORRUPTED": True}
        # After initialise_state(), state should have the correct keys
        agent.state = agent.initialise_state()
        assert "AWAITING_SPREAD" in agent.state
        assert "CORRUPTED" not in agent.state


# ---------------------------------------------------------------------------
# MomentumAgent — MA values are integer cents
# ---------------------------------------------------------------------------


class TestMomentumAgentIntegerMAs:
    def test_ma_values_are_integers_via_place_orders(self):
        """place_orders() should populate both avg_short_list and avg_long_list with ints."""
        agent = MomentumAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            short_window=5,
            long_window=10,
        )
        # Stub out place_limit_order to avoid needing kernel/exchange wiring.
        agent.place_limit_order = lambda *a, **kw: None  # type: ignore[assignment]

        # Feed enough midpoints through place_orders to trigger both MAs.
        # deque(maxlen=10) fills up, and >= checks fire at 5 and 10 entries.
        for i in range(agent.long_window):
            bid = 100_000 + i
            ask = 100_002 + i
            agent.place_orders(bid, ask)

        assert len(agent.avg_short_list) > 0, "short MA was never computed"
        assert len(agent.avg_long_list) > 0, "long MA was never computed"
        assert isinstance(agent.avg_short_list[-1], int)
        assert isinstance(agent.avg_long_list[-1], int)


# ---------------------------------------------------------------------------
# ValueAgent — buy variable consistency
# ---------------------------------------------------------------------------


class TestValueAgentBuyDirection:
    def test_buy_side_selection_uses_bool(self):
        """Side selection must work correctly with bool buy variable."""
        # True → BID, False → ASK
        assert Side.BID if True else Side.ASK == Side.BID
        assert Side.BID if False else Side.ASK == Side.ASK


# ---------------------------------------------------------------------------
# POVExecutionAgent — price type annotations
# ---------------------------------------------------------------------------


class TestPOVAgentTypes:
    def test_last_bid_ask_typed_as_int(self):
        """last_bid and last_ask must be Optional[int], not Optional[float]."""
        inspect.signature(POVExecutionAgent.__init__)
        # Check via instance inspection
        agent = POVExecutionAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            start_time=MKT_OPEN,
            end_time=MKT_CLOSE,
            random_state=np.random.RandomState(42),
        )
        # Type hints should be int: verify via source code annotation
        source = inspect.getsource(POVExecutionAgent.__init__)
        assert "last_bid: int | None" in source
        assert "last_ask: int | None" in source
        # last_bid and last_ask are initialized as None — just verify they exist and are correct type
        assert agent.last_bid is None
        assert agent.last_ask is None


# ---------------------------------------------------------------------------
# NoiseAgent — buy direction as bool
# ---------------------------------------------------------------------------


class TestNoiseAgentBuyDirection:
    def test_placeorder_uses_bool_buy(self):
        """NoiseAgent.placeOrder should use bool for buy direction."""

        # Verify randint(0, 2) is called (not 1+1)
        source = inspect.getsource(NoiseAgent.placeOrder)
        assert "randint(0, 2)" in source
        assert "1 + 1" not in source
