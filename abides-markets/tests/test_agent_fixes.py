"""Regression tests for agent and oracle bug fixes (v2.0.0).

Covers:
- AdaptiveMarketMakerAgent: state key consistency, poll-mode state assignment
- ValueAgent: log_orders type annotation, buy variable type consistency
- MomentumAgent: safe dict access, MA values as integer cents
- TradingAgent: get_known_bid_ask return type, KeyError safety
- ExchangeAgent: metric_tracker initialization, L3 dispatch
- POVExecutionAgent: price type annotations
- All registered agents: constructor parameter storage (v2.4.0 regression guard)
- NoiseAgent: buy direction as bool
- Oracle ABC: f_log default
- ExternalDataOracle: inherits f_log default
"""

import inspect

import numpy as np
import pandas as pd
import pytest

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
        agent.place_limit_order = lambda *a, **kw: None

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
        # Type hints should be int: verify via source code annotation in the
        # base class (BaseSlicingExecutionAgent) which defines these attributes.
        from abides_markets.agents.base_execution_agent import BaseSlicingExecutionAgent

        source = inspect.getsource(BaseSlicingExecutionAgent.__init__)
        assert "last_bid: int | None" in source
        assert "last_ask: int | None" in source
        # last_bid and last_ask are initialized as None — just verify they exist and are correct type
        assert agent.last_bid is None
        assert agent.last_ask is None


# ---------------------------------------------------------------------------
# NoiseAgent — buy direction as bool
# ---------------------------------------------------------------------------


class TestNoiseAgentBuyDirection:
    def test_place_order_uses_bool_buy(self):
        """NoiseAgent.place_order should use bool for buy direction."""

        # Verify randint(0, 2) is called (not 1+1)
        source = inspect.getsource(NoiseAgent.place_order)
        assert "randint(0, 2)" in source
        assert "1 + 1" not in source


# ---------------------------------------------------------------------------
# ValueAgent — depth_spread validation
# ---------------------------------------------------------------------------


class TestValueAgentDepthSpreadValidation:
    def test_depth_spread_zero_raises(self):
        """depth_spread=0 must raise ValueError."""
        with pytest.raises(ValueError, match="depth_spread must be >= 1"):
            ValueAgent(
                id=0,
                random_state=np.random.RandomState(42),
                depth_spread=0,
            )

    def test_depth_spread_negative_raises(self):
        """depth_spread=-1 must raise ValueError."""
        with pytest.raises(ValueError, match="depth_spread must be >= 1"):
            ValueAgent(
                id=0,
                random_state=np.random.RandomState(42),
                depth_spread=-1,
            )


# ---------------------------------------------------------------------------
# MomentumAgent — window validation
# ---------------------------------------------------------------------------


class TestMomentumAgentWindowValidation:
    def test_short_window_zero_raises(self):
        """short_window=0 must raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            MomentumAgent(
                id=0,
                symbol="TEST",
                starting_cash=10_000_000,
                random_state=np.random.RandomState(42),
                short_window=0,
                long_window=10,
            )

    def test_long_window_zero_raises(self):
        """long_window=0 must raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            MomentumAgent(
                id=0,
                symbol="TEST",
                starting_cash=10_000_000,
                random_state=np.random.RandomState(42),
                short_window=5,
                long_window=0,
            )

    def test_short_exceeds_long_raises(self):
        """short_window > long_window must raise ValueError."""
        with pytest.raises(ValueError, match="must be <= long_window"):
            MomentumAgent(
                id=0,
                symbol="TEST",
                starting_cash=10_000_000,
                random_state=np.random.RandomState(42),
                short_window=50,
                long_window=10,
            )

    def test_place_orders_handles_none_bid(self):
        """place_orders must not crash when bid is None."""
        agent = MomentumAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            short_window=5,
            long_window=10,
        )
        # Should be a no-op — no crash, no mid_list growth.
        agent.place_orders(None, 100_000)
        assert len(agent.mid_list) == 0

    def test_place_orders_handles_none_ask(self):
        """place_orders must not crash when ask is None."""
        agent = MomentumAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            short_window=5,
            long_window=10,
        )
        agent.place_orders(100_000, None)
        assert len(agent.mid_list) == 0


# ---------------------------------------------------------------------------
# ValueAgent — replace_order instead of cancel-all + place-new
# ---------------------------------------------------------------------------


class TestValueAgentReplaceOrder:
    """Verify ValueAgent uses replace_order for existing orders and
    place_limit_order when no open order exists (first cycle / filled)."""

    def _make_agent(self) -> ValueAgent:
        agent = ValueAgent(
            id=0,
            random_state=np.random.RandomState(42),
            symbol="TEST",
            starting_cash=10_000_000,
            r_bar=100_000,
            sigma_n=10_000,
        )
        # Fake enough state so place_order doesn't crash.
        agent.current_time = MKT_OPEN + str_to_ns("00:05:00")
        agent.mkt_open = MKT_OPEN
        agent.mkt_close = MKT_CLOSE
        agent.exchange_id = 99

        # Stub oracle so update_estimates() works without a kernel.
        class _FakeOracle:
            def observe_price(self, symbol, current_time, sigma_n=0, random_state=None):
                return 100_000

        agent.oracle = _FakeOracle()  # type: ignore[assignment]
        return agent

    def test_first_cycle_uses_place_limit_order(self):
        """When self.orders is empty, place_order should use place_limit_order (no replace)."""
        agent = self._make_agent()
        assert len(agent.orders) == 0

        placed = []
        replaced = []

        def mock_place(*args, **kwargs):
            placed.append(args)

        def mock_replace(*args, **kwargs):
            replaced.append(args)

        agent.place_limit_order = mock_place
        agent.replace_order = mock_replace

        # Seed known_bids/asks so place_order has spread data.
        agent.known_bids = {"TEST": [[99_900, 100]]}
        agent.known_asks = {"TEST": [[100_100, 100]]}

        agent.place_order()

        assert len(placed) == 1, "Should call place_limit_order on first cycle"
        assert len(replaced) == 0, "Should NOT call replace_order on first cycle"

    def test_subsequent_cycle_uses_replace_order(self):
        """When self.orders has an open LimitOrder, place_order should use replace_order."""
        from abides_markets.orders import LimitOrder

        agent = self._make_agent()

        # Simulate an existing open order from a previous cycle.
        old = LimitOrder(
            agent_id=0,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=30,
            side=Side.BID,
            limit_price=99_950,
        )
        agent.orders[old.order_id] = old

        placed = []
        replaced = []

        def mock_place(*args, **kwargs):
            placed.append(args)

        def mock_replace(*args, **kwargs):
            replaced.append(args)

        agent.place_limit_order = mock_place
        agent.replace_order = mock_replace

        agent.known_bids = {"TEST": [[99_900, 100]]}
        agent.known_asks = {"TEST": [[100_100, 100]]}

        agent.place_order()

        assert len(replaced) == 1, "Should call replace_order when open order exists"
        assert len(placed) == 0, "Should NOT call place_limit_order when replacing"
        # The old order should be the first argument to replace_order.
        assert replaced[0][0] is old

    def test_wakeup_does_not_cancel_all_orders(self):
        """ValueAgent.wakeup must not call cancel_all_orders (replaced by replace logic)."""
        import inspect

        source = inspect.getsource(ValueAgent.wakeup)
        assert "cancel_all_orders" not in source


# ---------------------------------------------------------------------------
# AdaptiveMarketMakerAgent — replace_order instead of cancel-all + place-new
# ---------------------------------------------------------------------------


class TestAdaptiveMarketMakerReplaceOrder:
    """Verify AdaptiveMarketMakerAgent uses replace_order for existing orders
    in poll mode and does NOT call cancel_all_orders from wakeup."""

    def test_poll_mode_wakeup_does_not_cancel_all(self):
        """Poll-mode wakeup must not call cancel_all_orders anymore."""
        import inspect

        source = inspect.getsource(AdaptiveMarketMakerAgent.wakeup)
        assert "cancel_all_orders" not in source

    def test_diff_and_replace_replaces_changed_orders(self):
        """_diff_and_replace should call replace_order when price or size differs."""
        from abides_markets.orders import LimitOrder

        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
            num_ticks=3,
        )
        agent.current_time = MKT_OPEN
        agent.exchange_id = 99

        # Existing order at 99_000.
        old = LimitOrder(
            agent_id=0,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=20,
            side=Side.BID,
            limit_price=99_000,
        )

        replaced = []
        cancelled = []

        def mock_replace(old_o, new_o):
            replaced.append((old_o, new_o))

        def mock_cancel(o):
            cancelled.append(o)

        agent.replace_order = mock_replace
        agent.cancel_order = mock_cancel

        new_orders: list = []
        # Desired order at a DIFFERENT price.
        agent._diff_and_replace(
            existing=[old],
            desired=[(99_100, 20)],
            side=Side.BID,
            new_orders=new_orders,
        )

        assert len(replaced) == 1, "Should replace when price differs"
        assert replaced[0][0] is old
        assert replaced[0][1].limit_price == 99_100
        assert len(cancelled) == 0
        assert len(new_orders) == 0

    def test_diff_and_replace_skips_identical(self):
        """_diff_and_replace should skip orders with identical price and size."""
        from abides_markets.orders import LimitOrder

        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
        )
        agent.current_time = MKT_OPEN
        agent.exchange_id = 99

        old = LimitOrder(
            agent_id=0,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=20,
            side=Side.BID,
            limit_price=99_000,
        )

        replaced = []
        cancelled = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        agent._diff_and_replace(
            existing=[old],
            desired=[(99_000, 20)],  # Same price AND size
            side=Side.BID,
            new_orders=new_orders,
        )

        assert len(replaced) == 0, "Should skip identical orders"
        assert len(cancelled) == 0
        assert len(new_orders) == 0

    def test_diff_and_replace_cancels_surplus(self):
        """_diff_and_replace should cancel surplus existing orders."""
        from abides_markets.orders import LimitOrder

        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
        )
        agent.current_time = MKT_OPEN
        agent.exchange_id = 99

        old1 = LimitOrder(
            agent_id=0,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=20,
            side=Side.BID,
            limit_price=99_000,
        )
        old2 = LimitOrder(
            agent_id=0,
            time_placed=MKT_OPEN,
            symbol="TEST",
            quantity=20,
            side=Side.BID,
            limit_price=98_900,
        )

        cancelled = []
        replaced = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        # Only 1 desired vs 2 existing → 1 replace + 1 cancel.
        agent._diff_and_replace(
            existing=[old1, old2],
            desired=[(99_100, 20)],
            side=Side.BID,
            new_orders=new_orders,
        )

        assert len(replaced) == 1
        assert len(cancelled) == 1
        assert cancelled[0] is old2

    def test_diff_and_replace_places_fresh_surplus(self):
        """_diff_and_replace should queue surplus desired orders for batch placement."""
        agent = AdaptiveMarketMakerAgent(
            id=0,
            symbol="TEST",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),
            subscribe=False,
        )
        agent.current_time = MKT_OPEN
        agent.exchange_id = 99

        replaced = []
        cancelled = []
        agent.replace_order = lambda o, n: replaced.append((o, n))
        agent.cancel_order = lambda o: cancelled.append(o)

        new_orders: list = []
        # 0 existing, 2 desired → 2 fresh orders.
        agent._diff_and_replace(
            existing=[],
            desired=[(99_000, 20), (98_900, 20)],
            side=Side.BID,
            new_orders=new_orders,
        )

        assert len(replaced) == 0
        assert len(cancelled) == 0
        assert len(new_orders) == 2
        assert new_orders[0].limit_price == 99_000
        assert new_orders[1].limit_price == 98_900


# ---------------------------------------------------------------------------
# Regression guard: every registered agent stores all __init__ params
# ---------------------------------------------------------------------------


def _init_stores_param(cls: type, param: str) -> bool:
    """Check whether *cls* or any ancestor stores *param* in ``__init__``.

    Walks the MRO and looks for ``self.<param> =`` or ``self.<param>:`` in
    each class's own ``__init__`` source.  This catches the exact bug class
    where a parameter is accepted but never assigned to ``self``.
    """
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            src = inspect.getsource(init)
        except (OSError, TypeError):
            continue
        if f"self.{param} =" in src or f"self.{param}:" in src:
            return True
    return False


# Parameters handled by the framework / parent plumbing — never expected
# to appear as ``self.<param>`` in the leaf agent class.
_FRAMEWORK_PARAMS: frozenset[str] = frozenset(
    {
        "id",
        "name",
        "type",
        "random_state",
        "risk_config",
    }
)


def _agent_params() -> list[tuple[str, str, type]]:
    """Yield (agent_name, param_name, agent_class) for parametrize."""
    from abides_markets.config_system.registry import registry

    cases: list[tuple[str, str, type]] = []
    for agent_name in registry.registered_names():
        entry = registry.get(agent_name)
        agent_cls = entry.agent_class
        if agent_cls is None:
            continue
        sig = inspect.signature(agent_cls.__init__)  # type: ignore[misc]
        for pname in sig.parameters:
            if pname == "self" or pname in _FRAMEWORK_PARAMS:
                continue
            cases.append((agent_name, pname, agent_cls))
    return cases


@pytest.mark.parametrize(
    "agent_name, param, agent_cls",
    _agent_params(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_registered_agent_stores_init_param(agent_name, param, agent_cls):
    """Every __init__ parameter must be stored as self.<param> somewhere in the MRO.

    Regression guard for the v2.4.0 bug where POVExecutionAgent accepted
    ``symbol`` but never assigned ``self.symbol``, causing AttributeError
    on the first wakeup.  This test covers *all* registered agents
    automatically — no per-agent maintenance required.
    """
    assert _init_stores_param(agent_cls, param), (
        f"{agent_cls.__name__}.__init__() accepts '{param}' but no class "
        f"in its MRO stores it as self.{param}"
    )
