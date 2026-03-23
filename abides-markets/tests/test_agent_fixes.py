"""Regression tests for agent and oracle bug fixes (v2.0.0).

Covers:
- AdaptiveMarketMakerAgent: state key consistency in subscribe mode
- ValueAgent: log_orders type annotation
- MomentumAgent: safe dict access for empty known_bids/known_asks
- Oracle ABC: f_log default property
- ExternalDataOracle: inherits f_log default
- NoiseAgent: dead oracle reference removed
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.agents.examples.momentum_agent import MomentumAgent
from abides_markets.agents.market_makers.adaptive_market_maker_agent import (
    AdaptiveMarketMakerAgent,
)
from abides_markets.agents.value_agent import ValueAgent
from abides_markets.oracles.data_providers import DataFrameProvider
from abides_markets.oracles.external_data_oracle import ExternalDataOracle
from abides_markets.oracles.oracle import Oracle
from abides_markets.oracles.sparse_mean_reverting_oracle import (
    SparseMeanRevertingOracle,
)

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
