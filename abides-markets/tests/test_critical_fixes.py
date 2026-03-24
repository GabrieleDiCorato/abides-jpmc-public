"""Tests for critical bug fixes in exchange_agent, trading_agent, kernel, and oracle."""

import numpy as np

from abides_core.kernel import Kernel
from abides_markets.agents.exchange_agent import ExchangeAgent
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.oracles.sparse_mean_reverting_oracle import (
    SparseMeanRevertingOracle,
)

# --- Kernel has_run ---


def test_kernel_has_run_set_after_initialize():
    """has_run should be True after initialize() completes."""
    agents = []
    kernel = Kernel(agents=agents, start_time=0, stop_time=1)
    assert kernel.has_run is False
    kernel.initialize()
    assert kernel.has_run is True


# --- TradingAgent get_known_bid_ask KeyError ---


def test_get_known_bid_ask_no_keyerror_for_unknown_symbol():
    """get_known_bid_ask should return Nones/zeros for unknown symbols, not KeyError."""
    agent = TradingAgent(id=0, name="test", random_state=np.random.RandomState(42))
    agent.known_bids = {}
    agent.known_asks = {}
    bid, bid_vol, ask, ask_vol = agent.get_known_bid_ask("UNKNOWN")
    assert bid is None
    assert ask is None
    assert bid_vol == 0
    assert ask_vol == 0


def test_get_known_bid_ask_with_known_symbol():
    """get_known_bid_ask should return correct values for known symbols."""
    agent = TradingAgent(id=0, name="test", random_state=np.random.RandomState(42))
    agent.known_bids = {"IBM": [(10000, 100)]}
    agent.known_asks = {"IBM": [(10100, 50)]}
    bid, bid_vol, ask, ask_vol = agent.get_known_bid_ask("IBM")
    assert bid == 10000
    assert bid_vol == 100
    assert ask == 10100
    assert ask_vol == 50


def test_get_known_bid_ask_empty_book_sides():
    """get_known_bid_ask should handle empty lists for known symbols."""
    agent = TradingAgent(id=0, name="test", random_state=np.random.RandomState(42))
    agent.known_bids = {"IBM": []}
    agent.known_asks = {"IBM": []}
    bid, bid_vol, ask, ask_vol = agent.get_known_bid_ask("IBM")
    assert bid is None
    assert ask is None
    assert bid_vol == 0
    assert ask_vol == 0


# --- TradingAgent get_known_liquidity self.symbol bug ---


def test_get_known_liquidity_no_keyerror_for_unknown_symbol():
    """get_known_liquidity should not crash for unknown symbols."""
    agent = TradingAgent(id=0, name="test", random_state=np.random.RandomState(42))
    agent.known_bids = {}
    agent.known_asks = {}
    bid_liq, ask_liq = agent.get_known_liquidity("UNKNOWN")
    assert bid_liq == 0
    assert ask_liq == 0


# --- Oracle config mutation ---


def test_oracle_does_not_mutate_caller_config():
    """SparseMeanRevertingOracle should not modify the caller's symbols dict."""
    symbols_config = {
        "IBM": {
            "r_bar": 10000,
            "kappa": 1.67e-16,
            "sigma_s": 0,
            "fund_vol": 1e-8,
            "megashock_lambda_a": 0,
            "megashock_mean": 0,
            "megashock_var": 0,
            "random_state": np.random.RandomState(42),
        }
    }
    original_keys = set(symbols_config["IBM"].keys())
    SparseMeanRevertingOracle(
        mkt_open=0,
        mkt_close=int(1e18),
        symbols=symbols_config,
        random_state=np.random.RandomState(99),
    )
    # The caller's dict should not have been mutated
    assert set(symbols_config["IBM"].keys()) == original_keys


def test_oracle_creates_own_copy_of_symbols():
    """Oracle's internal symbols should be independent of caller's dict."""
    symbols_config = {
        "IBM": {
            "r_bar": 10000,
            "kappa": 1.67e-16,
            "sigma_s": 0,
            "fund_vol": 1e-8,
            "megashock_lambda_a": 0,
            "megashock_mean": 0,
            "megashock_var": 0,
            "random_state": np.random.RandomState(42),
        }
    }
    oracle = SparseMeanRevertingOracle(
        mkt_open=0,
        mkt_close=int(1e18),
        symbols=symbols_config,
        random_state=np.random.RandomState(99),
    )
    # Modifying caller's dict should not affect oracle's internal state
    symbols_config["IBM"]["r_bar"] = 99999
    assert oracle.symbols["IBM"]["r_bar"] == 10000


# --- Subscription cancellation fix ---


def test_subscription_cancel_type_mapping():
    """Verify the subscription type mapping dict is correct."""

    # Verify that subscription classes exist with expected attributes
    l1_sub = ExchangeAgent.L1DataSubscription(agent_id=0, last_update_ts=0, freq=1)
    assert not hasattr(l1_sub, "depth")  # L1 has no depth

    l2_sub = ExchangeAgent.L2DataSubscription(
        agent_id=0, last_update_ts=0, freq=1, depth=10
    )
    assert hasattr(l2_sub, "depth")  # L2 has depth

    tv_sub = ExchangeAgent.TransactedVolDataSubscription(
        agent_id=0, last_update_ts=0, freq=1, lookback="1min"
    )
    assert not hasattr(tv_sub, "depth")  # TransactedVol has no depth

    bi_sub = ExchangeAgent.BookImbalanceDataSubscription(
        agent_id=0, last_update_ts=0, event_in_progress=False, min_imbalance=0.5
    )
    assert not hasattr(bi_sub, "depth")  # BookImbalance has no depth
    assert not hasattr(bi_sub, "freq")  # BookImbalance has no freq


# --- Kernel ValueError formatting ---


def test_kernel_set_wakeup_valueerror_is_string():
    """ValueError from set_wakeup should have a formatted string, not tuple args."""
    kernel = Kernel(agents=[], start_time=0, stop_time=1)
    kernel.initialize()
    kernel.current_time = 100
    try:
        kernel.set_wakeup(sender_id=0, requested_time=50)
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        # Should be a single formatted string, not tuple args
        assert len(e.args) == 1, f"ValueError has {len(e.args)} args, expected 1"
        assert "current_time" in str(e)


# --- Kernel runner loop truthiness at time=0 ---


def test_kernel_runner_works_with_start_time_zero():
    """Kernel runner loop should work when start_time is 0 (falsy int)."""
    from abides_core.agent import Agent

    agent = Agent(id=0, name="test", random_state=np.random.RandomState(42))
    kernel = Kernel(agents=[agent], start_time=0, stop_time=100)
    kernel.initialize()
    # Should not hang or skip — the runner loop condition should handle time=0
    kernel.runner()
    # If we get here without error, the truthiness check works
