"""
Test that ABIDES simulations are perfectly replicable when using the same seed.

This test verifies that:
1. Oracle fundamental values are deterministic
2. Order book states are deterministic
3. No global random state is used (simulations are isolated)
"""

import numpy as np

from abides_core import Kernel
from abides_core.utils import str_to_ns
from abides_markets.configs import rmsc04


def capture_simulation_state(config):
    """
    Run a simulation and capture oracle prices and order book states.

    Returns:
        dict with oracle_prices (list of fundamental values) and
        order_book_snapshots (list of order book states)
    """
    # Get oracle and exchange from config
    oracle = config["oracle"]
    exchange = config["agents"][0]  # First agent is the exchange

    # Capture oracle fundamental values at regular intervals
    oracle_prices = []

    start_time = config["start_time"]

    # Sample oracle prices every 10 seconds during market hours
    mkt_open = start_time + str_to_ns("09:30:00")
    mkt_close = start_time + str_to_ns("10:00:00")

    sample_interval = str_to_ns("10s")
    current_time = mkt_open

    # Get ticker from first symbol in oracle
    ticker = list(oracle.symbols.keys())[0]

    # Create a random state for oracle observations (deterministic per run)
    obs_random_state = np.random.RandomState(seed=42)

    while current_time <= mkt_close:
        # Observe fundamental value with no noise
        fund_value = oracle.observe_price(
            ticker, current_time, sigma_n=0, random_state=obs_random_state
        )
        oracle_prices.append({"time": current_time, "fundamental_value": fund_value})
        current_time += sample_interval

    # Create and run kernel
    kernel = Kernel(
        random_state=config["random_state_kernel"],
        start_time=config["start_time"],
        stop_time=config["stop_time"],
        agents=config["agents"],
        agent_latency_model=config["agent_latency_model"],
        default_computation_delay=config["default_computation_delay"],
        oracle=config["oracle"],
        skip_log=True,  # Don't generate log files during tests
    )

    # Run the simulation
    kernel.run()

    # After simulation, capture order book snapshots
    order_book_snapshots = []

    # Get the order book from the exchange's internal state
    if hasattr(exchange, "order_books") and ticker in exchange.order_books:
        order_book = exchange.order_books[ticker]

        # Capture final order book state
        snapshot = {
            "bids": [],
            "asks": [],
            "last_trade": (
                order_book.last_trade if hasattr(order_book, "last_trade") else None
            ),
        }

        # Get top 10 levels of bids and asks (bids/asks are lists of PriceLevel objects)
        if hasattr(order_book, "bids") and order_book.bids:
            for price_level in order_book.bids[:10]:
                snapshot["bids"].append(
                    {"price": price_level.price, "volume": price_level.total_quantity}
                )

        if hasattr(order_book, "asks") and order_book.asks:
            for price_level in order_book.asks[:10]:
                snapshot["asks"].append(
                    {"price": price_level.price, "volume": price_level.total_quantity}
                )

        order_book_snapshots.append(snapshot)

    return {
        "oracle_prices": oracle_prices,
        "order_book_snapshots": order_book_snapshots,
    }


def test_rmsc04_replicability_with_same_seed():
    """
    Test that running rmsc04 twice with the same seed produces identical results.
    """
    seed = 123456789

    # Run simulation 1
    config1 = rmsc04.build_config(
        seed=seed,
        end_time="10:00:00",
        num_noise_agents=100,  # Use fewer agents for faster test
        num_value_agents=10,
        num_momentum_agents=5,
    )
    state1 = capture_simulation_state(config1)

    # Run simulation 2 with same seed
    config2 = rmsc04.build_config(
        seed=seed,
        end_time="10:00:00",
        num_noise_agents=100,
        num_value_agents=10,
        num_momentum_agents=5,
    )
    state2 = capture_simulation_state(config2)

    # Verify oracle prices are identical
    assert len(state1["oracle_prices"]) == len(
        state2["oracle_prices"]
    ), "Oracle price series have different lengths"

    for i, (price1, price2) in enumerate(
        zip(state1["oracle_prices"], state2["oracle_prices"])
    ):
        assert (
            price1["time"] == price2["time"]
        ), f"Oracle sample {i}: times differ ({price1['time']} != {price2['time']})"
        assert (
            price1["fundamental_value"] == price2["fundamental_value"]
        ), f"Oracle sample {i}: fundamental values differ ({price1['fundamental_value']} != {price2['fundamental_value']})"

    print(
        f"✓ Oracle prices are identical across {len(state1['oracle_prices'])} samples"
    )

    # Verify order book snapshots are identical
    assert len(state1["order_book_snapshots"]) == len(
        state2["order_book_snapshots"]
    ), "Different number of order book snapshots"

    for i, (snapshot1, snapshot2) in enumerate(
        zip(state1["order_book_snapshots"], state2["order_book_snapshots"])
    ):
        # Check bids
        assert len(snapshot1["bids"]) == len(
            snapshot2["bids"]
        ), f"Snapshot {i}: different number of bid levels"

        for j, (bid1, bid2) in enumerate(zip(snapshot1["bids"], snapshot2["bids"])):
            assert (
                bid1["price"] == bid2["price"]
            ), f"Snapshot {i}, bid level {j}: prices differ"
            assert (
                bid1["volume"] == bid2["volume"]
            ), f"Snapshot {i}, bid level {j}: volumes differ"

        # Check asks
        assert len(snapshot1["asks"]) == len(
            snapshot2["asks"]
        ), f"Snapshot {i}: different number of ask levels"

        for j, (ask1, ask2) in enumerate(zip(snapshot1["asks"], snapshot2["asks"])):
            assert (
                ask1["price"] == ask2["price"]
            ), f"Snapshot {i}, ask level {j}: prices differ"
            assert (
                ask1["volume"] == ask2["volume"]
            ), f"Snapshot {i}, ask level {j}: volumes differ"

        # Check last trade
        assert (
            snapshot1["last_trade"] == snapshot2["last_trade"]
        ), f"Snapshot {i}: last trade differs"

    print(
        f"✓ Order book snapshots are identical across {len(state1['order_book_snapshots'])} snapshots"
    )


def test_rmsc04_replicability_with_different_seeds():
    """
    Test that running rmsc04 with different seeds produces different results.
    This verifies that the random number generation is actually working.
    """
    seed1 = 123456789
    seed2 = 987654321

    # Run simulation 1
    config1 = rmsc04.build_config(
        seed=seed1,
        end_time="10:00:00",
        num_noise_agents=100,
        num_value_agents=10,
        num_momentum_agents=5,
    )
    state1 = capture_simulation_state(config1)

    # Run simulation 2 with different seed
    config2 = rmsc04.build_config(
        seed=seed2,
        end_time="10:00:00",
        num_noise_agents=100,
        num_value_agents=10,
        num_momentum_agents=5,
    )
    state2 = capture_simulation_state(config2)

    # Verify oracle prices are different
    differences_found = False
    for price1, price2 in zip(state1["oracle_prices"], state2["oracle_prices"]):
        if price1["fundamental_value"] != price2["fundamental_value"]:
            differences_found = True
            break

    assert (
        differences_found
    ), "Oracle prices are identical despite different seeds - randomness not working!"

    print("✓ Different seeds produce different oracle prices as expected")


# ===================================================================
# Additional seed/RNG isolation tests
# ===================================================================


def test_oracle_isolation_from_agent_count_change():
    """Adding agents must not change oracle's fundamental value series.

    The oracle has its own sub-seeded RandomState, so changing the number
    of agents (which draws from the kernel's RNG for agent seeds) must NOT
    change the oracle's output.
    """
    seed = 42

    config_a = rmsc04.build_config(
        seed=seed,
        end_time="10:00:00",
        num_noise_agents=50,
        num_value_agents=5,
        num_momentum_agents=2,
    )
    config_b = rmsc04.build_config(
        seed=seed,
        end_time="10:00:00",
        num_noise_agents=80,  # different agent count
        num_value_agents=8,
        num_momentum_agents=3,
    )

    oracle_a = config_a["oracle"]
    oracle_b = config_b["oracle"]

    ticker = list(oracle_a.symbols.keys())[0]
    mkt_open = config_a["start_time"] + str_to_ns("09:30:00")

    obs_rs_a = np.random.RandomState(99)
    obs_rs_b = np.random.RandomState(99)

    for offset in [0, str_to_ns("5s"), str_to_ns("30s"), str_to_ns("1min")]:
        ts = mkt_open + offset
        val_a = oracle_a.observe_price(ticker, ts, obs_rs_a, sigma_n=0)
        val_b = oracle_b.observe_price(ticker, ts, obs_rs_b, sigma_n=0)
        assert val_a == val_b, (
            f"Oracle value differs at offset={offset}: {val_a} vs {val_b}. "
            "Agent count change leaked into oracle RNG."
        )


def test_config_build_same_seed_same_agent_ids():
    """Same seed → identical agent lists (names, types)."""
    seed = 12345

    def build():
        return rmsc04.build_config(
            seed=seed,
            end_time="10:00:00",
            num_noise_agents=20,
            num_value_agents=5,
            num_momentum_agents=2,
        )

    c1 = build()
    c2 = build()

    names1 = [a.name for a in c1["agents"]]
    names2 = [a.name for a in c2["agents"]]
    assert names1 == names2

    types1 = [type(a).__name__ for a in c1["agents"]]
    types2 = [type(a).__name__ for a in c2["agents"]]
    assert types1 == types2


if __name__ == "__main__":
    print("Testing ABIDES seed replicability...")
    print("\n1. Testing replicability with same seed...")
    test_rmsc04_replicability_with_same_seed()

    print("\n2. Testing differences with different seeds...")
    test_rmsc04_replicability_with_different_seeds()

    print("\n3. Testing oracle isolation from agent count...")
    test_oracle_isolation_from_agent_count_change()

    print("\n4. Testing config build determinism...")
    test_config_build_same_seed_same_agent_ids()

    print("\n✅ All replicability tests passed!")
