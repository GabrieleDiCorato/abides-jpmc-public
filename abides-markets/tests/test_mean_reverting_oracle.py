"""Tests for MeanRevertingOracle.

MeanRevertingOracle is deprecated in favour of SparseMeanRevertingOracle.
These tests exercise the fixed pd.date_range call, the .loc access pattern
with integer nanosecond keys, and the safety guards (step-count limit and
deprecation warning).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.oracles.mean_reverting_oracle import MeanRevertingOracle

# Use a tiny time range (1000 nanoseconds) to keep tests fast.
# The oracle interprets nanoseconds as discrete time steps.
DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + 1000  # 1000 nanosecond steps

R_BAR = 100_000
SYMBOLS = {"TEST": {"r_bar": R_BAR, "kappa": 0.05, "sigma_s": 100}}


def _make_oracle(seed=42):
    random_state = np.random.RandomState(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MeanRevertingOracle(MKT_OPEN, MKT_CLOSE, SYMBOLS, random_state)


class TestMeanRevertingOracleConstruction:
    def test_creates_series_for_symbol(self):
        oracle = _make_oracle()
        assert "TEST" in oracle.r
        assert isinstance(oracle.r["TEST"], pd.Series)

    def test_series_length_matches_time_range(self):
        oracle = _make_oracle()
        assert len(oracle.r["TEST"]) == 1000

    def test_series_has_datetime_index(self):
        oracle = _make_oracle()
        assert isinstance(oracle.r["TEST"].index, pd.DatetimeIndex)

    def test_first_value_is_r_bar(self):
        oracle = _make_oracle()
        assert oracle.r["TEST"].iloc[0] == R_BAR

    def test_values_are_integers(self):
        oracle = _make_oracle()
        assert oracle.r["TEST"].dtype in (np.int32, np.int64, int)

    def test_values_non_negative(self):
        oracle = _make_oracle()
        assert (oracle.r["TEST"] >= 0).all()


class TestMeanRevertingOracleAccess:
    def test_get_daily_open_price(self):
        oracle = _make_oracle()
        price = oracle.get_daily_open_price("TEST", MKT_OPEN)
        assert price == R_BAR

    def test_observe_price_at_open(self):
        oracle = _make_oracle()
        random_state = np.random.RandomState(0)
        # With sigma_n=0, observe_price returns the exact fundamental value
        price = oracle.observe_price("TEST", MKT_OPEN, random_state, sigma_n=0)
        assert price == R_BAR

    def test_observe_price_at_close(self):
        oracle = _make_oracle()
        random_state = np.random.RandomState(0)
        # Querying at or after mkt_close returns the close price
        price = oracle.observe_price("TEST", MKT_CLOSE, random_state, sigma_n=0)
        assert isinstance(price, (int, np.integer))

    def test_observe_price_mid_series(self):
        oracle = _make_oracle()
        random_state = np.random.RandomState(0)
        mid_time = MKT_OPEN + 500
        price = oracle.observe_price("TEST", mid_time, random_state, sigma_n=0)
        assert isinstance(price, (int, np.integer))
        assert price >= 0

    def test_noisy_observation_varies(self):
        """With sigma_n > 0, different random states give different observations."""
        oracle = _make_oracle()
        prices = set()
        for seed in range(10):
            rs = np.random.RandomState(seed)
            prices.add(oracle.observe_price("TEST", MKT_OPEN, rs, sigma_n=1000))
        # With 10 different seeds, we should get multiple distinct noisy values
        assert len(prices) > 1


class TestMeanRevertingOracleReproducibility:
    def test_same_seed_same_series(self):
        oracle_a = _make_oracle(seed=42)
        oracle_b = _make_oracle(seed=42)
        pd.testing.assert_series_equal(oracle_a.r["TEST"], oracle_b.r["TEST"])

    def test_different_seed_different_series(self):
        oracle_a = _make_oracle(seed=42)
        oracle_b = _make_oracle(seed=99)
        assert not oracle_a.r["TEST"].equals(oracle_b.r["TEST"])


# --- Safety guards ---


class TestMeanRevertingOracleSafety:
    def test_rejects_large_time_range(self):
        """Constructing with > 1 000 000 steps should raise ValueError."""
        rng = np.random.RandomState(42)
        with (
            pytest.raises(ValueError, match="exceeds the maximum"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", DeprecationWarning)
            MeanRevertingOracle(MKT_OPEN, MKT_OPEN + 2_000_000, SYMBOLS, rng)

    def test_deprecation_warning(self):
        """Construction should emit a DeprecationWarning."""
        rng = np.random.RandomState(42)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            MeanRevertingOracle(MKT_OPEN, MKT_CLOSE, SYMBOLS, rng)

    def test_small_range_still_works(self):
        """A 1000-step range should still construct successfully."""
        oracle = _make_oracle()
        assert len(oracle.r["TEST"]) == 1000
