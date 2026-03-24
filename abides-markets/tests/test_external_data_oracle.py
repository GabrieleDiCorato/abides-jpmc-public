"""Tests for ExternalDataOracle with batch and point data providers.

Covers both operating modes (batch via DataFrameProvider / raw dict,
point via a mock PointDataProvider), interpolation strategies, edge
cases, and noisy observations.
"""

import numpy as np
import pandas as pd
import pytest

from abides_core.utils import datetime_str_to_ns, str_to_ns
from abides_markets.oracles.data_providers import (
    DataFrameProvider,
    InterpolationStrategy,
)
from abides_markets.oracles.external_data_oracle import ExternalDataOracle

# ---------------------------------------------------------------------------
# Fixtures — shared time setup (mirrors test_mean_reverting_oracle.py)
# ---------------------------------------------------------------------------

DATE = datetime_str_to_ns("20210205")
MKT_OPEN = DATE + str_to_ns("09:30:00")
MKT_CLOSE = MKT_OPEN + 1000  # 1000 nanosecond steps

PRICE_INITIAL = 100_000  # $1,000.00 in cents


def _make_series(
    start: int = MKT_OPEN,
    end: int = MKT_CLOSE,
    step: int = 1,
    base_price: int = PRICE_INITIAL,
    drift: int = 10,
) -> pd.Series:
    """Build a simple time-indexed pd.Series with a small linear drift."""
    timestamps = range(start, end, step)
    prices = [base_price + i * drift for i, _ in enumerate(timestamps)]
    index = pd.to_datetime(list(timestamps), unit="ns")
    return pd.Series(prices, index=index, dtype=int)


def _make_oracle_batch(
    series: pd.Series | None = None,
    symbol: str = "TEST",
    interpolation: InterpolationStrategy = InterpolationStrategy.FORWARD_FILL,
) -> ExternalDataOracle:
    """Helper: create an ExternalDataOracle in batch mode."""
    if series is None:
        series = _make_series()
    return ExternalDataOracle(
        mkt_open=MKT_OPEN,
        mkt_close=MKT_CLOSE,
        symbols=[symbol],
        data={symbol: series},
        interpolation=interpolation,
    )


class _MockPointProvider:
    """Minimal PointDataProvider for testing."""

    def __init__(self, base_price: int = PRICE_INITIAL) -> None:
        self.base_price = base_price
        self.call_count = 0

    def get_fundamental_at(self, symbol: str, timestamp: int) -> int:
        self.call_count += 1
        # Deterministic price: shifts by 1 cent per nanosecond
        return self.base_price + (timestamp - MKT_OPEN)


def _make_oracle_point(
    provider: _MockPointProvider | None = None,
    cache_size: int = 100,
) -> ExternalDataOracle:
    """Helper: create an ExternalDataOracle in point mode."""
    if provider is None:
        provider = _MockPointProvider()
    return ExternalDataOracle(
        mkt_open=MKT_OPEN,
        mkt_close=MKT_CLOSE,
        symbols=["TEST"],
        provider=provider,
        cache_size=cache_size,
    )


# ===================================================================
# Batch Mode Tests
# ===================================================================


class TestBatchModeConstruction:
    def test_from_raw_dict(self):
        series = _make_series()
        oracle = ExternalDataOracle(
            MKT_OPEN, MKT_CLOSE, ["TEST"], data={"TEST": series}
        )
        assert oracle._mode == "batch"
        assert "TEST" in oracle.r

    def test_from_dataframe_provider(self):
        series = _make_series()
        provider = DataFrameProvider({"TEST": series})
        oracle = ExternalDataOracle(MKT_OPEN, MKT_CLOSE, ["TEST"], provider=provider)
        assert oracle._mode == "batch"

    def test_rejects_both_provider_and_data(self):
        series = _make_series()
        provider = DataFrameProvider({"TEST": series})
        with pytest.raises(ValueError, match="not both"):
            ExternalDataOracle(
                MKT_OPEN, MKT_CLOSE, ["TEST"], provider=provider, data={"TEST": series}
            )

    def test_rejects_neither_provider_nor_data(self):
        with pytest.raises(ValueError, match="must be given"):
            ExternalDataOracle(MKT_OPEN, MKT_CLOSE, ["TEST"])

    def test_rejects_empty_series(self):
        empty = pd.Series(dtype=int)
        with pytest.raises(ValueError, match="empty series"):
            ExternalDataOracle(MKT_OPEN, MKT_CLOSE, ["TEST"], data={"TEST": empty})

    def test_multiple_symbols(self):
        s1 = _make_series(base_price=100_000)
        s2 = _make_series(base_price=200_000)
        oracle = ExternalDataOracle(
            MKT_OPEN, MKT_CLOSE, ["A", "B"], data={"A": s1, "B": s2}
        )
        assert oracle.get_daily_open_price("A", MKT_OPEN) == 100_000
        assert oracle.get_daily_open_price("B", MKT_OPEN) == 200_000


class TestBatchModeAccess:
    def test_get_daily_open_price(self):
        oracle = _make_oracle_batch()
        price = oracle.get_daily_open_price("TEST", MKT_OPEN)
        assert price == PRICE_INITIAL

    def test_observe_price_exact_no_noise(self):
        oracle = _make_oracle_batch()
        rs = np.random.RandomState(42)
        price = oracle.observe_price("TEST", MKT_OPEN, rs, sigma_n=0)
        assert price == PRICE_INITIAL

    def test_observe_price_mid_series(self):
        oracle = _make_oracle_batch()
        rs = np.random.RandomState(42)
        mid = MKT_OPEN + 500
        price = oracle.observe_price("TEST", mid, rs, sigma_n=0)
        # drift = 10 per step, step 500 → 100_000 + 500*10 = 105_000
        assert price == PRICE_INITIAL + 500 * 10

    def test_observe_price_at_close_clamps(self):
        oracle = _make_oracle_batch()
        rs = np.random.RandomState(42)
        # Querying at or after mkt_close returns the last available value
        price = oracle.observe_price("TEST", MKT_CLOSE, rs, sigma_n=0)
        assert isinstance(price, int)
        assert price >= PRICE_INITIAL

    def test_noisy_observation_varies(self):
        oracle = _make_oracle_batch()
        prices = set()
        for seed in range(10):
            rs = np.random.RandomState(seed)
            prices.add(oracle.observe_price("TEST", MKT_OPEN, rs, sigma_n=1000))
        assert len(prices) > 1

    def test_noisy_observation_centered_on_fundamental(self):
        """Mean of many noisy observations should be close to fundamental."""
        oracle = _make_oracle_batch()
        observations = []
        for seed in range(1000):
            rs = np.random.RandomState(seed)
            observations.append(oracle.observe_price("TEST", MKT_OPEN, rs, sigma_n=100))
        mean_obs = np.mean(observations)
        assert abs(mean_obs - PRICE_INITIAL) < 50  # within ~$0.50


# ===================================================================
# Point Mode Tests
# ===================================================================


class TestPointModeConstruction:
    def test_point_mode_detected(self):
        oracle = _make_oracle_point()
        assert oracle._mode == "point"

    def test_cache_initialized_per_symbol(self):
        oracle = _make_oracle_point()
        assert "TEST" in oracle._caches


class TestPointModeAccess:
    def test_get_daily_open_price(self):
        oracle = _make_oracle_point()
        price = oracle.get_daily_open_price("TEST", MKT_OPEN)
        assert price == PRICE_INITIAL

    def test_observe_price_no_noise(self):
        oracle = _make_oracle_point()
        rs = np.random.RandomState(42)
        price = oracle.observe_price("TEST", MKT_OPEN + 100, rs, sigma_n=0)
        assert price == PRICE_INITIAL + 100

    def test_cache_prevents_duplicate_calls(self):
        provider = _MockPointProvider()
        oracle = _make_oracle_point(provider=provider)
        rs = np.random.RandomState(42)
        # First call — cache miss
        oracle.observe_price("TEST", MKT_OPEN + 50, rs, sigma_n=0)
        count_after_first = provider.call_count
        # Second call — cache hit
        oracle.observe_price("TEST", MKT_OPEN + 50, rs, sigma_n=0)
        assert provider.call_count == count_after_first

    def test_cache_eviction(self):
        provider = _MockPointProvider()
        oracle = _make_oracle_point(provider=provider, cache_size=5)
        rs = np.random.RandomState(42)
        # Fill cache with 5 entries
        for i in range(5):
            oracle.observe_price("TEST", MKT_OPEN + i, rs, sigma_n=0)
        assert len(oracle._caches["TEST"]) == 5
        # 6th entry should evict the oldest
        oracle.observe_price("TEST", MKT_OPEN + 5, rs, sigma_n=0)
        assert len(oracle._caches["TEST"]) == 5

    def test_noisy_point_observation(self):
        oracle = _make_oracle_point()
        prices = set()
        for seed in range(10):
            rs = np.random.RandomState(seed)
            prices.add(oracle.observe_price("TEST", MKT_OPEN, rs, sigma_n=1000))
        assert len(prices) > 1


# ===================================================================
# Interpolation Strategy Tests
# ===================================================================


class TestInterpolationStrategies:
    """Test with a sparse series (every 100ns) queried at intermediate timestamps."""

    def _make_sparse_series(self) -> pd.Series:
        """Create data at every 100ns within the test window."""
        return _make_series(step=100, drift=100)

    def test_forward_fill(self):
        series = self._make_sparse_series()
        oracle = _make_oracle_batch(
            series=series, interpolation=InterpolationStrategy.FORWARD_FILL
        )
        rs = np.random.RandomState(42)
        # Query at MKT_OPEN + 50 — between data at +0 and +100
        # Forward fill should return the value at +0
        price = oracle.observe_price("TEST", MKT_OPEN + 50, rs, sigma_n=0)
        expected_at_0 = PRICE_INITIAL
        assert price == expected_at_0

    def test_nearest(self):
        series = self._make_sparse_series()
        oracle = _make_oracle_batch(
            series=series, interpolation=InterpolationStrategy.NEAREST
        )
        rs = np.random.RandomState(42)
        # Query at MKT_OPEN + 80 — closer to +100 than +0
        price = oracle.observe_price("TEST", MKT_OPEN + 80, rs, sigma_n=0)
        expected_at_100 = PRICE_INITIAL + 100  # drift=100 for step 1
        assert price == expected_at_100

    def test_nearest_at_midpoint(self):
        series = self._make_sparse_series()
        oracle = _make_oracle_batch(
            series=series, interpolation=InterpolationStrategy.NEAREST
        )
        rs = np.random.RandomState(42)
        # Query at MKT_OPEN + 50 — exactly between +0 and +100
        # "nearest" resolves ties to the earlier value
        price = oracle.observe_price("TEST", MKT_OPEN + 50, rs, sigma_n=0)
        assert price in (PRICE_INITIAL, PRICE_INITIAL + 100)

    def test_linear(self):
        series = self._make_sparse_series()
        oracle = _make_oracle_batch(
            series=series, interpolation=InterpolationStrategy.LINEAR
        )
        rs = np.random.RandomState(42)
        # Query at MKT_OPEN + 50 — should linearly interpolate
        price = oracle.observe_price("TEST", MKT_OPEN + 50, rs, sigma_n=0)
        # Between 100_000 (at +0) and 100_100 (at +100), midpoint = 100_050
        assert price == PRICE_INITIAL + 50


# ===================================================================
# Edge Cases
# ===================================================================


class TestEdgeCases:
    def test_single_point_series(self):
        """Oracle works with a series containing exactly one data point."""
        index = pd.to_datetime([MKT_OPEN], unit="ns")
        series = pd.Series([PRICE_INITIAL], index=index, dtype=int)
        oracle = _make_oracle_batch(series=series)
        rs = np.random.RandomState(42)
        price = oracle.observe_price("TEST", MKT_OPEN, rs, sigma_n=0)
        assert price == PRICE_INITIAL

    def test_query_exactly_on_data_point(self):
        series = _make_series(step=100)
        oracle = _make_oracle_batch(series=series)
        rs = np.random.RandomState(42)
        price = oracle.observe_price("TEST", MKT_OPEN + 100, rs, sigma_n=0)
        # Step 1 at MKT_OPEN+100 → base + 1*10 = 100_010
        assert price == PRICE_INITIAL + 10

    def test_oracle_is_subclass_of_oracle_base(self):
        from abides_markets.oracles.oracle import Oracle

        oracle = _make_oracle_batch()
        assert isinstance(oracle, Oracle)
