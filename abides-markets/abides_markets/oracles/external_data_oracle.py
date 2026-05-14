"""Oracle that serves externally-provided fundamental value data.

``ExternalDataOracle`` bridges custom data sources into the ABIDES simulation.
It accepts data through two mechanisms:

1. **Batch mode** — a ``BatchDataProvider`` (or raw ``Dict[str, pd.Series]``)
   loads the full fundamental value series at construction time.  Lookups use
   ``pd.Series.asof()`` for O(log n) access with configurable interpolation.

2. **Point mode** — a ``PointDataProvider`` returns individual values on
   demand.  An LRU cache with configurable size keeps memory bounded.

Both modes expose the same ``get_daily_open_price`` / ``observe_price`` API
as the built-in oracles, so the ``ExternalDataOracle`` is a drop-in
replacement anywhere ABIDES expects an ``Oracle``.
"""

import logging
from collections import OrderedDict
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from abides_core import NanosecondTime

from .data_providers import (
    BatchDataProvider,
    DataFrameProvider,
    InterpolationStrategy,
    PointDataProvider,
)
from .oracle import Oracle

logger = logging.getLogger(__name__)


class _LRUCache:
    """Minimal ordered-dict LRU cache with a fixed max size.

    Using a simple ``OrderedDict`` rather than ``functools.lru_cache``
    because we need per-symbol caches that are easy to inspect and reset.
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[Any, int] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> int | None:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def put(self, key: Any, value: int) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class ExternalDataOracle(Oracle):
    """Oracle backed by externally-provided fundamental value data.

    Construction
    ------------
    Exactly **one** of ``provider`` or ``data`` must be given.

    Parameters
    ----------
    mkt_open:
        Market open time in nanoseconds.
    mkt_close:
        Market close time in nanoseconds.
    symbols:
        List of symbol strings to serve.
    provider:
        A ``BatchDataProvider`` or ``PointDataProvider`` instance.
    data:
        Pre-loaded data as ``Dict[str, pd.Series]``.  Each series must have
        a ``DatetimeIndex`` and integer values in cents.  Mutually exclusive
        with *provider*.
    interpolation:
        Strategy for resolving timestamp gaps in batch mode.
        Ignored in point mode.  Default: ``FORWARD_FILL``.
    cache_size:
        Maximum entries in the per-symbol LRU cache (point mode only).
        Default: 10 000.

    Examples
    --------
    Batch mode from raw data::

        data = {"AAPL": my_series}
        oracle = ExternalDataOracle(mkt_open, mkt_close, ["AAPL"], data=data)

    Batch mode from a provider::

        oracle = ExternalDataOracle(
            mkt_open, mkt_close, ["AAPL"], provider=my_csv_provider
        )

    Point mode (on-demand, memory-bounded)::

        oracle = ExternalDataOracle(
            mkt_open, mkt_close, ["AAPL"],
            provider=my_db_provider, cache_size=5_000,
        )
    """

    def __init__(
        self,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: list[str],
        provider: BatchDataProvider | PointDataProvider | None = None,
        data: dict[str, pd.Series] | None = None,
        interpolation: InterpolationStrategy = InterpolationStrategy.FORWARD_FILL,
        cache_size: int = 10_000,
    ) -> None:
        if provider is not None and data is not None:
            raise ValueError("Specify either 'provider' or 'data', not both.")
        if provider is None and data is None:
            raise ValueError("One of 'provider' or 'data' must be given.")

        self.mkt_open = mkt_open
        self.mkt_close = mkt_close
        self.symbols_list = list(symbols)
        self.interpolation = interpolation

        # Determine operating mode
        if data is not None:
            self._mode = "batch"
            self._init_batch(DataFrameProvider(data))
        elif isinstance(provider, BatchDataProvider):
            self._mode = "batch"
            self._init_batch(provider)
        elif isinstance(provider, PointDataProvider):
            self._mode = "point"
            self._init_point(provider, cache_size)
        else:
            raise TypeError(
                f"provider must implement BatchDataProvider or PointDataProvider, "
                f"got {type(provider).__name__}"
            )

        logger.debug(
            "ExternalDataOracle initialized in %s mode for symbols %s",
            self._mode,
            self.symbols_list,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_batch(self, provider: BatchDataProvider) -> None:
        """Eagerly load the full series for every symbol."""
        self.r: dict[str, pd.Series] = {}
        self._linear_xs: dict[str, np.ndarray] = {}
        self._linear_ys: dict[str, np.ndarray] = {}
        self._linear_base: dict[str, int] = {}
        for symbol in self.symbols_list:
            series = provider.get_fundamental_series(
                symbol, self.mkt_open, self.mkt_close
            )
            if series.empty:
                raise ValueError(
                    f"Provider returned empty series for symbol '{symbol}'"
                )
            # Ensure sorted DatetimeIndex
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index, unit="ns")
            series = series.sort_index()

            # For LINEAR, store numpy arrays for np.interp at query time
            # (O(log n) per lookup instead of materializing a nanosecond index).
            # Subtract base timestamp to avoid float64 precision loss at ~1.6e18.
            if self.interpolation == InterpolationStrategy.LINEAR:
                raw_xs = series.index.astype(np.int64).values
                self._linear_base[symbol] = raw_xs[0]
                self._linear_xs[symbol] = (raw_xs - raw_xs[0]).astype(float)
                self._linear_ys[symbol] = series.values.astype(float)

            self.r[symbol] = series

    def _init_point(self, provider: PointDataProvider, cache_size: int) -> None:
        """Set up lazy per-symbol LRU caches."""
        self._provider = provider
        self._caches: dict[str, _LRUCache] = {
            s: _LRUCache(maxsize=cache_size) for s in self.symbols_list
        }

    # ------------------------------------------------------------------
    # Oracle API
    # ------------------------------------------------------------------

    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """Return the fundamental value at market open."""
        if self._mode == "batch":
            series = self.r[symbol]
            ts = pd.Timestamp(mkt_open, unit="ns")
            if self.interpolation == InterpolationStrategy.FORWARD_FILL:
                value = series.asof(ts)
            elif self.interpolation == InterpolationStrategy.NEAREST:
                idx = series.index.get_indexer([ts], method="nearest")[0]
                value = series.iloc[idx]
            elif self.interpolation == InterpolationStrategy.LINEAR:
                value = int(
                    round(
                        np.interp(
                            mkt_open - self._linear_base[symbol],
                            self._linear_xs[symbol],
                            self._linear_ys[symbol],
                        )
                    )
                )
            else:
                value = series.asof(ts)
            return int(value)
        else:
            return self._point_lookup(symbol, mkt_open)

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 0,
    ) -> int:
        """Return a (potentially noisy) observation of the fundamental value.

        Parameters
        ----------
        symbol:
            The equity symbol to query.
        current_time:
            Current simulation time in nanoseconds.
        random_state:
            Agent-specific RNG for reproducible noisy observations.
        sigma_n:
            Observation noise variance.  Set to 0 for exact values.
            NOTE: this is *variance*, not standard deviation.
        """
        # Clamp to market close
        t = min(current_time, self.mkt_close - 1)

        if self._mode == "batch":
            r_t = self._batch_lookup(symbol, t)
        else:
            r_t = self._point_lookup(symbol, t)

        # Apply observation noise
        obs = (
            r_t
            if sigma_n == 0
            else int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))
        )

        logger.debug(
            "ExternalDataOracle: fundamental=%d at t=%d, observation=%d",
            r_t,
            current_time,
            obs,
        )
        return obs

    # ------------------------------------------------------------------
    # Internal lookup helpers
    # ------------------------------------------------------------------

    def _batch_lookup(self, symbol: str, timestamp: NanosecondTime) -> int:
        """Look up a value from the in-memory series."""
        series = self.r[symbol]
        ts = pd.Timestamp(timestamp, unit="ns")

        if self.interpolation == InterpolationStrategy.FORWARD_FILL:
            value = series.asof(ts)
        elif self.interpolation == InterpolationStrategy.NEAREST:
            idx = series.index.get_indexer([ts], method="nearest")[0]
            value = series.iloc[idx]
        elif self.interpolation == InterpolationStrategy.LINEAR:
            value = int(
                round(
                    np.interp(
                        timestamp - self._linear_base[symbol],
                        self._linear_xs[symbol],
                        self._linear_ys[symbol],
                    )
                )
            )
        else:
            value = series.asof(ts)

        if pd.isna(value):
            raise ValueError(
                f"No data available for '{symbol}' at or before "
                f"timestamp {timestamp} (interpolation={self.interpolation.name})"
            )
        return int(value)

    def _point_lookup(self, symbol: str, timestamp: NanosecondTime) -> int:
        """Look up a value via the point provider, with LRU caching."""
        cache = self._caches[symbol]
        cached = cache.get(timestamp)
        if cached is not None:
            return cached

        value = self._provider.get_fundamental_at(symbol, timestamp)
        cache.put(timestamp, value)
        return value
