import datetime as dt
import logging
import warnings
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from abides_core import NanosecondTime

from .oracle import Oracle

logger = logging.getLogger(__name__)

_MAX_STEPS = 1_000_000


class MeanRevertingOracle(Oracle):
    """The MeanRevertingOracle requires three parameters: a mean fundamental value,
    a mean reversion coefficient, and a shock variance.  It constructs and retains
    a fundamental value time series for each requested symbol, and provides noisy
    observations of those values upon agent request.  The expectation is that
    agents using such an oracle will know the mean-reverting equation and all
    relevant parameters, but will not know the random shocks applied to the
    sequence at each time step.

    Historical dates are effectively meaningless to this oracle.  It is driven by
    the numpy random number seed contained within the experimental config file.
    This oracle uses the nanoseconds portion of the current simulation time as
    discrete "time steps".  A suggestion: to keep wallclock runtime reasonable,
    have the agents operate for only ~1000 nanoseconds, but interpret nanoseconds
    as seconds or minutes.

    .. deprecated::
        Use :class:`SparseMeanRevertingOracle` instead.  This oracle pre-generates
        a nanosecond-resolution time series and will OOM for any realistic trading
        day (6.5 h ≈ 2.34×10¹³ steps).
    """

    def __init__(
        self,
        mkt_open: NanosecondTime,
        mkt_close: NanosecondTime,
        symbols: dict[str, dict[str, Any]],
        random_state: np.random.RandomState,
    ) -> None:
        warnings.warn(
            "MeanRevertingOracle is deprecated — use SparseMeanRevertingOracle "
            "for real-time-scale simulations.",
            DeprecationWarning,
            stacklevel=2,
        )

        n_steps = mkt_close - mkt_open
        if n_steps > _MAX_STEPS:
            raise ValueError(
                f"MeanRevertingOracle: time range of {n_steps:,} ns exceeds the "
                f"maximum of {_MAX_STEPS:,}.  Pre-generating a nanosecond-resolution "
                f"series this large would exhaust memory.  "
                f"Use SparseMeanRevertingOracle instead."
            )

        # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
        # inner keys: r_bar, kappa, sigma_s.
        self.mkt_open: NanosecondTime = mkt_open
        self.mkt_close: NanosecondTime = mkt_close
        self.symbols: dict[str, dict[str, Any]] = symbols
        self.random_state: np.random.RandomState = random_state

        # The dictionary r holds the fundamenal value series for each symbol.
        self.r: dict[str, pd.Series] = {}

        then = dt.datetime.now()

        for symbol in symbols:
            s = symbols[symbol]
            logger.debug(
                "MeanRevertingOracle computing fundamental value series for {}", symbol
            )
            self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)

        now = dt.datetime.now()

        logger.debug("MeanRevertingOracle initialized for symbols {}", symbols)
        logger.debug("MeanRevertingOracle initialization took {}", now - then)

    def generate_fundamental_value_series(
        self, symbol: str, r_bar: int, kappa: float, sigma_s: float
    ) -> pd.Series:
        """Generates the fundamental value series for a single stock symbol.

        Arguments:
            symbol: The symbold to calculate the fundamental value series for.
            r_bar: The mean fundamental value.
            kappa: The mean reversion coefficient.
            sigma_s: The shock variance.  (Note: NOT STANDARD DEVIATION)

        Because the oracle uses the global np.random PRNG to create the
        fundamental value series, it is important to create the oracle BEFORE
        the agents.  In this way the addition of a new agent will not affect the
        sequence created.  (Observations using the oracle will use an agent's
        PRNG and thus not cause a problem.)
        """

        # Turn variance into std.
        sigma_s = sqrt(sigma_s)

        # Create the time series into which values will be projected and initialize the first value.
        date_range = pd.date_range(
            self.mkt_open, self.mkt_close, inclusive="left", freq="ns"
        )

        s = pd.Series(index=date_range)
        r = np.zeros(len(s.index))
        r[0] = r_bar

        # Predetermine the random shocks for all time steps (at once, for computation speed).
        shock = self.random_state.normal(scale=sigma_s, size=(r.shape[0]))

        # Compute the mean reverting fundamental value series.
        for t in range(1, r.shape[0]):
            r[t] = max(0, (kappa * r_bar) + ((1 - kappa) * r[t - 1]) + shock[t])

        # Replace the series values with the fundamental value series.  Round and convert to
        # integer cents.
        s[:] = np.round(r)

        return s.astype(int)

    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        """Return the daily open price for the symbol given.

        In the case of the MeanRevertingOracle, this will simply be the first
        fundamental value, which is also the fundamental mean. We will use the
        mkt_open time as given, however, even if it disagrees with this.
        """

        # If we did not already know mkt_open, we should remember it.
        if (mkt_open is not None) and (self.mkt_open is None):
            self.mkt_open = mkt_open

        logger.debug(
            "Oracle: client requested {symbol} at market open: {}", self.mkt_open
        )

        open_price = self.r[symbol].loc[pd.Timestamp(self.mkt_open, unit="ns")]
        logger.debug("Oracle: market open price was was {}", open_price)

        return int(open_price)

    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int:
        """Return a noisy observation of the current fundamental value.

        While the fundamental value for a given equity at a given time step does
        not change, multiple agents observing that value will receive different
        observations.

        Only the Exchange or other privileged agents should use noisy=False.

        sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.

        Each agent must pass its RandomState object to ``observe_price``.  This
        ensures that each agent will receive the same answers across multiple
        same-seed simulations even if a new agent has been added to the experiment.
        """

        # If the request is made after market close, return the close price.
        if current_time >= self.mkt_close:
            r_t = self.r[symbol].loc[pd.Timestamp(self.mkt_close - 1, unit="ns")]
        else:
            r_t = self.r[symbol].loc[pd.Timestamp(current_time, unit="ns")]

        # Generate a noisy observation of fundamental value at the current time.
        obs = (
            r_t
            if sigma_n == 0
            else int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))
        )

        logger.debug("Oracle: current fundamental value is {} at {}", r_t, current_time)
        logger.debug("Oracle: giving client value observation {}", obs)

        # Reminder: all simulator prices are specified in integer cents.
        return obs
