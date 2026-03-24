import abc
from typing import Any

import numpy as np

from abides_core import NanosecondTime


class Oracle(abc.ABC):
    f_log: dict[str, list[dict[str, Any]]] = {}
    """Fundamental value log.  Subclasses that track history override in __init__."""

    @abc.abstractmethod
    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime, cents: bool = True
    ) -> int:
        pass

    @abc.abstractmethod
    def observe_price(
        self,
        symbol: str,
        current_time: NanosecondTime,
        random_state: np.random.RandomState,
        sigma_n: int = 1000,
    ) -> int:
        pass
