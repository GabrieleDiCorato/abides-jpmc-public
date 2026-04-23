"""
General purpose utility functions for the simulator, attached to no particular class.
Available to any agent or other module/utility.  Should not require references to
any simulator object (kernel, agent, etc).
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from . import NanosecondTime


def subdict(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """
    Returns a dictionnary with only the keys defined in the keys list
    Arguments:
        - d: original dictionnary
        - keys: list of keys to keep
    Returns:
        - dictionnary with only the subset of keys
    """
    return {k: v for k, v in d.items() if k in keys}


def restrictdict(d: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """
    Returns a dictionnary with only the intersections of the keys defined in the keys list and the keys in the o
    Arguments:
        - d: original dictionnary
        - keys: list of keys to keep
    Returns:
        - dictionnary with only the subset of keys
    """
    inter = [k for k in d if k in keys]
    return subdict(d, inter)


def custom_eq(a: Any, b: Any) -> bool:
    """returns a==b or True if both a and b are null"""
    return bool((a == b) | ((a != a) & (b != b)))


# Utility function to get agent wake up times to follow a U-quadratic distribution.
def get_wake_time(open_time, close_time, random_state, a=0, b=1) -> NanosecondTime:
    """
    Draw a time U-quadratically distributed between open_time and close_time.

    For details on U-quadtratic distribution see https://en.wikipedia.org/wiki/U-quadratic_distribution.

    Args:
        open_time: Market opening time
        close_time: Market closing time
        random_state: numpy RandomState instance for reproducibility
        a: Lower time limit for distribution support
        b: Upper time limit for distribution support
    """
    if random_state is None:
        raise TypeError("random_state is required for get_wake_time")

    def cubic_pow(n: float) -> float:
        """Helper function: returns *real* cube root of a float."""

        if n < 0:
            return float(-((-n) ** (1.0 / 3.0)))
        else:
            return float(n ** (1.0 / 3.0))

    #  Use inverse transform sampling to obtain variable sampled from U-quadratic
    def u_quadratic_inverse_cdf(y):
        alpha = 12 / ((b - a) ** 3)
        beta = (b + a) / 2
        result = cubic_pow((3 / alpha) * y - (beta - a) ** 3) + beta
        return result

    uniform_0_1 = random_state.rand()
    random_multiplier = u_quadratic_inverse_cdf(uniform_0_1)
    wake_time = open_time + random_multiplier * (close_time - open_time)

    return int(wake_time)


def fmt_ts(timestamp: NanosecondTime) -> str:
    """
    Converts a timestamp stored as nanoseconds into a human readable string.
    """
    return pd.Timestamp(timestamp, unit="ns").strftime("%Y-%m-%d %H:%M:%S")  # type: ignore[no-any-return]


def str_to_ns(string: str | NanosecondTime) -> NanosecondTime:
    """
    Converts a human readable time-delta string into nanoseconds.

    Arguments:
        string: String to convert into nanoseconds. Uses Pandas to do this.

    Examples:
        - "1s" -> 1e9 ns
        - "1min" -> 6e10 ns
        - "00:00:30" -> 3e10 ns
    """
    import re

    # If already a numeric type, return as int64
    if isinstance(string, (int, float, np.integer, np.floating)):
        return int(string)

    # Handle 'm' as minutes (pandas treats 'm' as months in newer versions)
    # Replace standalone 'm' with 'min' for minute interpretation
    # But don't replace 'ms' (milliseconds), 'min', 'minute', etc.
    if re.match(r"^[\d.]+m$", string.lower()):
        string = string[:-1] + "min"

    # Handle 'd' as days (deprecated in pandas)
    if re.match(r"^[\d.]+d$", string.lower()):
        string = string[:-1] + "D"

    # Handle uppercase 'S' as seconds (deprecated in pandas 2.2+, use lowercase 's')
    if re.match(r"^[\d.]+S$", string):
        string = string[:-1] + "s"

    return int(pd.to_timedelta(string).value)


def datetime_str_to_ns(string: str) -> NanosecondTime:
    """
    Takes a datetime written as a string and returns in nanosecond unix timestamp.

    Arguments:
        string: String to convert into nanoseconds. Uses Pandas to do this.
    """
    return int(pd.Timestamp(string).value)


def ns_date(ns_datetime: NanosecondTime) -> NanosecondTime:
    """
    Takes a datetime in nanoseconds unix timestamp and rounds it to that day at 00:00.

    Arguments:
        ns_datetime: Nanosecond time value to round.
    """
    return ns_datetime - (ns_datetime % (24 * 3600 * int(1e9)))


def parse_logs_df(end_state: dict) -> pd.DataFrame:
    """
    Takes the end_state dictionnary returned by an ABIDES simulation goes through all
    the agents, extracts their log, and un-nest them returns a single dataframe with the
    logs from all the agents warning: this is meant to be used for debugging and
    exploration.
    """
    agents = end_state["agents"]
    dfs = []
    for agent in agents:
        messages = []
        for m in agent.log:
            m = {
                "EventTime": m[0] if isinstance(m[0], (int, np.int64)) else 0,
                "EventType": m[1],
                "Event": m[2],
            }
            event = m.get("Event")
            if event is None:
                event = {"EmptyEvent": True}
            elif not isinstance(event, dict):
                event = {"ScalarEventValue": event}
            else:
                pass
            with contextlib.suppress(KeyError):
                del m["Event"]
            m.update(event)
            if m.get("agent_id") is None:
                m["agent_id"] = agent.id
            m["agent_type"] = agent.type
            messages.append(m)
        dfs.append(pd.DataFrame(messages))

    return pd.concat(dfs)


# caching utils: not used by abides but useful to have
def input_sha_wrapper(func: Callable) -> Callable:
    """
    compute a sha for the function call by looking at function name and inputs for the call
    """

    def inner(*args, **kvargs):
        argspec = inspect.getfullargspec(func)
        index_first_kv = len(argspec.args) - (
            len(argspec.defaults) if argspec.defaults is not None else 0
        )
        if len(argspec.args) > 0:
            total_kvargs = dict(
                (k, v) for k, v in zip(argspec.args[index_first_kv:], argspec.defaults)
            )
        else:
            total_kvargs = {}
        total_kvargs.update(kvargs)
        input_sha = (
            func.__name__
            + "_"
            + hashlib.sha1(str.encode(str((args, total_kvargs)))).hexdigest()
        )
        return {"input_sha": input_sha}

    return inner


def cache_wrapper(
    func: Callable, cache_dir="cache/", force_recompute=False
) -> Callable:
    """
    local caching decorator
    checks the functional call sha is only there is specified directory
    """

    def inner(*args, **kvargs):
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        sha_call = input_sha_wrapper(func)(*args, **kvargs)
        cache_path = cache_dir + sha_call["input_sha"] + ".pkl"
        if os.path.isfile(cache_path) and not force_recompute:
            with open(cache_path, "rb") as handle:
                result = pickle.load(handle)
            return result
        else:
            result = func(*args, **kvargs)
            with open(cache_path, "wb") as handle:
                pickle.dump(result, handle)
            return result

    return inner
