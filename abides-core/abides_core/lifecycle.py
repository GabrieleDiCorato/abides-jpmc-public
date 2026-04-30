"""Kernel lifecycle state machine.

The kernel transitions through four states::

    CONSTRUCTED  -- after __init__
       │
       ▼  initialize()
    INITIALIZED  -- ready to dispatch
       │
       ▼  runner()
    RUNNING      -- inside the event loop (gym yields keep state at RUNNING)
       │
       ▼  terminate()
    TERMINATED   -- end of simulation; reset() loops back to INITIALIZED

Transitions are checked at the public method boundary in :class:`Kernel`.
Out-of-order calls raise :class:`RuntimeError` rather than silently
corrupting kernel state.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Iterable


class KernelState(Enum):
    """Kernel lifecycle states. See module docstring for transitions."""

    CONSTRUCTED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    TERMINATED = auto()


def assert_transition(
    current: KernelState, allowed: Iterable[KernelState], action: str
) -> None:
    """Raise :class:`RuntimeError` if ``current`` is not in ``allowed``.

    ``action`` is the name of the public method being invoked; used in
    the error message to make the diagnosis trivial.
    """
    allowed_set = set(allowed)
    if current not in allowed_set:
        names = sorted(s.name for s in allowed_set)
        raise RuntimeError(
            f"Kernel.{action}() not allowed in state {current.name}; "
            f"allowed states: {names}."
        )
