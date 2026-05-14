"""Protocol describing the kernel's contract with an outer-loop hook.

The :class:`Kernel` does not import :mod:`abides_gym`; instead it
accepts any object that satisfies :class:`RunnerHook`. ABIDES-gym
environments use this hook to step the simulation between agent
actions, but the contract is fully generic and not gym-specific.
"""

from __future__ import annotations

from typing import Any, Protocol


class RunnerHook(Protocol):
    """Minimal contract for an outer-loop hook driving the kernel."""

    def update_raw_state(self) -> None:
        """Recompute the raw state cached on the hook."""

    def get_raw_state(self) -> Any:
        """Return the most recent raw state.

        The kernel does not interpret this value; it forwards it to
        the caller of :meth:`Kernel.runner`.
        """

    def apply_actions(self, actions: list[dict[str, Any]]) -> None:
        """Apply a batch of actions before resuming the event loop."""
