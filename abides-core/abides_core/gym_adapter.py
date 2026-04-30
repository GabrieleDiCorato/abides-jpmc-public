"""Protocol describing the kernel's contract with a gym experimental agent.

The :class:`Kernel` does not import :mod:`abides_gym`; instead it accepts
any object that satisfies :class:`GymAdapter`. ABIDES-gym agents already
implement these methods, so registration is a one-line change in the
gym environment.
"""

from __future__ import annotations

from typing import Any, Protocol


class GymAdapter(Protocol):
    """Minimal contract a gym experimental agent must satisfy."""

    def update_raw_state(self) -> None:
        """Recompute the raw state cached on the adapter."""

    def get_raw_state(self) -> Any:
        """Return the most recent raw state.

        The kernel does not interpret this value; it is forwarded to the
        gym environment which converts it into observations / rewards.
        """

    def apply_actions(self, actions: list[dict[str, Any]]) -> None:
        """Apply a batch of gym actions before resuming the event loop."""
