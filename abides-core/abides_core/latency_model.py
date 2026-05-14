"""Latency models for kernel message dispatch.

Hierarchy
---------

- :class:`LatencyModel` is an abstract base class. The kernel only ever
  calls :meth:`LatencyModel.get_latency` on a concrete subclass.
- :class:`CubicLatencyModel` — original cubic-jitter behaviour, owns
  ``min_latency``, ``connected``, ``jitter``, ``jitter_clip``,
  ``jitter_unit``. Stochastic per call.
- :class:`DeterministicLatencyModel` — fixed ``min_latency`` matrix /
  scalar. No RNG draws.
- :class:`UniformLatencyModel` — a single integer for every pair. No
  RNG draws.
- :class:`MatrixLatencyModel` — a per-pair integer matrix. No RNG
  draws.
- :class:`MessageTypeAwareLatencyModel` — wraps a mapping
  ``type[Message] -> LatencyModel`` plus a default. Routes each call
  by the message class.

Construction goes through :class:`LatencyModelFactory` for
configuration-driven setups; subclasses can also be instantiated
directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

from .message import Message


def _extract(param: float | bool | np.ndarray, sid: int, rid: int):
    """Pick the value for sender ``sid`` / recipient ``rid`` from a
    parameter that may be scalar, 1-D (sender-indexed), or 2-D
    (sender-row, recipient-column).
    """
    if np.isscalar(param):
        return param
    if isinstance(param, np.ndarray):
        if param.ndim == 1:
            return param[sid]
        if param.ndim == 2:
            return param[sid, rid]
    raise TypeError(
        "LatencyModel parameter must be scalar, 1-D ndarray, or 2-D ndarray "
        f"(got {type(param).__name__})"
    )


class LatencyModel(ABC):
    """Abstract base for all latency models.

    Subclasses must implement :meth:`get_latency`. Every subclass must
    accept the ``message_class`` keyword (and may ignore it) so the
    kernel can pass it unconditionally.
    """

    @abstractmethod
    def get_latency(
        self,
        sender_id: int,
        recipient_id: int,
        *,
        message_class: type[Message] | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> int:
        """Return latency (in nanoseconds) for one message from
        ``sender_id`` to ``recipient_id``.

        ``message_class`` may be inspected by routing models;
        non-routing models ignore it. ``random_state`` is the kernel
        RNG; deterministic models ignore it.
        """


class CubicLatencyModel(LatencyModel):
    """Cubic-jitter model.

    Final latency for one message is
    ``min_latency + (jitter / x**3) * (min_latency / jitter_unit)``,
    where ``x`` is uniformly drawn from ``(jitter_clip, 1]``.

    Arguments:
      min_latency: Pairwise minimum latency. Scalar, 1-D (per sender),
        or 2-D (sender-row, recipient-column). Integer nanoseconds.
      connected: ``True`` or a boolean array. ``False`` for a (sender,
        recipient) pair returns ``-1`` (unreachable).
      jitter: Cubic ``a`` parameter in ``[0, 1]``. Higher = wider tail.
      jitter_clip: Lower bound of the uniform draw, in ``[0, 1)``.
        Higher = tighter tail.
      jitter_unit: Fraction of ``min_latency`` taken as the jitter
        unit.
    """

    def __init__(
        self,
        *,
        min_latency: float | np.ndarray,
        connected: bool | np.ndarray = True,
        jitter: float | np.ndarray = 0.5,
        jitter_clip: float | np.ndarray = 0.1,
        jitter_unit: float | np.ndarray = 10.0,
    ) -> None:
        self.min_latency = min_latency
        self.connected = connected
        self.jitter = jitter
        self.jitter_clip = jitter_clip
        self.jitter_unit = jitter_unit

    def get_latency(
        self,
        sender_id: int,
        recipient_id: int,
        *,
        message_class: type[Message] | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> int:
        if random_state is None:
            raise ValueError("CubicLatencyModel requires random_state for jitter draw")
        if not _extract(self.connected, sender_id, recipient_id):
            return -1
        min_latency = _extract(self.min_latency, sender_id, recipient_id)
        a = _extract(self.jitter, sender_id, recipient_id)
        clip = _extract(self.jitter_clip, sender_id, recipient_id)
        unit = _extract(self.jitter_unit, sender_id, recipient_id)
        x = random_state.uniform(low=clip, high=1.0)
        return int(min_latency + ((a / x**3) * (min_latency / unit)))


class DeterministicLatencyModel(LatencyModel):
    """Returns ``min_latency`` for every pair, with no randomness.

    Arguments:
      min_latency: Scalar, 1-D, or 2-D pairwise minimum latency in
        integer nanoseconds.
    """

    def __init__(self, *, min_latency: float | np.ndarray) -> None:
        self.min_latency = min_latency

    def get_latency(
        self,
        sender_id: int,
        recipient_id: int,
        *,
        message_class: type[Message] | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> int:
        return int(_extract(self.min_latency, sender_id, recipient_id))


class UniformLatencyModel(LatencyModel):
    """Returns a single configured constant for every pair. No RNG."""

    def __init__(self, latency: int | float) -> None:
        self._latency: int = int(latency)

    def get_latency(
        self,
        sender_id: int,
        recipient_id: int,
        *,
        message_class: type[Message] | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> int:
        return self._latency


class MatrixLatencyModel(LatencyModel):
    """Latency from a per-pair integer matrix ``M[sender, recipient]``.
    No RNG.
    """

    def __init__(
        self,
        matrix: np.ndarray | list[list[int]] | list[list[float]],
    ) -> None:
        self._matrix: np.ndarray = np.ascontiguousarray(matrix, dtype=np.int64)

    def get_latency(
        self,
        sender_id: int,
        recipient_id: int,
        *,
        message_class: type[Message] | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> int:
        return int(self._matrix[sender_id, recipient_id])


class MessageTypeAwareLatencyModel(LatencyModel):
    """Routes each call to a concrete model based on ``message_class``.

    Arguments:
      default: Model used when ``message_class`` is ``None`` or not in
        ``mapping``.
      mapping: ``type[Message] -> LatencyModel``. Lookup is by exact
        class; subclass dispatch is not performed.
    """

    def __init__(
        self,
        *,
        default: LatencyModel,
        mapping: Mapping[type[Message], LatencyModel],
    ) -> None:
        self._default = default
        self._mapping: dict[type[Message], LatencyModel] = dict(mapping)

    def get_latency(
        self,
        sender_id: int,
        recipient_id: int,
        *,
        message_class: type[Message] | None = None,
        random_state: np.random.RandomState | None = None,
    ) -> int:
        model = (
            self._mapping.get(message_class, self._default)
            if message_class is not None
            else self._default
        )
        return model.get_latency(
            sender_id,
            recipient_id,
            message_class=message_class,
            random_state=random_state,
        )


class LatencyModelFactory:
    """Single entry point for configuration-driven latency model
    construction. Each classmethod centralises the relevant validation.
    """

    @classmethod
    def cubic(
        cls,
        *,
        min_latency: float | np.ndarray,
        connected: bool | np.ndarray = True,
        jitter: float | np.ndarray = 0.5,
        jitter_clip: float | np.ndarray = 0.1,
        jitter_unit: float | np.ndarray = 10.0,
    ) -> CubicLatencyModel:
        return CubicLatencyModel(
            min_latency=min_latency,
            connected=connected,
            jitter=jitter,
            jitter_clip=jitter_clip,
            jitter_unit=jitter_unit,
        )

    @classmethod
    def deterministic(
        cls, *, min_latency: float | np.ndarray
    ) -> DeterministicLatencyModel:
        return DeterministicLatencyModel(min_latency=min_latency)

    @classmethod
    def uniform(cls, latency: int | float) -> UniformLatencyModel:
        return UniformLatencyModel(latency=latency)

    @classmethod
    def matrix(
        cls,
        matrix: np.ndarray | list[list[int]] | list[list[float]],
    ) -> MatrixLatencyModel:
        return MatrixLatencyModel(matrix=matrix)

    @classmethod
    def by_message_type(
        cls,
        *,
        default: LatencyModel,
        mapping: Mapping[type[Message], LatencyModel],
    ) -> MessageTypeAwareLatencyModel:
        return MessageTypeAwareLatencyModel(default=default, mapping=mapping)
