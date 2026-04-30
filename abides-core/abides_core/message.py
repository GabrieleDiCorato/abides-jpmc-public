import warnings
from dataclasses import dataclass


@dataclass
class Message:
    """The base Message class no longer holds envelope/header information, however any
    desired information can be placed in the arbitrary body.

    Delivery metadata is now handled outside the message itself.

    The body may be overridden by specific message type subclasses.
    """

    @property
    def message_id(self) -> int:
        """Deprecated: use the kernel's heap entries for ordering instead.

        Returns ``id(self)`` so existing callers do not crash. Heap
        ordering inside the kernel uses a per-kernel sequence counter
        embedded in ``_HeapEntry`` and never reads this attribute.
        """
        warnings.warn(
            "Message.message_id is deprecated and will be removed; "
            "kernel heap ordering uses a per-kernel sequence counter.",
            DeprecationWarning,
            stacklevel=2,
        )
        return id(self)

    def type(self) -> str:
        return self.__class__.__name__


@dataclass
class MessageBatch(Message):
    """
    Helper used for batching multiple messages being sent by the same sender to the same
    destination together. If very large numbers of messages are being sent this way,
    using this class can help performance.
    """

    messages: list[Message]


@dataclass
class WakeupMsg(Message):
    """
    Empty message sent to agents when woken up.
    """

    pass
