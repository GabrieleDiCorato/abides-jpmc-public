import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import NanosecondTime
from .message import Message, MessageBatch
from .utils import fmt_ts

if TYPE_CHECKING:
    from abides_core.kernel import Kernel

logger = logging.getLogger(__name__)


class _UninitializedKernel:
    """Sentinel used as the placeholder ``Agent.kernel`` value before
    the kernel has attached itself via :meth:`Agent.kernel_initializing`.

    Any attribute access raises :class:`RuntimeError` with a clear
    message, surfacing the underlying bug (an agent method ran before
    kernel attach) instead of an opaque ``AttributeError`` on ``None``.
    The sentinel is typed as :class:`Kernel` at the attribute level so
    static type checkers do not have to deal with ``Kernel | None``
    everywhere.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Agent.kernel accessed before kernel_initializing(); attribute={name!r}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<UninitializedKernel>"


_UNINITIALIZED_KERNEL: "Kernel" = _UninitializedKernel()  # type: ignore[assignment]


class Agent:
    """
    Base Agent class

    Attributes:
        id: Must be a unique number (usually autoincremented).
        name: For human consumption, should be unique (often type + number).
        type: For machine aggregation of results, should be same for all agents
            following the same strategy (incl. parameter settings).
        random_state: an np.random.RandomState object, already seeded. Every agent
            is given a random state to use for any stochastic needs.
        log_events: flag to log or not the events during the simulation
        log_to_file: flag to write on disk or not the logged events
    """

    def __init__(
        self,
        id: int,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        log_events: bool = True,
        log_to_file: bool = True,
    ) -> None:
        self.id: int = id
        self.type: str = type or self.__class__.__name__
        self.name: str = name or f"{self.type}_{self.id}"
        if random_state is None:
            raise ValueError(f"random_state is required for agent {self.name}")
        self.random_state: np.random.RandomState = random_state
        self.log_events: bool = log_events
        self.log_to_file: bool = log_to_file & log_events

        # Kernel is supplied via kernel_initializing method of kernel lifecycle.
        # Sentinel default raises RuntimeError on any attribute access so
        # pre-attach use is loud, and static type-checkers see ``Kernel``
        # rather than ``Kernel | None``.
        self.kernel: Kernel = _UNINITIALIZED_KERNEL

        # What time does the agent think it is?  Should be updated each time
        # the agent wakes via wakeup or receive_message.  (For convenience
        # of reference throughout the Agent class hierarchy, NOT THE
        # CANONICAL TIME.)
        self.current_time: NanosecondTime = 0

        # Agents may choose to maintain a log.  During simulation,
        # it should be stored as a list of dictionaries.  The expected
        # keys by default are: EventTime, EventType, Event.  Other
        # Columns may be added, but will then require specializing
        # parsing and will increase output dataframe size.  If there
        # is a non-empty log, it will be written to disk as a Dataframe
        # at kernel termination.

        # It might, or might not, make sense to formalize these log Events
        # as a class, with enumerated EventTypes and so forth.
        self.log: list[tuple[NanosecondTime, str, Any]] = []

        self.logEvent("AGENT_TYPE", type)

    ### Flow of required kernel listening methods:
    ### init -> start -> (entire simulation) -> end -> terminate

    def kernel_initializing(self, kernel: "Kernel") -> None:
        """
        Called by the kernel one time when simulation first begins.

        No other agents are guaranteed to exist at this time.

        Kernel reference must be retained, as this is the only time the agent can
        "see" it.

        Arguments:
            kernel: The Kernel instance running the experiment.
        """

        self.kernel = kernel

        logger.debug(f"{self.name} exists!")

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        """
        Called by the kernel one time after simulationInitializing.

        All other agents are guaranteed to exist at this time.

        Base Agent schedules a wakeup call for the first available timestamp.
        Subclass agents may override this behavior as needed.

        Arguments:
            start_time: The earliest time for which the agent can schedule a wakeup call
                (or could receive a message).
        """

        logger.debug(
            f"Agent {self.id} ({self.name}) requesting kernel wakeup at time {fmt_ts(start_time)}"
        )

        self.set_wakeup(start_time)

    def kernel_stopping(self) -> None:
        """
        Called by the kernel one time before simulationTerminating.

        All other agents are guaranteed to exist at this time.
        """

        pass

    def kernel_terminating(self) -> None:
        """
        Called by the kernel one time when simulation terminates.

        No other agents are guaranteed to exist at this time.
        """

        # If this agent has been maintaining a log, convert it to a Dataframe
        # and request that the Kernel write it to disk before terminating.
        if self.log and self.log_to_file:
            df_log = pd.DataFrame(self.log, columns=("EventTime", "EventType", "Event"))
            df_log.set_index("EventTime", inplace=True)
            self.write_log(df_log)

    ### Methods for internal use by agents (e.g. bookkeeping).

    def logEvent(
        self,
        event_type: str,
        event: Any = "",
        append_summary_log: bool = False,
        deepcopy_event: bool = False,
    ) -> None:
        """
        Adds an event to this agent's log.

        The deepcopy of the Event field, often an object, ensures later state
        changes to the object will not retroactively update the logged event.

        Arguments:
            event_type: label of the event (e.g., Order submitted, order accepted last trade etc....)
            event: actual event to be logged
            append_summary_log:
            deepcopy_event: Set to False to skip deepcopying the event object.
        """

        if not self.log_events:
            return

        # We can make a single copy of the object (in case it is an arbitrary
        # class instance) for both potential log targets, because we don't
        # alter logs once recorded.
        if deepcopy_event:
            event = deepcopy(event)

        self.log.append((self.current_time, event_type, event))

        if append_summary_log:
            self.kernel.append_summary_log(self.id, event_type, event)

    ### Methods required for communication from other agents.
    ### The kernel will _not_ call these methods on its own behalf,
    ### only to pass traffic from other agents..

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """
        Called each time a message destined for this agent reaches the front of the
        kernel's priority queue.

        Arguments:
            current_time: The simulation time at which the kernel is delivering this
                message -- the agent should treat this as "now".
            sender_id: The ID of the agent who sent the message.
            message: An object guaranteed to inherit from the message.Message class.
        """

        self.current_time = current_time

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"At {fmt_ts(current_time)}, agent {self.id} ({self.name}) received: {message}"
            )

    def wakeup(self, current_time: NanosecondTime) -> None:
        """
        Agents can request a wakeup call at a future simulation time using
        ``Agent.set_wakeup()``.

        This is the method called when the wakeup time arrives.

        Arguments:
            current_time: The simulation time at which the kernel is delivering this
                message -- the agent should treat this as "now".
        """

        self.current_time = current_time

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"At {fmt_ts(current_time)}, agent {self.id} ({self.name}) received wakeup."
            )

    ### Presently the kernel expects agent IDs only, not agent references.
    ### It is possible this could change in the future.  Normal agents will
    ### not typically wish to request additional delay.
    def send_message(self, recipient_id: int, message: Message, delay: int = 0) -> None:
        """
        Sends a message to another Agent.

        Arguments:
            recipient_id: ID of the agent receiving the message.
            message: The ``Message`` class instance to send.
            delay: Represents an agent's request for ADDITIONAL delay (beyond the
                Kernel's mandatory computation + latency delays). Represents parallel
                pipeline processing delays (that should delay the transmission of
                messages but do not make the agent "busy" and unable to respond to new
                messages)
        """

        self.kernel.send_message(self.id, recipient_id, message, delay=delay)

    def send_message_batch(
        self, recipient_id: int, messages: list[Message], delay: NanosecondTime = 0
    ) -> None:
        """
        Sends a batch of messages to another Agent.

        Arguments:
            recipient_id: ID of the agent receiving the messages.
            messages: A list of ``Message`` class instances to send.
            delay: Represents an agent's request for ADDITIONAL delay (beyond the
            Kernel's mandatory computation + latency delays). Represents parallel
            pipeline processing delays (that should delay the transmission of messages
            but do not make the agent "busy" and unable to respond to new messages)
        """

        self.kernel.send_message(
            self.id, recipient_id, MessageBatch(messages), delay=delay
        )

    def set_wakeup(self, requested_time: NanosecondTime) -> None:
        """
        Called to receive a "wakeup call" from the kernel at some requested future time.

        Arguments:
            requested_time: Defaults to the next possible timestamp. Wakeup time cannot
                be the current time or a past time.
        """

        self.kernel.set_wakeup(self.id, requested_time)

    @property
    def computation_delay(self) -> int:
        """The agent's per-action computation delay in nanoseconds.

        Reads directly from the kernel's contiguous ``int64`` storage.
        """
        return int(self.kernel._agent_computation_delays[self.id])

    @computation_delay.setter
    def computation_delay(self, value: int) -> None:
        self.kernel.set_agent_compute_delay(self.id, value)

    def get_computation_delay(self):
        """Queries thr agent's current computation delay from the kernel."""

        return self.computation_delay

    def set_computation_delay(self, requested_delay: int) -> None:
        """
        Calls the kernel to update the agent's computation delay.

        This does not initiate a global delay, nor an immediate delay for the agent.
        Rather it sets the new default delay for the calling agent. The delay will be
        applied upon every return from wakeup or recvMsg.

        Note that this delay IS applied to any messages sent by the agent during the
        current wake cycle (simulating the messages popping out at the end of its
        "thinking" time).

        Also note that we DO permit a computation delay of zero, but this should really
        only be used for special or massively parallel agents.

        Arguments:
            requested_delay: delay given in nanoseconds.
        """

        self.kernel.set_agent_compute_delay(
            sender_id=self.id, requested_delay=requested_delay
        )

    def delay(self, additional_delay: int) -> None:
        """
        Accumulates a temporary delay for the current wake cycle for this agent.

        This will apply the total delay (at time of send_message) to each message, and
        will modify the agent's next available time slot.  These happen on top of the
        agent's compute delay BUT DO NOT ALTER IT.  (i.e. effects are transient). Mostly
        useful for staggering outbound messages.

        Arguments:
            additional_delay: additional delay given in nanoseconds.
        """

        self.kernel.delay_agent(sender_id=self.id, additional_delay=additional_delay)

    def report_metric(self, key: str, value: float) -> None:
        """
        Dispatch a numeric metric to every :class:`KernelObserver`
        registered on the kernel. The kernel itself never aggregates;
        observers decide what to do with the value.

        Arguments:
            key: short metric name (e.g. ``"ending_value"``).
            value: numeric value, cast to ``float`` by each observer.
        """
        for observer in self.kernel._observers:
            observer.on_metric(self.id, self.type, key, float(value))

    def write_log(self, df_log: pd.DataFrame, filename: str | None = None) -> None:
        """
        Called by the agent, usually at the very end of the simulation just before
        kernel shutdown, to write to disk any log dataframe it has been accumulating
        during simulation.

        The format can be decided by the agent, although changes will require a special
        tool to read and parse the logs.  The Kernel places the log in a unique
        directory per run, with one filename per agent, also decided by the Kernel using
        agent type, id, etc.

        If filename is None the Kernel will construct a filename based on the name of
        the Agent requesting log archival.

        Arguments:
            df_log: dataframe that contains all the logged events during the simulation
            filename: Location on disk to write the log to.
        """

        self.kernel.write_log(self.id, df_log, filename)

    ### Internal methods that should not be modified without a very good reason.

    def __lt__(self, other) -> bool:
        # Required by Python3 for this object to be placed in a priority queue.
        return f"{self.id}" < f"{other.id}"
