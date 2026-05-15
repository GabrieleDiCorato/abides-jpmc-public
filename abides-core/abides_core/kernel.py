import heapq
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from . import NanosecondTime
from .agent import Agent
from .latency_model import LatencyModel, UniformLatencyModel
from .lifecycle import KernelState, assert_transition
from .log_writer import BZ2PickleLogWriter, LogWriter, NullLogWriter
from .message import Message, MessageBatch, WakeupMsg
from .observers import KernelObserver
from .oracle import Oracle
from .run_result import KernelRunResult
from .runner_hook import RunnerHook
from .utils import fmt_ts, str_to_ns

logger = logging.getLogger(__name__)

_DEFAULT_START_TIME: int = str_to_ns("09:30:00")
_DEFAULT_STOP_TIME: int = str_to_ns("16:00:00")


# Single shared instance: WakeupMsg carries no data, so reusing one
# instance saves an allocation per scheduled wakeup.
_WAKEUP_SINGLETON = WakeupMsg()


@dataclass(order=True)
class _HeapEntry:
    """Heap entry used by the kernel's event queue.

    Ordering is by ``(deliver_at, seq)`` only; sender / recipient /
    message are excluded from comparison so messages do not need to be
    orderable themselves. ``seq`` is a per-kernel monotonic counter
    that breaks ties deterministically and resets on
    ``Kernel.initialize()``.
    """

    deliver_at: int
    seq: int
    sender_id: int = field(compare=False)
    recipient_id: int = field(compare=False)
    message: Message = field(compare=False)


class Kernel:
    """
    ABIDES Kernel

    Arguments:
        agents: List of agents to include in the simulation.
        random_state: Required ``np.random.RandomState`` used for kernel-level
            stochastic operations (latency jitter, message tie-breaking when
            relevant). Callers must derive this from a master seed via the
            config-system seed-derivation helpers; the kernel never
            synthesizes one.
        start_time: Timestamp giving the start time of the simulation.
        stop_time: Timestamp giving the end time of the simulation.
        default_computation_delay: time penalty applied to an agent each time it is
            awakened (wakeup or recvMsg).
        default_latency: latency imposed on each computation, modeled physical latency in systems and avoid infinite loop of events happening at the same exact time (in ns)
        agent_latency_model: Model of latency used for the network of agents.
        skip_log: if True, no log saved on disk.
        log_dir: directory where data is store.
        oracle: Optional market-data oracle. The kernel does not call
            it directly; agents read ``self.kernel.oracle`` when they
            need fundamental prices.
        observers: Sequence of :class:`KernelObserver` instances that
            receive ``on_metric`` and ``on_terminate`` callbacks. The
            kernel never aggregates metrics itself.

    Invariant:
        ``agents[i].id == i`` for every agent. The kernel relies on this to
        index its parallel per-agent state arrays. Violations raise
        ``ValueError`` at construction time.
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        random_state: np.random.RandomState,
        start_time: NanosecondTime = _DEFAULT_START_TIME,
        stop_time: NanosecondTime = _DEFAULT_STOP_TIME,
        default_computation_delay: int = 1,
        default_latency: int = 1,
        agent_latency_model: LatencyModel | None = None,
        skip_log: bool = True,
        log_dir: str | None = None,
        log_root: str | os.PathLike = "./log",
        log_writer: LogWriter | None = None,
        oracle: Oracle | None = None,
        observers: Sequence[KernelObserver] = (),
        agent_computation_delays: np.ndarray | None = None,
        runner_hook: RunnerHook | None = None,
    ) -> None:
        # Enforce the agents[i].id == i invariant before anything else uses
        # the parallel per-agent state arrays.
        for idx, agent in enumerate(agents):
            if agent.id != idx:
                raise ValueError(
                    f"Kernel agents list violates agents[i].id == i invariant: "
                    f"agents[{idx}].id == {agent.id}"
                )

        # Typed first-class slots replace the v2 ``custom_properties``
        # bag. Oracle is consumed by exchange / value agents directly;
        # observers receive metric and terminate events.
        self.oracle: Oracle | None = oracle
        self._observers: tuple[KernelObserver, ...] = tuple(observers)

        self.random_state: np.random.RandomState = random_state

        # A single message queue to keep everything organized by increasing
        # delivery timestamp.
        self.messages: list[_HeapEntry] = []

        # Per-kernel monotonic counter used as the heap tiebreaker. Reset
        # by initialize() so seeded simulations are reproducible across
        # kernel.reset() and across parallel workers in the same process.
        self._next_seq: int = 0

        # Timestamp at which the Kernel was created.  Primarily used to
        # create a unique log directory for this run.  Also used to
        # print some elapsed time and messages per second statistics.
        self.kernel_wall_clock_start: datetime = datetime.now()

        # The Kernel maintains a summary log to which agents can write
        # information that should be centralized for very fast access
        # by separate statistical summary programs.  Detailed event
        # logging should go only to the agent's individual log.  This
        # is for things like "final position value" and such.
        self.summary_log: list[dict[str, Any]] = []

        # Lifecycle state machine. Transitions are validated at the public
        # method boundary (initialize / runner / terminate / reset). The
        # public ``Kernel.state`` property exposes this as the single source
        # of truth for run progress.
        self._state: KernelState = KernelState.CONSTRUCTED

        # agents must be a list of agents for the simulation,
        #        based on class agent.Agent
        self.agents: list[Agent] = agents

        # Runner hook slot. ``runner_hook=`` is the only registration path;
        # the kernel never auto-detects hook-shaped agents.
        self._runner_hook: RunnerHook | None = runner_hook

        logger.debug(
            "Runner hook %s",
            "registered" if self._runner_hook is not None else "not present",
        )

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        self.start_time: NanosecondTime = start_time
        self.stop_time: NanosecondTime = stop_time

        # This is a NanosecondTime that includes the date.
        self.current_time: NanosecondTime = start_time

        # Should the Kernel skip writing agent logs?
        self.skip_log: bool = skip_log

        # If a log directory was not specified, use the initial wallclock.
        self.log_dir: str = log_dir or str(
            int(self.kernel_wall_clock_start.timestamp())
        )

        # Resolve the log writer. Explicit ``log_writer=`` wins; otherwise
        # build the default disk writer or a no-op based on ``skip_log``.
        if log_writer is None:
            log_writer = (
                NullLogWriter()
                if skip_log
                else BZ2PickleLogWriter(log_root, self.log_dir)
            )
        self._log_writer: LogWriter = log_writer

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays.  The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time.  (i.e. it cannot act again while
        # it is still "in the future")
        #
        # Stored as numpy int64 arrays for fast O(1) hot-loop access.
        n = len(self.agents)
        self._agent_current_times: np.ndarray = np.full(
            n, self.start_time, dtype=np.int64
        )

        # agent_computation_delays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only).  It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg).  The
        # penalty applies _after_ the agent acts, before it may act again.
        if agent_computation_delays is None:
            self._agent_computation_delays: np.ndarray = np.full(
                n, default_computation_delay, dtype=np.int64
            )
        else:
            if agent_computation_delays.dtype != np.int64:
                raise ValueError(
                    "agent_computation_delays must have dtype int64, got "
                    f"{agent_computation_delays.dtype}"
                )
            if agent_computation_delays.shape != (n,):
                raise ValueError(
                    "agent_computation_delays must have shape (n_agents,), got "
                    f"{agent_computation_delays.shape} for n_agents={n}"
                )
            self._agent_computation_delays = agent_computation_delays.copy()

        # Type index for O(1) ``find_agents_by_type`` lookups. Walks each
        # agent's MRO so callers may query with a base class and still
        # get every subclass instance (preserves ``isinstance`` semantics).
        self._agents_by_type: dict[type, list[int]] = {}
        for agent in self.agents:
            for cls in type(agent).__mro__:
                self._agents_by_type.setdefault(cls, []).append(agent.id)

        # All message dispatch goes through a LatencyModel. When the caller
        # does not provide one explicitly we fall back to a uniform model
        # built from ``default_latency``; matrix-shaped models must be
        # constructed by the caller and passed via ``agent_latency_model=``.
        if agent_latency_model is None:
            agent_latency_model = UniformLatencyModel(default_latency)
        self.agent_latency_model: LatencyModel = agent_latency_model

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent.  This is applied to each message sent
        # and upon return from wakeup/receive_message, in addition to the
        # agent's standard computation delay.  However, it never carries
        # over to future wakeup/receive_message calls.  It is useful for
        # staggering of sent messages.
        self.current_agent_additional_delay: int = 0

        self.show_trace_messages: bool = False

        # Wall-clock anchor and message counter, set in initialize().
        self.event_queue_wall_clock_start: datetime | None = None
        self.ttl_messages: int = 0

        logger.debug("Kernel initialized")

    def run(self) -> KernelRunResult:
        """
        Wrapper to run the entire simulation (when not running in ABIDES-Gym mode).

        3 Steps:
          - Simulation Instantiation
          - Simulation Run
          - Simulation Termination

        Returns:
            :class:`KernelRunResult` with scheduling facts. Agents are
            still accessible via ``self.agents``; domain metrics are
            delivered to registered observers during ``terminate()``.
        """
        self.initialize()

        self.runner()

        return self.terminate()

    # This is called to actually start the simulation, once all agent
    # configuration is done.
    def initialize(self) -> None:
        """
        Instantiation of the simulation:
          - Creation of the different object of the simulation.
          - Instantiation of the latency network
          - Calls on the kernel_initializing and KernelStarting of the different agents
        """
        assert_transition(
            self._state,
            (KernelState.CONSTRUCTED, KernelState.TERMINATED),
            "initialize",
        )

        logger.debug("Kernel started")
        logger.debug("Simulation started!")

        # Reset per-run state so the kernel is safe to ``reset()`` and
        # re-``initialize()`` mid-process (gym training loops, parallel
        # workers in the same interpreter, etc.). Note that
        # ``agent_computation_delays`` is intentionally not cleared here:
        # it carries per-agent overrides from the constructor that must
        # persist across resets.
        self._agent_current_times[:] = self.start_time
        self.ttl_messages = 0
        self.summary_log.clear()
        self.messages.clear()
        self._next_seq = 0
        self.current_agent_additional_delay = 0

        # Note that num_simulations has not yet been really used or tested
        # for anything.  Instead we have been running multiple simulations
        # with coarse parallelization from a shell script

        # Event notification for kernel init (agents should not try to
        # communicate with other agents, as order is unknown).  Agents
        # should initialize any internal resources that may be needed
        # to communicate with other agents during agent.kernel_starting().
        # Kernel passes self-reference for agents to retain, so they can
        # communicate with the kernel in the future (as it does not have
        # an agentID).
        logger.debug("--- Agent.kernel_initializing() ---")
        for agent in self.agents:
            agent.kernel_initializing(self)

        # Event notification for kernel start (agents may set up
        # communications or references to other agents, as all agents
        # are guaranteed to exist now).  Agents should obtain references
        # to other agents they require for proper operation (exchanges,
        # brokers, subscription services...).  Note that we generally
        # don't (and shouldn't) permit agents to get direct references
        # to other agents (like the exchange) as they could then bypass
        # the Kernel, and therefore simulation "physics" to send messages
        # directly and instantly or to perform disallowed direct inspection
        # of the other agent's state.  Agents should instead obtain the
        # agent ID of other agents, and communicate with them only via
        # the Kernel.  Direct references to utility objects that are not
        # agents are acceptable (e.g. oracles).
        logger.debug("--- Agent.kernel_starting() ---")
        for agent in self.agents:
            agent.kernel_starting(self.start_time)

        # Set the kernel to its start_time.
        self.current_time = self.start_time

        logger.debug("--- Kernel Clock started ---")
        logger.debug(f"Kernel.current_time is now {fmt_ts(self.current_time)}")

        # Start processing the Event Queue.
        logger.debug("--- Kernel Event Queue begins ---")
        logger.debug(
            f"Kernel will start processing messages. Queue length: {len(self.messages)}"
        )

        # Track starting wall clock time and total message count for stats at the end.
        self.event_queue_wall_clock_start = datetime.now()
        self.ttl_messages = 0

        self._state = KernelState.INITIALIZED

    def runner(
        self, hook_actions: tuple[Agent, list[dict[str, Any]]] | None = None
    ) -> dict[str, Any]:
        """
        Start the simulation and processing of the message queue.

        Arguments:
            hook_actions: Optional ``(agent, actions)`` tuple. When
                provided, ``agent.apply_actions(actions)`` is invoked
                before the event loop resumes; this is how outer-loop
                hooks (e.g. ABIDES-gym) inject actions between
                kernel steps.

        Returns:
            Dictionary with two keys:
              - ``done``: ``True`` once the simulation has reached
                ``stop_time`` or the message queue is empty.
              - ``result``: the raw state returned by the registered
                :class:`RunnerHook`, or ``None`` when no hook is
                attached.
        """
        # Apply pending hook actions (e.g. a gym step) before resuming.
        if hook_actions is not None:
            exp_agent, action_list = hook_actions
            exp_agent.apply_actions(action_list)

        assert_transition(
            self._state,
            (KernelState.INITIALIZED, KernelState.RUNNING),
            "runner",
        )
        self._state = KernelState.RUNNING

        # Process messages until there aren't any (at which point there never can
        # be again, because agents only "wake" in response to messages), or until
        # the kernel stop time is reached.
        while (
            self.messages
            and self.current_time is not None
            and (self.current_time <= self.stop_time)
        ):
            # Get the next message in timestamp order (delivery time) and extract it.
            entry = heapq.heappop(self.messages)
            self.current_time = entry.deliver_at
            sender_id = entry.sender_id
            recipient_id = entry.recipient_id
            message = entry.message

            # Periodically print the simulation time and total messages, even if muted.
            if self.ttl_messages % 100000 == 0:
                logger.info(
                    "--- Simulation time: %s, messages processed: %s, wallclock elapsed: %.2fs ---",
                    fmt_ts(self.current_time),
                    f"{self.ttl_messages:,}",
                    (
                        datetime.now() - self.event_queue_wall_clock_start
                    ).total_seconds(),
                )

            if self.show_trace_messages:
                logger.debug("--- Kernel Event Queue pop ---")
                logger.debug(
                    "Kernel handling %s message for agent %d at time %s",
                    message.type(),
                    recipient_id,
                    self.current_time,
                )

            self.ttl_messages += 1

            # In between messages, always reset the current_agent_additional_delay.
            self.current_agent_additional_delay = 0

            # If the agent is busy in the future, requeue at its busy-until time.
            if self._agent_current_times[recipient_id] > self.current_time:
                self._enqueue(
                    int(self._agent_current_times[recipient_id]),
                    sender_id,
                    recipient_id,
                    message,
                )
                if self.show_trace_messages:
                    logger.debug(
                        "Agent in future: requeued for %s",
                        fmt_ts(int(self._agent_current_times[recipient_id])),
                    )
                continue

            # Set agent's current time to global current time for start of processing.
            self._agent_current_times[recipient_id] = self.current_time

            # Dispatch.
            wakeup_result: Any = None
            if message.__class__ is WakeupMsg:
                wakeup_result = self.agents[recipient_id].wakeup(self.current_time)
            elif message.__class__ is MessageBatch:
                for sub in message.messages:
                    self.agents[recipient_id].receive_message(
                        self.current_time, sender_id, sub
                    )
            else:
                self.agents[recipient_id].receive_message(
                    self.current_time, sender_id, message
                )

            # Advance AFTER delivery so any Agent.delay() call inside
            # wakeup() / receive_message() takes effect on the agent's
            # own next slot.
            self._agent_current_times[recipient_id] += (
                self._agent_computation_delays[recipient_id]
                + self.current_agent_additional_delay
            )

            if self.show_trace_messages:
                logger.debug(
                    "After dispatch, agent %d delayed from %s to %s",
                    recipient_id,
                    fmt_ts(self.current_time),
                    fmt_ts(int(self._agent_current_times[recipient_id])),
                )

            # Catch kernel interruption signal (runner hook's raw state).
            if wakeup_result is not None:
                return {"done": False, "result": wakeup_result}

        if not self.messages:
            logger.debug("--- Kernel Event Queue empty ---")

        if self.current_time and (self.current_time > self.stop_time):
            logger.debug("--- Kernel Stop Time surpassed ---")

        # if gets here means sim queue is fully processed, return to show sim is done
        if self._runner_hook is not None:
            self._runner_hook.update_raw_state()
            return {"done": True, "result": self._runner_hook.get_raw_state()}
        else:
            return {"done": True, "result": None}

    def terminate(self) -> KernelRunResult:
        """
        Termination of the simulation. Called once the queue is empty, or the gym environement is done, or the simulation
        reached kernel stop time:
          - Calls the kernel_stopping of the agents
          - Calls the kernel_terminating of the agents
          - Dispatches ``on_terminate`` to every registered observer

        Returns:
            :class:`KernelRunResult` — scheduling facts only. Domain
            metrics are delivered to observers; agents are accessible
            via ``self.agents``.
        """
        assert_transition(
            self._state,
            (KernelState.INITIALIZED, KernelState.RUNNING),
            "terminate",
        )
        event_queue_wall_clock_stop = datetime.now()

        event_queue_wall_clock_elapsed = (
            event_queue_wall_clock_stop - self.event_queue_wall_clock_start
        )

        # Event notification for kernel end (agents may communicate with
        # other agents, as all agents are still guaranteed to exist).
        # Agents should not destroy resources they may need to respond
        # to final communications from other agents.
        logger.debug("--- Agent.kernel_stopping() ---")
        for agent in self.agents:
            agent.kernel_stopping()

        # Event notification for kernel termination (agents should not
        # attempt communication with other agents, as order of termination
        # is unknown).  Agents should clean up all used resources as the
        # simulation program may not actually terminate if num_simulations > 1.
        logger.debug("\n--- Agent.kernel_terminating() ---")
        for agent in self.agents:
            agent.kernel_terminating()

        elapsed_seconds = event_queue_wall_clock_elapsed.total_seconds()
        logger.info(
            f"Event Queue elapsed: {event_queue_wall_clock_elapsed}, messages: {self.ttl_messages:,}, messages per second: {self.ttl_messages / elapsed_seconds if elapsed_seconds > 0 else 0:0.1f}"
        )

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernel_terminating, but the Kernel must write out the summary
        # log itself.
        self.write_summary_log()

        # Notify observers last so they see fully-terminated agent state.
        for observer in self._observers:
            observer.on_terminate(self)

        logger.info("Simulation ending!")

        self._state = KernelState.TERMINATED
        return KernelRunResult(
            elapsed=event_queue_wall_clock_elapsed,
            slowest_agent_finish_time=int(self._agent_current_times.max()),
            messages_processed=self.ttl_messages,
        )

    def reset(self) -> None:
        """
        Used in the gym core environment:
          - First calls termination of the kernel, to close previous simulation
          - Then initializes a new simulation
          - Then runs the simulation (not specifying any action this time).
        """

        if self._state in (KernelState.INITIALIZED, KernelState.RUNNING):
            self.terminate()

        self.initialize()
        self.runner()

    def send_message(
        self, sender_id: int, recipient_id: int, message: Message, delay: int = 0
    ) -> None:
        """
        Called by an agent to send a message to another agent.

        The kernel supplies its own current_time (i.e. "now") to prevent possible abuse
        by agents. The kernel will handle computational delay penalties and/or network
        latency.

        Arguments:
            sender_id: ID of the agent sending the message.
            recipient_id: ID of the agent receiving the message.
            message: The ``Message`` class instance to send.
            delay: Represents an agent's request for ADDITIONAL delay (beyond the
                Kernel's mandatory computation + latency delays). Represents parallel
                pipeline processing delays (that should delay the transmission of
                messages but do not make the agent "busy" and unable to respond to new
                messages)
        """

        # Apply the agent's current computation delay to effectively "send" the message
        # at the END of the agent's current computation period when it is done "thinking".
        # NOTE: sending multiple messages on a single wake will transmit all at the same
        # time, at the end of computation.  To avoid this, use Agent.delay() to accumulate
        # a temporary delay (current cycle only) that will also stagger messages.

        # The optional pipeline delay parameter DOES push the send time forward, since it
        # represents "thinking" time before the message would be sent.  We don't use this
        # for much yet, but it could be important later.

        # This means message delay (before latency) is the agent's standard computation
        # delay PLUS any accumulated delay for this wake cycle PLUS any one-time
        # requested delay for this specific message only.
        sent_time = self._compute_delay(sender_id, delay)

        # Apply communication delay via the (always-present) latency model.
        latency = self._latency(sender_id, recipient_id, message)
        deliver_at = sent_time + latency
        if self.show_trace_messages:
            logger.debug(
                f"Kernel applied latency {latency}, accumulated delay {self.current_agent_additional_delay}, one-time delay {delay} on send_message from: {self.agents[sender_id].name} to {self.agents[recipient_id].name}, scheduled for {fmt_ts(deliver_at)}"
            )

        # Finally drop the message in the queue with priority == delivery time.
        self._enqueue(deliver_at, sender_id, recipient_id, message)

        if self.show_trace_messages:
            logger.debug(
                f"Sent time: {sent_time}, current time {fmt_ts(self.current_time)}, computation delay {int(self._agent_computation_delays[sender_id])}"
            )
            logger.debug(f"Message queued: {message}")

    def _compute_delay(self, sender_id: int, one_time_delay: int = 0) -> int:
        """Return the kernel timestamp at which a message sent by
        ``sender_id`` can be considered transmitted, accounting for the
        agent's standing computation delay, any accumulated per-cycle
        delay, and the optional one-time pipeline delay.
        """
        return (
            self.current_time
            + int(self._agent_computation_delays[sender_id])
            + self.current_agent_additional_delay
            + one_time_delay
        )

    def _latency(
        self,
        sender_id: int,
        recipient_id: int,
        message: Message,
    ) -> int:
        """Network latency for a single message between ``sender_id``
        and ``recipient_id``. Routes through the configured
        :class:`LatencyModel` and forwards the message class so
        message-type-aware models can dispatch on it.
        """
        return self.agent_latency_model.get_latency(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_class=type(message),
            random_state=self.random_state,
        )

    def _enqueue(
        self,
        deliver_at: int,
        sender_id: int,
        recipient_id: int,
        message: Message,
    ) -> None:
        """Push a heap entry with a fresh per-kernel sequence number."""
        heapq.heappush(
            self.messages,
            _HeapEntry(
                deliver_at=deliver_at,
                seq=self._next_seq,
                sender_id=sender_id,
                recipient_id=recipient_id,
                message=message,
            ),
        )
        self._next_seq += 1

    def set_wakeup(
        self, sender_id: int, requested_time: NanosecondTime | None = None
    ) -> None:
        """
        Called by an agent to receive a "wakeup call" from the kernel at some requested
        future time.

        NOTE: The agent is responsible for maintaining any required state; the kernel
        will not supply any parameters to the ``wakeup()`` call.

        Arguments:
            sender_id: The ID of the agent making the call.
            requested_time: Defaults to the next possible timestamp.  Wakeup time cannot
            be the current time or a past time.
        """

        if requested_time is None:
            requested_time = self.current_time + 1

        if self.current_time and (requested_time < self.current_time):
            raise ValueError(
                f"set_wakeup() called with requested time not in future: "
                f"current_time: {self.current_time}, requested_time: {requested_time}"
            )

        if self.show_trace_messages:
            logger.debug(
                f"Kernel adding wakeup for agent {sender_id} at time {fmt_ts(requested_time)}"
            )

        self._enqueue(requested_time, sender_id, sender_id, _WAKEUP_SINGLETON)

    def set_agent_compute_delay(self, sender_id: int, requested_delay: int) -> None:
        """
        Called by an agent to update its computation delay.

        This does not initiate a global delay, nor an immediate delay for the agent.
        Rather it sets the new default delay for the calling agent. The delay will be
        applied upon every return from wakeup or recvMsg. Note that this delay IS
        applied to any messages sent by the agent during the current wake cycle
        (simulating the messages popping out at the end of its "thinking" time).

        Also note that we DO permit a computation delay of zero, but this should really
        only be used for special or massively parallel agents.

        Arguments:
            sender_id: The ID of the agent making the call.
            requested_delay: delay given in nanoseconds.
        """

        # requested_delay should be in whole nanoseconds.
        if not isinstance(requested_delay, int):
            raise ValueError(
                f"Requested computation delay must be whole nanoseconds. "
                f"requested_delay: {requested_delay}"
            )

        # requested_delay must be non-negative.
        if requested_delay < 0:
            raise ValueError(
                f"Requested computation delay must be non-negative nanoseconds. "
                f"requested_delay: {requested_delay}"
            )

        self._agent_computation_delays[sender_id] = requested_delay

    def get_agent_compute_delay(self, sender_id: int) -> int:
        """Return the current computation delay for the given agent.

        Arguments:
            sender_id: The ID of the agent to query.
        """
        return int(self._agent_computation_delays[sender_id])

    def delay_agent(self, sender_id: int, additional_delay: int) -> None:
        """
        Called by an agent to accumulate temporary delay for the current wake cycle.

        This will apply the total delay (at time of send_message) to each message, and
        will modify the agent's next available time slot.  These happen on top of the
        agent's compute delay BUT DO NOT ALTER IT. (i.e. effects are transient). Mostly
        useful for staggering outbound messages.

        Arguments:
            sender_id: The ID of the agent making the call.
            additional_delay: additional delay given in nanoseconds.
        """

        # additional_delay should be in whole nanoseconds.
        if not isinstance(additional_delay, int):
            raise ValueError(
                f"Additional delay must be whole nanoseconds. "
                f"additional_delay: {additional_delay}"
            )

        # additional_delay must be non-negative.
        if additional_delay < 0:
            raise ValueError(
                f"Additional delay must be non-negative nanoseconds. "
                f"additional_delay: {additional_delay}"
            )

        self.current_agent_additional_delay += additional_delay

    def find_agents_by_type(self, agent_type: type[Agent]) -> list[int]:
        """
        Returns the IDs of any agents that are of the given type.

        Lookup is O(1): the kernel pre-indexes every agent under each class
        in its MRO at construction time, so passing a base class still
        returns every subclass instance (preserves ``isinstance`` semantics).

        Arguments:
            agent_type: The agent type to search for.

        Returns:
            A list of agent IDs that are instances of the type.
        """
        return list(self._agents_by_type.get(agent_type, []))

    @property
    def state(self) -> KernelState:
        """Current lifecycle state of the kernel.

        ``KernelState`` is the single source of truth for run progress.
        Use ``kernel.state == KernelState.TERMINATED`` instead of any
        legacy ``has_run`` flag.
        """
        return self._state

    def write_log(
        self, sender_id: int, df_log: pd.DataFrame, filename: str | None = None
    ) -> None:
        """
        Called by any agent, usually at the very end of the simulation just before
        kernel shutdown, to write to disk any log dataframe it has been accumulating
        during simulation.

        The format can be decided by the agent, although changes will require a special
        tool to read and parse the logs.  The Kernel places the log in a unique
        directory per run, with one filename per agent, also decided by the Kernel using
        agent type, id, etc.

        If there are too many agents, placing all these files in a directory might be
        unfortunate. Also if there are too many agents, or if the logs are too large,
        memory could become an issue. In this case, we might have to take a speed hit to
        write logs incrementally.

        If filename is not None, it will be used as the filename. Otherwise, the Kernel
        will construct a filename based on the name of the Agent requesting log archival.

        Arguments:
            sender_id: The ID of the agent making the call.
            df_log: dataframe representation of the log that contains all the events logged during the simulation.
            filename: Location on disk to write the log to.
        """
        agent_name = self.agents[sender_id].name
        self._log_writer.write_agent_log(agent_name, df_log, filename)

    def append_summary_log(self, sender_id: int, event_type: str, event: Any) -> None:
        """
        We don't even include a timestamp, because this log is for one-time-only summary
        reporting, like starting cash, or ending cash.

        Arguments:
            sender_id: The ID of the agent making the call.
            event_type: The type of the event.
            event: The event to append to the log.
        """
        self.summary_log.append(
            {
                "AgentID": sender_id,
                "AgentStrategy": self.agents[sender_id].type,
                "EventType": event_type,
                "Event": event,
            }
        )

    def write_summary_log(self) -> None:
        df_log = pd.DataFrame(self.summary_log)
        self._log_writer.write_summary_log(df_log)
