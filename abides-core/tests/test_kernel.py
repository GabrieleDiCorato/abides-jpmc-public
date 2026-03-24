"""Tests for Kernel correctness fixes."""

import heapq

import numpy as np

from abides_core.agent import Agent
from abides_core.kernel import Kernel
from abides_core.message import Message, MessageBatch
from abides_core.utils import str_to_ns

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubAgent(Agent):
    """Minimal agent that records received messages."""

    def __init__(self, id: int) -> None:
        super().__init__(
            id=id,
            name=f"Stub_{id}",
            random_state=np.random.RandomState(seed=id),
            log_events=False,
        )
        self.received: list = []

    def wakeup(self, current_time):
        super().wakeup(current_time)

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        self.received.append(message)


# ---------------------------------------------------------------------------
# Step 1.1: Latency matrix aliasing
# ---------------------------------------------------------------------------


class TestLatencyMatrixAliasing:
    """Rows of the default latency matrix must be independent lists."""

    def test_rows_are_independent_objects(self):
        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(agents=agents, skip_log=True)
        # Each row must be a distinct list object
        assert kernel.agent_latency[0] is not kernel.agent_latency[1]
        assert kernel.agent_latency[1] is not kernel.agent_latency[2]

    def test_mutating_one_row_does_not_affect_others(self):
        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(agents=agents, default_latency=100, skip_log=True)
        # Mutate row 0
        kernel.agent_latency[0][0] = 999
        # Row 1 must be unchanged
        assert kernel.agent_latency[1][0] == 100
        assert kernel.agent_latency[2][0] == 100

    def test_default_values_correct(self):
        agents = [StubAgent(i) for i in range(4)]
        kernel = Kernel(agents=agents, default_latency=42, skip_log=True)
        for row in kernel.agent_latency:
            assert row == [42, 42, 42, 42]


# ---------------------------------------------------------------------------
# Step 1.2: MessageBatch delay
# ---------------------------------------------------------------------------


class TestMessageBatchDelay:
    """A MessageBatch delivery must apply computation delay ONCE, not N times."""

    def _make_kernel(self, comp_delay=100):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            default_computation_delay=comp_delay,
            skip_log=True,
        )
        kernel.initialize()
        # Drain any wakeup messages queued during initialize
        kernel.runner()
        return kernel, agents

    def test_single_message_delay(self):
        """Baseline: single message applies delay once."""
        comp_delay = 100
        kernel, agents = self._make_kernel(comp_delay)

        t = kernel.agent_current_times[1]
        # Enqueue a single message from agent 0 to agent 1
        deliver_at = t + 1  # just past agent's current time
        kernel.current_time = kernel.start_time  # reset so runner loop proceeds
        heapq.heappush(kernel.messages, (deliver_at, (0, 1, Message())))

        kernel.runner()

        # Agent 1 should have been delayed by exactly comp_delay from delivery time
        assert kernel.agent_current_times[1] == deliver_at + comp_delay

    def test_batch_delay_applied_once(self):
        """MessageBatch of N messages must apply delay only ONCE."""
        comp_delay = 100
        kernel, agents = self._make_kernel(comp_delay)

        t = kernel.agent_current_times[1]
        deliver_at = t + 1
        n_messages = 10
        batch = MessageBatch(messages=[Message() for _ in range(n_messages)])
        kernel.current_time = kernel.start_time
        heapq.heappush(kernel.messages, (deliver_at, (0, 1, batch)))

        kernel.runner()

        # Key assertion: delay is 1× comp_delay, NOT n_messages × comp_delay
        assert kernel.agent_current_times[1] == deliver_at + comp_delay
        # All messages in batch should have been delivered
        assert len(agents[1].received) == n_messages
