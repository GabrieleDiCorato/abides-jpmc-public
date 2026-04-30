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
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        # Each row must be a distinct list object
        assert kernel.agent_latency[0] is not kernel.agent_latency[1]
        assert kernel.agent_latency[1] is not kernel.agent_latency[2]

    def test_mutating_one_row_does_not_affect_others(self):
        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(agents=agents, default_latency=100, skip_log=True, seed=1)
        # Mutate row 0
        kernel.agent_latency[0][0] = 999
        # Row 1 must be unchanged
        assert kernel.agent_latency[1][0] == 100
        assert kernel.agent_latency[2][0] == 100

    def test_default_values_correct(self):
        agents = [StubAgent(i) for i in range(4)]
        kernel = Kernel(agents=agents, default_latency=42, skip_log=True, seed=1)
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
            seed=1,
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


# ---------------------------------------------------------------------------
# PR 1: Hygiene fixes
# ---------------------------------------------------------------------------


class _CoreGymAgentBase(Agent):
    """Stand-in for the abides-gym CoreGymAgent base class.

    The kernel detects gym agents by class name lookup, not by import,
    so the simulated base just needs ``__name__ == 'CoreGymAgent'``.
    """


# Force the detection key without importing abides-gym.
_CoreGymAgentBase.__name__ = "CoreGymAgent"


class _FakeGymAgent(_CoreGymAgentBase):
    def __init__(self, id: int) -> None:
        super().__init__(
            id=id,
            name=f"Gym_{id}",
            random_state=np.random.RandomState(seed=id),
            log_events=False,
        )


class TestWriteSummaryLogRespectsSkipLog:
    def test_skip_log_creates_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel.write_summary_log()
        assert not (tmp_path / "log").exists()


class TestTooManyGymAgentsRaisesValueError:
    def test_two_gym_agents_raises(self):
        import pytest

        agents = [_FakeGymAgent(0), _FakeGymAgent(1)]
        with pytest.raises(ValueError, match="one gym agent"):
            Kernel(agents=agents, skip_log=True, seed=1)


class TestTerminateZeroCountDoesNotCrash:
    def test_empty_mean_block_does_not_divide_by_zero(self):
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel.initialize()
        # Inject a value with no count — formerly would ZeroDivisionError.
        kernel.mean_result_by_agent_type["Foo"] = 100
        # No count key for "Foo" -> should be skipped, not crash.
        kernel.terminate()


class TestAgentIdInvariantViolationRaises:
    def test_swapped_ids_raises(self):
        import pytest

        a = StubAgent(0)
        b = StubAgent(1)
        # Swap their ids so agents[i].id != i.
        a.id, b.id = 1, 0
        with pytest.raises(ValueError, match="agents\\[i\\]\\.id"):
            Kernel(agents=[a, b], skip_log=True, seed=1)


# ---------------------------------------------------------------------------
# PR 2: State validation & reset hygiene
# ---------------------------------------------------------------------------


class TestCustomPropertiesReservedKey:
    def test_reserved_key_raises(self):
        import pytest

        agents = [StubAgent(0)]
        with pytest.raises(ValueError, match="reserved kernel attribute"):
            Kernel(
                agents=agents,
                skip_log=True,
                seed=1,
                custom_properties={"agents": [1, 2, 3]},
            )

    def test_user_keys_allowed(self):
        agents = [StubAgent(0)]
        sentinel = object()
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            seed=1,
            custom_properties={"oracle": sentinel, "my_thing": 42},
        )
        assert kernel.oracle is sentinel
        assert kernel.my_thing == 42


class TestInitializeClearsState:
    def test_initialize_clears_per_run_state(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            skip_log=True,
            seed=1,
        )
        kernel.initialize()
        # Mutate per-run state.
        kernel.custom_state["x"] = 1
        kernel.summary_log.append({"AgentID": 0})
        kernel.ttl_messages = 99
        kernel.current_agent_additional_delay = 500
        # Stuff a stale message that would survive across a re-init if
        # initialize() did not clear ``messages``.
        kernel.messages.append((kernel.start_time, (0, 0, object())))
        stale = kernel.messages[-1]
        # Re-initialize and verify the slate is clean.
        kernel.initialize()
        assert kernel.custom_state == {}
        assert kernel.summary_log == []
        assert kernel.ttl_messages == 0
        assert kernel.current_agent_additional_delay == 0
        # The stale message must not survive; only fresh wakeups enqueued
        # by ``kernel_starting`` may now be present.
        assert stale not in kernel.messages
        assert all(t == kernel.start_time for t in kernel.agent_current_times)


class TestRandomStateDeprecationWarning:
    def test_warns_when_neither_seed_nor_random_state(self):
        import warnings as _w

        agents = [StubAgent(0)]
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always")
            Kernel(agents=agents, skip_log=True)
        msgs = [str(w.message) for w in captured]
        assert any("explicit seed" in m for m in msgs), msgs

    def test_no_warning_with_seed(self):
        import warnings as _w

        agents = [StubAgent(0)]
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            Kernel(agents=agents, skip_log=True, seed=42)

    def test_no_warning_with_random_state(self):
        import warnings as _w

        agents = [StubAgent(0)]
        rs = np.random.RandomState(7)
        with _w.catch_warnings():
            _w.simplefilter("error", DeprecationWarning)
            Kernel(agents=agents, skip_log=True, random_state=rs)
