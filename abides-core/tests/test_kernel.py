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
# Step 1.1: Latency model wiring (PR 5a)
# ---------------------------------------------------------------------------


class TestLatencyModelWiring:
    """Kernel always builds a LatencyModel; legacy attrs are gone."""

    def test_default_wraps_into_uniform_latency_model(self):
        from abides_core.latency_model import UniformLatencyModel

        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(agents=agents, default_latency=42, skip_log=True, seed=1)
        assert isinstance(kernel.agent_latency_model, UniformLatencyModel)
        assert (
            kernel.agent_latency_model.get_latency(
                0, 1, random_state=kernel.random_state
            )
            == 42
        )

    def test_legacy_agent_latency_param_wraps_into_matrix_model(self):
        from abides_core.latency_model import MatrixLatencyModel

        agents = [StubAgent(i) for i in range(3)]
        matrix = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        kernel = Kernel(agents=agents, agent_latency=matrix, skip_log=True, seed=1)
        assert isinstance(kernel.agent_latency_model, MatrixLatencyModel)
        assert (
            kernel.agent_latency_model.get_latency(
                1, 2, random_state=kernel.random_state
            )
            == 60
        )

    def test_no_longer_exposes_agent_latency_attr(self):
        agents = [StubAgent(i) for i in range(2)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        assert not hasattr(kernel, "agent_latency")
        assert not hasattr(kernel, "latency_noise")


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

        t = int(kernel._agent_current_times[1])
        # Enqueue a single message from agent 0 to agent 1
        deliver_at = t + 1  # just past agent's current time
        kernel.current_time = kernel.start_time  # reset so runner loop proceeds
        kernel._enqueue(deliver_at, 0, 1, Message())

        kernel.runner()

        # Agent 1 should have been delayed by exactly comp_delay from delivery time
        assert kernel._agent_current_times[1] == deliver_at + comp_delay

    def test_batch_delay_applied_once(self):
        """MessageBatch of N messages must apply delay only ONCE."""
        comp_delay = 100
        kernel, agents = self._make_kernel(comp_delay)

        t = int(kernel._agent_current_times[1])
        deliver_at = t + 1
        n_messages = 10
        batch = MessageBatch(messages=[Message() for _ in range(n_messages)])
        kernel.current_time = kernel.start_time
        kernel._enqueue(deliver_at, 0, 1, batch)

        kernel.runner()

        # Key assertion: delay is 1× comp_delay, NOT n_messages × comp_delay
        assert kernel._agent_current_times[1] == deliver_at + comp_delay
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
    def test_metric_with_zero_count_does_not_divide_by_zero(self):
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel.initialize()
        # Inject a metric with count==0 — must be skipped, not crash.
        kernel.custom_state["agent_type_metrics"] = {
            "Foo": {"ending_value": {"sum": 100.0, "count": 0}}
        }
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
        assert all(t == kernel.start_time for t in kernel._agent_current_times)


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


# ---------------------------------------------------------------------------
# PR 3: Dispatch ordering bug + heap refactor
# ---------------------------------------------------------------------------


class _DelayingAgent(StubAgent):
    """Agent that calls self.delay(N) inside its message handler."""

    def __init__(self, id: int, extra_delay: int) -> None:
        super().__init__(id)
        self.extra_delay = extra_delay

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        self.delay(self.extra_delay)


class TestDispatchOrderingBug:
    def test_delay_in_receive_message_advances_agent_time(self):
        comp_delay = 100
        extra = 500
        agents = [StubAgent(0), _DelayingAgent(1, extra)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            default_computation_delay=comp_delay,
            skip_log=True,
            seed=1,
        )
        kernel.initialize()
        kernel.runner()  # drain initial wakeups

        deliver_at = int(kernel._agent_current_times[1]) + 1
        kernel.current_time = kernel.start_time
        kernel._enqueue(deliver_at, 0, 1, Message())
        kernel.runner()

        # Without the bug fix, this would equal deliver_at + comp_delay.
        # The dispatch ordering bug fix means delay(500) inside
        # receive_message takes effect on the agent's own next slot.
        assert kernel._agent_current_times[1] == deliver_at + comp_delay + extra


class TestHeapSequenceCounter:
    def test_sequence_resets_across_initialize(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            skip_log=True,
            seed=1,
        )
        kernel.initialize()
        baseline_first = kernel._next_seq
        for _ in range(5):
            kernel._enqueue(kernel.start_time + 1, 0, 1, Message())
        delta_first = kernel._next_seq - baseline_first

        kernel.initialize()
        baseline_second = kernel._next_seq
        for _ in range(5):
            kernel._enqueue(kernel.start_time + 1, 0, 1, Message())
        delta_second = kernel._next_seq - baseline_second

        # initialize() resets the counter; only the 5 enqueues below it count.
        assert delta_first == delta_second == 5
        assert baseline_first == baseline_second

    def test_heap_entry_ordering_does_not_touch_message(self):
        # Two entries with the same deliver_at: ordering must use seq,
        # never message.__lt__ (which no longer exists).
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel._enqueue(100, 0, 0, Message())
        kernel._enqueue(100, 0, 0, Message())
        # Pop both; must not raise even though Message has no __lt__.
        e1 = heapq.heappop(kernel.messages)
        e2 = heapq.heappop(kernel.messages)
        assert e1.seq < e2.seq


class TestWakeupSingleton:
    def test_set_wakeup_reuses_singleton(self):
        from abides_core.kernel import _WAKEUP_SINGLETON

        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel.current_time = 0
        kernel.set_wakeup(0, requested_time=10)
        kernel.set_wakeup(0, requested_time=20)
        # Both pending entries should reuse the singleton.
        wakeup_entries = [e for e in kernel.messages if e.message is _WAKEUP_SINGLETON]
        assert len(wakeup_entries) == 2


class TestMessageBatchDispatch:
    def test_batch_delivered_as_individual_messages(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            skip_log=True,
            seed=1,
        )
        kernel.initialize()
        kernel.runner()

        msgs = [Message() for _ in range(3)]
        batch = MessageBatch(messages=msgs)
        kernel.current_time = kernel.start_time
        kernel._enqueue(int(kernel._agent_current_times[1]) + 1, 0, 1, batch)
        kernel.runner()

        # All three sub-messages delivered, batch wrapper not.
        assert agents[1].received == msgs


# ---------------------------------------------------------------------------
# PR 4: report_metric() and removal of financial fields from kernel core
# ---------------------------------------------------------------------------


class TestReportMetricAggregation:
    def test_aggregates_by_type_and_key(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel.initialize()

        # Patch type so both stub agents report under the same type.
        for a in agents:
            a.type = "StubAgent"

        agents[0].report_metric("ending_value", 100)
        agents[1].report_metric("ending_value", 300)
        agents[0].report_metric("trades", 5)

        metrics = kernel.custom_state["agent_type_metrics"]["StubAgent"]
        assert metrics["ending_value"] == {"sum": 400.0, "count": 2}
        assert metrics["trades"] == {"sum": 5.0, "count": 1}

    def test_legacy_attrs_are_gone(self):
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        assert not hasattr(kernel, "mean_result_by_agent_type")
        assert not hasattr(kernel, "agent_count_by_type")

    def test_terminate_handles_missing_metrics(self):
        # No agent ever reports — terminate() must not crash.
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        kernel.initialize()
        kernel.terminate()


# ---------------------------------------------------------------------------
# PR 5a: Concrete LatencyModel subclasses
# ---------------------------------------------------------------------------


class TestLatencyModelSubclasses:
    def test_uniform_default_does_not_touch_random_state(self):
        from abides_core.latency_model import UniformLatencyModel

        # PR 5b: with the default noise=None, get_latency must NOT
        # consume any RNG draws. Pass a fresh RNG and assert its state
        # is unchanged.
        rs = np.random.RandomState(seed=42)
        before = rs.get_state()
        model = UniformLatencyModel(latency=100)
        for _ in range(10):
            assert model.get_latency(0, 1, random_state=rs) == 100
        after = rs.get_state()
        assert before[0] == after[0]
        assert (before[1] == after[1]).all()
        assert before[2] == after[2]

    def test_uniform_explicit_noise_consumes_one_draw(self):
        from abides_core.latency_model import UniformLatencyModel

        # Explicit noise=[1.0] restores the legacy pre-PR-5b behavior.
        rs1 = np.random.RandomState(seed=42)
        rs2 = np.random.RandomState(seed=42)
        model = UniformLatencyModel(latency=100, noise=[1.0])
        for _ in range(10):
            model.get_latency(0, 1, random_state=rs1)
            rs2.choice(1, p=[1.0])
        assert rs1.tomaxint() == rs2.tomaxint()

    def test_uniform_returns_constant_with_default_noise(self):
        from abides_core.latency_model import UniformLatencyModel

        rs = np.random.RandomState(seed=0)
        model = UniformLatencyModel(latency=250)
        assert model.get_latency(0, 1, random_state=rs) == 250
        assert model.get_latency(2, 3, random_state=rs) == 250

    def test_uniform_casts_float_latency_to_int(self):
        from abides_core.latency_model import UniformLatencyModel

        rs = np.random.RandomState(seed=0)
        model = UniformLatencyModel(latency=99.7)
        # int() truncates toward zero.
        assert model.get_latency(0, 1, random_state=rs) == 99

    def test_matrix_returns_pairwise_value(self):
        from abides_core.latency_model import MatrixLatencyModel

        rs = np.random.RandomState(seed=0)
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        model = MatrixLatencyModel(m)
        assert model.get_latency(0, 2, random_state=rs) == 3
        assert model.get_latency(2, 1, random_state=rs) == 8

    def test_cubic_ignores_random_state_kwarg(self):
        from abides_core.latency_model import LatencyModel

        own_rs = np.random.RandomState(seed=1)
        model = LatencyModel(
            random_state=own_rs,
            min_latency=np.array([[100, 200], [300, 400]], dtype=np.int64),
            latency_model="cubic",
        )
        # Pass a fresh, unused RandomState; cubic must NOT touch it.
        passed_rs = np.random.RandomState(seed=99)
        before = passed_rs.get_state()
        model.get_latency(0, 1, random_state=passed_rs)
        after = passed_rs.get_state()
        # State tuple equality: same algorithm name, same key array, same pos.
        assert before[0] == after[0]
        assert (before[1] == after[1]).all()
        assert before[2] == after[2]

    def test_deterministic_returns_int(self):
        from abides_core.latency_model import LatencyModel

        rs = np.random.RandomState(seed=0)
        model = LatencyModel(
            random_state=rs,
            min_latency=np.array([[100, 200], [300, 400]], dtype=np.int64),
            latency_model="deterministic",
        )
        result = model.get_latency(0, 1, random_state=rs)
        assert isinstance(result, int)
        assert result == 200


# ---------------------------------------------------------------------------
# PR 6: Per-agent state to numpy arrays + agent type index
# ---------------------------------------------------------------------------


class TestPerAgentStateNumpy:
    def test_per_agent_state_is_numpy_int64(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        assert isinstance(kernel._agent_current_times, np.ndarray)
        assert kernel._agent_current_times.dtype == np.int64
        assert kernel._agent_current_times.shape == (2,)
        assert isinstance(kernel._agent_computation_delays, np.ndarray)
        assert kernel._agent_computation_delays.dtype == np.int64
        assert kernel._agent_computation_delays.shape == (2,)

    def test_per_agent_computation_delays_overrides_applied(self):
        agents = [StubAgent(i) for i in range(4)]
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            seed=1,
            default_computation_delay=10,
            per_agent_computation_delays={1: 100, 3: 300},
        )
        assert list(kernel._agent_computation_delays) == [10, 100, 10, 300]

    def test_initialize_resets_current_times_in_place(self):
        # The slice-assign reset must mutate the same numpy buffer (no realloc),
        # so callers holding a reference to the array see the reset.
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        buf_id = id(kernel._agent_current_times)
        kernel._agent_current_times[0] = 999_999
        kernel.initialize()
        assert id(kernel._agent_current_times) == buf_id
        assert (kernel._agent_current_times == kernel.start_time).all()


class TestLegacyPerAgentAttrShims:
    def test_legacy_attr_warns_and_is_readonly(self):
        import warnings as _w

        # Reset the process-level warned set so the warning fires here even
        # if another test has already hit this attribute.
        from abides_core.kernel import _WARNED_DEPRECATED_ATTRS

        _WARNED_DEPRECATED_ATTRS.discard("agent_current_times")

        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always", DeprecationWarning)
            view = kernel.agent_current_times
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "agent_current_times" in str(w.message)
            for w in captured
        )
        # Returned view must be read-only.
        import pytest as _pytest

        with _pytest.raises(ValueError):
            view[0] = 0

    def test_legacy_attr_warns_only_once(self):
        import warnings as _w

        from abides_core.kernel import _WARNED_DEPRECATED_ATTRS

        _WARNED_DEPRECATED_ATTRS.discard("agent_computation_delays")

        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        with _w.catch_warnings(record=True) as captured:
            _w.simplefilter("always", DeprecationWarning)
            _ = kernel.agent_computation_delays
            _ = kernel.agent_computation_delays
            _ = kernel.agent_computation_delays
        relevant = [w for w in captured if "agent_computation_delays" in str(w.message)]
        assert len(relevant) == 1


class _SpecialStub(StubAgent):
    pass


class TestFindAgentsByTypeIndex:
    def test_returns_subclass_matches(self):
        # Indexing walks the MRO so passing the parent class returns
        # both parent-typed and subclass-typed agents.
        agents = [StubAgent(0), _SpecialStub(1), StubAgent(2), _SpecialStub(3)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        assert kernel.find_agents_by_type(_SpecialStub) == [1, 3]
        assert kernel.find_agents_by_type(StubAgent) == [0, 1, 2, 3]

    def test_unknown_type_returns_empty(self):
        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)

        class _Other(Agent):
            pass

        assert kernel.find_agents_by_type(_Other) == []

    def test_returned_list_is_a_copy(self):
        # Caller mutation of the returned list must not corrupt the index.
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(agents=agents, skip_log=True, seed=1)
        result = kernel.find_agents_by_type(StubAgent)
        result.append(999)
        assert kernel.find_agents_by_type(StubAgent) == [0, 1]
