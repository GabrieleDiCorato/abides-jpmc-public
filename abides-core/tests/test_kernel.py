"""Tests for Kernel correctness fixes."""

import heapq

import numpy as np
import pandas as pd

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
# Latency model wiring
# ---------------------------------------------------------------------------


class TestLatencyModelWiring:
    """Kernel always builds a LatencyModel; legacy attrs are gone."""

    def test_default_wraps_into_uniform_latency_model(self):
        from abides_core.latency_model import UniformLatencyModel

        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(
            agents=agents,
            default_latency=42,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        assert isinstance(kernel.agent_latency_model, UniformLatencyModel)
        assert (
            kernel.agent_latency_model.get_latency(
                0, 1, random_state=kernel.random_state
            )
            == 42
        )

    def test_no_longer_exposes_agent_latency_attr(self):
        agents = [StubAgent(i) for i in range(2)]
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        assert not hasattr(kernel, "agent_latency")
        assert not hasattr(kernel, "latency_noise")


# ---------------------------------------------------------------------------
# MessageBatch delivery delay
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
            random_state=np.random.RandomState(seed=1),
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
# Hygiene fixes
# ---------------------------------------------------------------------------


class TestWriteSummaryLogRespectsSkipLog:
    def test_skip_log_creates_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        kernel.write_summary_log()
        assert not (tmp_path / "log").exists()


class TestTerminateZeroCountDoesNotCrash:
    def test_metric_with_zero_count_does_not_divide_by_zero(self):
        from abides_core.observers import DefaultMetricsObserver

        agents = [StubAgent(0)]
        observer = DefaultMetricsObserver()
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
            observers=[observer],
        )
        kernel.initialize()
        # Inject a metric with count==0 — must be skipped, not crash.
        observer._aggregates["Foo"] = {"ending_value": {"sum": 100.0, "count": 0}}
        kernel.terminate()


class TestAgentIdInvariantViolationRaises:
    def test_swapped_ids_raises(self):
        import pytest

        a = StubAgent(0)
        b = StubAgent(1)
        # Swap their ids so agents[i].id != i.
        a.id, b.id = 1, 0
        with pytest.raises(ValueError, match="agents\\[i\\]\\.id"):
            Kernel(
                agents=[a, b], skip_log=True, random_state=np.random.RandomState(seed=1)
            )


# ---------------------------------------------------------------------------
# State validation and reset hygiene
# ---------------------------------------------------------------------------


class TestOracleAndObserverSlots:
    def test_oracle_slot_defaults_to_none(self):
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        assert kernel.oracle is None

    def test_oracle_slot_accepts_instance(self):
        agents = [StubAgent(0)]
        sentinel = object()
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
            oracle=sentinel,
        )
        assert kernel.oracle is sentinel

    def test_observers_default_empty(self):
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        assert kernel._observers == ()


class TestInitializeClearsState:
    def test_initialize_clears_per_run_state(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        kernel.initialize()
        # Mutate per-run state.
        kernel.summary_log.append({"AgentID": 0})
        kernel._run_stats.ttl_messages = 99
        kernel.current_agent_additional_delay = 500
        # Stuff a stale message that would survive across a re-init if
        # initialize() did not clear ``messages``.
        kernel.messages.append((kernel.start_time, (0, 0, object())))
        stale = kernel.messages[-1]
        # Re-initialize and verify the slate is clean.
        # terminate() must precede a second initialize() per the lifecycle contract.
        kernel.terminate()
        kernel.initialize()
        assert kernel.summary_log == []
        assert kernel._run_stats.ttl_messages == 0
        assert kernel.current_agent_additional_delay == 0
        # The stale message must not survive; only fresh wakeups enqueued
        # by ``kernel_starting`` may now be present.
        assert stale not in kernel.messages
        assert all(t == kernel.start_time for t in kernel._agent_current_times)


class TestRandomStateRequired:
    def test_missing_random_state_raises_type_error(self):
        import pytest as _pytest

        agents = [StubAgent(0)]
        # ``random_state`` is a required keyword-only argument.
        with _pytest.raises(TypeError):
            Kernel(agents=agents, skip_log=True)


# ---------------------------------------------------------------------------
# Dispatch ordering and heap sequencing
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
            random_state=np.random.RandomState(seed=1),
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
            random_state=np.random.RandomState(seed=1),
        )
        kernel.initialize()
        baseline_first = kernel._next_seq
        for _ in range(5):
            kernel._enqueue(kernel.start_time + 1, 0, 1, Message())
        delta_first = kernel._next_seq - baseline_first

        # terminate() must precede a second initialize() per the lifecycle contract.
        kernel.terminate()
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
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
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
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
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
            random_state=np.random.RandomState(seed=1),
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
# Metric reporting
# ---------------------------------------------------------------------------


class TestReportMetricAggregation:
    def test_aggregates_by_type_and_key(self):
        from abides_core.observers import DefaultMetricsObserver

        agents = [StubAgent(0), StubAgent(1)]
        observer = DefaultMetricsObserver()
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
            observers=[observer],
        )
        kernel.initialize()

        # Patch type so both stub agents report under the same type.
        for a in agents:
            a.type = "StubAgent"

        agents[0].report_metric("ending_value", 100)
        agents[1].report_metric("ending_value", 300)
        agents[0].report_metric("trades", 5)

        metrics = observer.aggregates["StubAgent"]
        assert metrics["ending_value"] == {"sum": 400.0, "count": 2}
        assert metrics["trades"] == {"sum": 5.0, "count": 1}

    def test_legacy_attrs_are_gone(self):
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
        assert not hasattr(kernel, "mean_result_by_agent_type")
        assert not hasattr(kernel, "agent_count_by_type")

    def test_terminate_handles_missing_metrics(self):
        # No agent ever reports — terminate() must not crash.
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
        kernel.initialize()
        kernel.terminate()


# ---------------------------------------------------------------------------
# Concrete LatencyModel subclasses
# ---------------------------------------------------------------------------


class TestLatencyModelSubclasses:
    def test_uniform_does_not_touch_random_state(self):
        from abides_core.latency_model import UniformLatencyModel

        # UniformLatencyModel is purely deterministic; it must not
        # consume any RNG draws.
        rs = np.random.RandomState(seed=42)
        before = rs.get_state()
        model = UniformLatencyModel(latency=100)
        for _ in range(10):
            assert model.get_latency(0, 1, random_state=rs) == 100
        after = rs.get_state()
        assert before[0] == after[0]
        assert (before[1] == after[1]).all()
        assert before[2] == after[2]

    def test_uniform_returns_constant(self):
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

    def test_cubic_uses_passed_random_state(self):
        from abides_core.latency_model import CubicLatencyModel

        model = CubicLatencyModel(
            min_latency=np.array([[100, 200], [300, 400]], dtype=np.int64),
        )
        passed_rs = np.random.RandomState(seed=99)
        before = passed_rs.get_state()
        model.get_latency(0, 1, random_state=passed_rs)
        after = passed_rs.get_state()
        # Cubic must have consumed at least one draw.
        assert before[2] != after[2] or not (before[1] == after[1]).all()

    def test_cubic_requires_random_state(self):
        import pytest as _pytest

        from abides_core.latency_model import CubicLatencyModel

        model = CubicLatencyModel(min_latency=100)
        with _pytest.raises(ValueError, match="random_state"):
            model.get_latency(0, 1)

    def test_deterministic_returns_int(self):
        from abides_core.latency_model import DeterministicLatencyModel

        model = DeterministicLatencyModel(
            min_latency=np.array([[100, 200], [300, 400]], dtype=np.int64),
        )
        result = model.get_latency(0, 1)
        assert isinstance(result, int)
        assert result == 200

    def test_message_type_aware_routes_by_class(self):
        from abides_core.latency_model import (
            MessageTypeAwareLatencyModel,
            UniformLatencyModel,
        )
        from abides_core.message import Message, WakeupMsg

        class _OtherMsg(Message):
            pass

        default = UniformLatencyModel(latency=10)
        wakeup_model = UniformLatencyModel(latency=999)
        model = MessageTypeAwareLatencyModel(
            default=default, mapping={WakeupMsg: wakeup_model}
        )
        assert model.get_latency(0, 1, message_class=WakeupMsg) == 999
        assert model.get_latency(0, 1, message_class=_OtherMsg) == 10
        assert model.get_latency(0, 1, message_class=None) == 10

    def test_factory_constructs_each_subclass(self):
        from abides_core.latency_model import (
            CubicLatencyModel,
            DeterministicLatencyModel,
            LatencyModelFactory,
            MatrixLatencyModel,
            MessageTypeAwareLatencyModel,
            UniformLatencyModel,
        )

        assert isinstance(LatencyModelFactory.cubic(min_latency=10), CubicLatencyModel)
        assert isinstance(
            LatencyModelFactory.deterministic(min_latency=10),
            DeterministicLatencyModel,
        )
        assert isinstance(LatencyModelFactory.uniform(10), UniformLatencyModel)
        assert isinstance(
            LatencyModelFactory.matrix([[1, 2], [3, 4]]), MatrixLatencyModel
        )
        default = LatencyModelFactory.uniform(1)
        assert isinstance(
            LatencyModelFactory.by_message_type(default=default, mapping={}),
            MessageTypeAwareLatencyModel,
        )


# ---------------------------------------------------------------------------
# Per-agent numpy arrays and type index
# ---------------------------------------------------------------------------


class TestPerAgentStateNumpy:
    def test_per_agent_state_is_numpy_int64(self):
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
        assert isinstance(kernel._agent_current_times, np.ndarray)
        assert kernel._agent_current_times.dtype == np.int64
        assert kernel._agent_current_times.shape == (2,)
        assert isinstance(kernel._agent_computation_delays, np.ndarray)
        assert kernel._agent_computation_delays.dtype == np.int64
        assert kernel._agent_computation_delays.shape == (2,)

    def test_per_agent_computation_delays_overrides_applied(self):
        agents = [StubAgent(i) for i in range(4)]
        delays = np.array([10, 100, 10, 300], dtype=np.int64)
        kernel = Kernel(
            agents=agents,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
            default_computation_delay=10,
            agent_computation_delays=delays,
        )
        assert list(kernel._agent_computation_delays) == [10, 100, 10, 300]
        # Kernel must defensively copy.
        delays[1] = 999
        assert kernel._agent_computation_delays[1] == 100

    def test_agent_computation_delays_dtype_validation(self):
        import pytest

        agents = [StubAgent(0), StubAgent(1)]
        with pytest.raises(ValueError, match="dtype int64"):
            Kernel(
                agents=agents,
                skip_log=True,
                random_state=np.random.RandomState(seed=1),
                agent_computation_delays=np.array([0, 0], dtype=np.int32),
            )

    def test_agent_computation_delays_shape_validation(self):
        import pytest

        agents = [StubAgent(0), StubAgent(1)]
        with pytest.raises(ValueError, match="shape"):
            Kernel(
                agents=agents,
                skip_log=True,
                random_state=np.random.RandomState(seed=1),
                agent_computation_delays=np.array([0, 0, 0], dtype=np.int64),
            )

    def test_initialize_resets_current_times_in_place(self):
        # The slice-assign reset must mutate the same numpy buffer (no realloc),
        # so callers holding a reference to the array see the reset.
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
        buf_id = id(kernel._agent_current_times)
        kernel._agent_current_times[0] = 999_999
        kernel.initialize()
        assert id(kernel._agent_current_times) == buf_id
        assert (kernel._agent_current_times == kernel.start_time).all()


class _SpecialStub(StubAgent):
    pass


class TestFindAgentsByTypeIndex:
    def test_returns_subclass_matches(self):
        # Indexing walks the MRO so passing the parent class returns
        # both parent-typed and subclass-typed agents.
        agents = [StubAgent(0), _SpecialStub(1), StubAgent(2), _SpecialStub(3)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
        assert kernel.find_agents_by_type(_SpecialStub) == [1, 3]
        assert kernel.find_agents_by_type(StubAgent) == [0, 1, 2, 3]

    def test_unknown_type_returns_empty(self):
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )

        class _Other(Agent):
            pass

        assert kernel.find_agents_by_type(_Other) == []

    def test_returned_list_is_a_copy(self):
        # Caller mutation of the returned list must not corrupt the index.
        agents = [StubAgent(0), StubAgent(1)]
        kernel = Kernel(
            agents=agents, skip_log=True, random_state=np.random.RandomState(seed=1)
        )
        result = kernel.find_agents_by_type(StubAgent)
        result.append(999)
        assert kernel.find_agents_by_type(StubAgent) == [0, 1]


# ---------------------------------------------------------------------------
# LogWriter, KernelState lifecycle, RunnerHook, keyword-only __init__
# ---------------------------------------------------------------------------


class TestLogWriter:
    def test_null_log_writer_creates_no_files(self, tmp_path):
        from abides_core.log_writer import NullLogWriter

        writer = NullLogWriter()
        df = pd.DataFrame({"a": [1, 2, 3]})
        writer.write_agent_log("Foo", df)
        writer.write_summary_log(df)
        # tmp_path must remain empty.
        assert list(tmp_path.iterdir()) == []

    def test_bz2_log_writer_round_trip(self, tmp_path):
        from abides_core.log_writer import BZ2PickleLogWriter

        writer = BZ2PickleLogWriter(root=tmp_path, run_id="run42")
        df = pd.DataFrame({"a": [1, 2, 3]})
        writer.write_agent_log("Agent X", df)
        writer.write_summary_log(df)
        out = tmp_path / "run42"
        assert (out / "AgentX.bz2").exists()
        assert (out / "summary_log.bz2").exists()
        round_tripped = pd.read_pickle(out / "AgentX.bz2", compression="bz2")
        assert round_tripped.equals(df)

    def test_bz2_log_writer_creates_dir_lazily(self, tmp_path):
        from abides_core.log_writer import BZ2PickleLogWriter

        writer = BZ2PickleLogWriter(root=tmp_path, run_id="lazy")
        # Constructing must not create the run dir.
        assert not (tmp_path / "lazy").exists()
        writer.write_agent_log("Foo", pd.DataFrame({"a": [1]}))
        assert (tmp_path / "lazy").exists()

    def test_kernel_uses_injected_log_writer(self, tmp_path):
        import pandas as _pd

        class _RecordingWriter:
            def __init__(self):
                self.agent_calls: list[tuple[str, str | None]] = []
                self.summary_calls: int = 0

            def write_agent_log(self, name, df, filename=None):
                self.agent_calls.append((name, filename))

            def write_summary_log(self, df):
                self.summary_calls += 1

        writer = _RecordingWriter()
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents, random_state=np.random.RandomState(seed=1), log_writer=writer
        )
        kernel.write_log(0, _pd.DataFrame({"a": [1]}))
        kernel.write_summary_log()
        assert writer.agent_calls == [(agents[0].name, None)]
        assert writer.summary_calls == 1


class TestKernelLifecycle:
    def test_rejects_runner_before_initialize(self):
        import pytest as _pytest

        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, random_state=np.random.RandomState(seed=1))
        with _pytest.raises(RuntimeError, match="runner"):
            kernel.runner()

    def test_rejects_initialize_after_initialize(self):
        import pytest as _pytest

        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, random_state=np.random.RandomState(seed=1))
        kernel.initialize()
        with _pytest.raises(RuntimeError, match="initialize"):
            kernel.initialize()

    def test_rejects_terminate_before_initialize(self):
        import pytest as _pytest

        agents = [StubAgent(0)]
        kernel = Kernel(agents=agents, random_state=np.random.RandomState(seed=1))
        with _pytest.raises(RuntimeError, match="terminate"):
            kernel.terminate()

    def test_full_lifecycle_then_reset(self):
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            random_state=np.random.RandomState(seed=1),
        )
        kernel.initialize()
        kernel.runner()
        kernel.terminate()
        # reset() must succeed from TERMINATED.
        kernel.reset()


class _FakeRunnerHook:
    """Stand-in implementing the RunnerHook Protocol."""

    def __init__(self):
        self.actions: list = []
        self.update_calls = 0

    def update_raw_state(self):
        self.update_calls += 1

    def get_raw_state(self):
        return {"raw": True}

    def apply_actions(self, actions):
        self.actions.append(actions)


class TestRunnerHook:
    def test_explicit_registration(self):
        hook = _FakeRunnerHook()
        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents,
            random_state=np.random.RandomState(seed=1),
            runner_hook=hook,
        )
        assert kernel._runner_hook is hook


class TestKernelInitKeywordOnly:
    def test_positional_start_time_raises(self):
        import pytest as _pytest

        agents = [StubAgent(0)]
        with _pytest.raises(TypeError):
            # start_time passed positionally should fail with the
            # keyword-only signature.
            Kernel(agents, str_to_ns("09:30:00"))


# ---------------------------------------------------------------------------
# _UninitializedKernel sentinel and Agent.computation_delay property
# ---------------------------------------------------------------------------


class TestUninitializedKernelSentinel:
    """Agent.kernel must raise RuntimeError before kernel_initializing()."""

    def test_kernel_attr_access_before_attach_raises(self):
        import pytest as _pytest

        from abides_core.agent import _UNINITIALIZED_KERNEL

        with _pytest.raises(RuntimeError, match="Agent.kernel accessed before"):
            _ = _UNINITIALIZED_KERNEL.oracle

    def test_arbitrary_attr_access_raises(self):
        import pytest as _pytest

        from abides_core.agent import _UNINITIALIZED_KERNEL

        with _pytest.raises(RuntimeError, match="attribute='nonexistent'"):
            _ = _UNINITIALIZED_KERNEL.nonexistent

    def test_agent_kernel_default_is_sentinel(self):
        from abides_core.agent import _UninitializedKernel

        agent = StubAgent(0)
        assert isinstance(agent.kernel, _UninitializedKernel)

    def test_agent_kernel_attached_after_initialize(self):
        from abides_core.agent import _UninitializedKernel

        agents = [StubAgent(0)]
        kernel = Kernel(
            agents=agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        # Before initialize: still the sentinel.
        assert isinstance(agents[0].kernel, _UninitializedKernel)
        kernel.initialize()
        # After initialize: real kernel object.
        assert agents[0].kernel is kernel


class TestAgentComputationDelayProperty:
    """Agent.computation_delay property reads from and writes to the kernel array."""

    def _make_kernel(self, *extra_agents):
        all_agents = [StubAgent(0), *extra_agents]
        for i, a in enumerate(all_agents):
            a.id = i
        kernel = Kernel(
            agents=all_agents,
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            default_computation_delay=50,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        kernel.initialize()
        return kernel, all_agents

    def test_property_reads_default(self):
        kernel, (agent,) = self._make_kernel()
        assert agent.computation_delay == 50

    def test_property_setter_updates_kernel_array(self):
        kernel, (agent,) = self._make_kernel()
        agent.computation_delay = 999
        assert kernel._agent_computation_delays[0] == 999

    def test_getter_reflects_setter(self):
        kernel, (agent,) = self._make_kernel()
        agent.computation_delay = 777
        assert agent.computation_delay == 777

    def test_per_agent_overrides_visible_through_property(self):
        a0, a1, a2 = StubAgent(0), StubAgent(1), StubAgent(2)
        delays = np.array([10, 200, 10], dtype=np.int64)
        kernel = Kernel(
            agents=[a0, a1, a2],
            start_time=str_to_ns("09:30:00"),
            stop_time=str_to_ns("16:00:00"),
            agent_computation_delays=delays,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        kernel.initialize()
        assert a0.computation_delay == 10
        assert a1.computation_delay == 200
        assert a2.computation_delay == 10


# ---------------------------------------------------------------------------
# Kernel get/set agent compute delay
# ---------------------------------------------------------------------------


class TestKernelGetComputeDelay:
    """Kernel.get_agent_compute_delay / set_agent_compute_delay round-trip."""

    def test_get_returns_default(self):
        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=42,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        for i in range(3):
            assert kernel.get_agent_compute_delay(i) == 42

    def test_get_after_set(self):
        agents = [StubAgent(i) for i in range(3)]
        kernel = Kernel(
            agents=agents,
            default_computation_delay=50,
            skip_log=True,
            random_state=np.random.RandomState(seed=1),
        )
        kernel.set_agent_compute_delay(1, 999)
        assert kernel.get_agent_compute_delay(1) == 999
        # Neighbours unchanged.
        assert kernel.get_agent_compute_delay(0) == 50
        assert kernel.get_agent_compute_delay(2) == 50
