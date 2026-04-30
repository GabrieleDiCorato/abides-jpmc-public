# Kernel Architecture

`abides_core.kernel.Kernel` is the discrete-event simulator at the heart
of ABIDES. This page describes its current public contract and the
collaborator types it relies on. It supersedes the kernel-related
sections in `llm-gotchas.md` for architecture-level questions; the
gotchas page remains the place to look for concrete pitfalls.

## Responsibilities

| Responsibility | Owner |
|----------------|-------|
| Event loop (heap pop / dispatch / requeue) | `Kernel.runner` |
| Per-agent next-available time tracking | `Kernel._agent_current_times` (numpy `int64`) |
| Per-agent computation delays | `Kernel._agent_computation_delays` (numpy `int64`) |
| Heap ordering | `_HeapEntry` (per-kernel `_next_seq` tiebreaker) |
| Network latency | `LatencyModel` injected at construction |
| Lifecycle state machine | `KernelState` (`abides_core.lifecycle`) |
| On-disk logging | `LogWriter` (`abides_core.log_writer`) |
| Gym integration | `GymAdapter` (`abides_core.gym_adapter`) |
| Per-agent-type metric aggregation | `Agent.report_metric` → `kernel.custom_state["agent_type_metrics"]` |
| Agent-by-type lookup | `Kernel._agents_by_type` (MRO-indexed) |

## Lifecycle state machine

```
CONSTRUCTED ──initialize()──▶ INITIALIZED ──runner()──▶ RUNNING
     ▲                            │                       │
     │                            └────────terminate()────┘
     │                                                    │
     └──────reset() (terminate + initialize)──────────────┘
                                                          │
                                                          ▼
                                                     TERMINATED
                                                          │
                                                  initialize()
                                                          │
                                                          ▼
                                                    INITIALIZED
```

Out-of-order public calls (e.g. `runner()` from `CONSTRUCTED`,
`initialize()` from `INITIALIZED`) raise `RuntimeError`. `reset()` is
safe from any state: it terminates the current run if one is in flight
and then re-initialises.

## Collaborator protocols

### `LogWriter`

```python
class LogWriter(Protocol):
    def write_agent_log(
        self, agent_name: str, df_log: pd.DataFrame, filename: str | None = None
    ) -> None: ...
    def write_summary_log(self, df_log: pd.DataFrame) -> None: ...
```

- Default: `NullLogWriter` if `skip_log=True`, otherwise
  `BZ2PickleLogWriter(log_root, log_dir)` (legacy
  `<root>/<run_id>/<name>.bz2` format, run directory materialised
  lazily on first write).
- Inject your own with `log_writer=...` for streaming, S3, parquet,
  etc. without touching the kernel.

### `GymAdapter`

```python
class GymAdapter(Protocol):
    def update_raw_state(self) -> None: ...
    def get_raw_state(self) -> Any: ...
    def apply_actions(self, actions: list[dict[str, Any]]) -> None: ...
```

- Pass `gym_adapter=<your_adapter>` to `Kernel(...)`.
- Legacy auto-detection (scan `agents` for any class with a
  `CoreGymAgent` base) still works for one release with a
  `DeprecationWarning`.
- The adapter is invoked when the runner reaches the end of the heap
  (`update_raw_state` + `get_raw_state`) and on `runner(agent_actions=…)`
  re-entry (`apply_actions`).

### `LatencyModel`

`LatencyModel` is the only pre-existing collaborator. The kernel
always routes `send_message` through `self.agent_latency_model`. If
the constructor receives the legacy `agent_latency` matrix or
`default_latency`, the kernel wraps them into `MatrixLatencyModel` /
`UniformLatencyModel` so the rest of the code path stays uniform.
`LatencyModel.get_latency` is called with a keyword-only
`random_state` reference; subclasses that ignore it (cubic latency,
which holds its own RNG) should still accept the kwarg.

## Per-agent state (numpy)

`_agent_current_times` and `_agent_computation_delays` are
`numpy.ndarray[int64]`. Internal kernel code uses these underscore
attributes. The legacy public names (`agent_current_times`,
`agent_computation_delays`) remain as **read-only deprecation
properties** that emit a one-shot `DeprecationWarning` per attribute
name and return a `flags.writeable=False` view.

`find_agents_by_type` is O(1): every agent is pre-indexed under each
class in its MRO at construction time, so callers may query with a
parent class and still get every subclass instance.

## Heap entries

The event queue stores `_HeapEntry(deliver_at, seq, sender_id,
recipient_id, message)` instances. Ordering is by `(deliver_at, seq)`
only; messages do not implement `__lt__` and are not compared. `seq`
is a per-kernel monotonic counter reset by `initialize()`, which
makes simulations reproducible across `reset()` calls and across
parallel workers in the same process.

The dispatcher is single-path: both `WakeupMsg` and ordinary messages
advance `_agent_current_times[recipient_id]` **after** delivery, so
`Agent.delay()` calls inside `wakeup()` or `receive_message()` take
effect on the agent's own next slot.

## Reserved attribute names

`Kernel.custom_properties` is a freeform dict but cannot shadow
kernel-managed attributes. The set is defined as
`_KERNEL_RESERVED_ATTRS` at the top of `kernel.py`. Submitting a
reserved key raises `ValueError` at construction time.

## Constructor surface

```python
Kernel(
    agents,                        # positional
    *,                             # everything below is keyword-only
    start_time=DEFAULT,
    stop_time=DEFAULT,
    default_computation_delay=1,
    default_latency=1,
    agent_latency=None,            # legacy; wrapped into MatrixLatencyModel
    latency_noise=None,            # legacy; forwarded to wrapped model
    agent_latency_model=None,      # explicit LatencyModel instance
    skip_log=True,
    seed=None,                     # ValueError-then-deprecation if no RNG provided
    log_dir=None,
    log_root="./log",              # PR 7
    log_writer=None,               # PR 7
    custom_properties=None,
    random_state=None,
    per_agent_computation_delays=None,
    gym_adapter=None,              # PR 7
)
```

The constructor validates the `agents[i].id == i` invariant, refuses
reserved `custom_properties` keys, and wraps all latency-model
parameters into a single `LatencyModel` instance. After
construction the kernel is in `KernelState.CONSTRUCTED`.

## See also

- [llm-gotchas.md](llm-gotchas.md) — concrete pitfalls and safe patterns
- [config-system.md](config-system.md) — declarative config builder
- [parallel-simulation.md](parallel-simulation.md) — multiprocessing & RNG
