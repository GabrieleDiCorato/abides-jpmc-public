# Event Logging — Refactoring & Improvement Plan

**Status:** Plan (not yet executed). Decisions encoded; implementation pending.
**Scope:** All three logging subsystems in `abides-core`, `abides-markets`,
`abides-gym` — primarily System B (per-agent `Agent.logEvent`) and System C
(`Kernel.summary_log`). System A (stdlib `logging`) is touched only at the
edges.
**Companion analysis:** [`docs/reference/logging-architecture.md`](../reference/logging-architecture.md).
**Audience:** Maintainers and contributors implementing the refactor.

---

## 0. Context and framing

ABIDES is a discrete-event market simulation. The per-agent event log
(`Agent.logEvent` → `agent.log` → `BZ2PickleLogWriter`) is functionally a
**drop-copy stream**: agents emit business events, a sink persists them
out-of-band of trading logic, downstream tooling reconstructs state from
the stream. Treating it as a drop-copy clarifies the design: producers
should not know about the sink, sinks should be pluggable, and the on-disk
format is one implementation among many.

Three real problems today:

1. **Memory is unbounded.** `agent.log` grows linearly for the whole
   run; nothing flushes mid-simulation. The same applies to
   `OrderBook.book_log2` and `OrderBook.history`, which the exchange
   accumulates in-process and the runner reads at terminate. Hard
   ceiling for long sims and for `abides-gym` training loops.
2. **End-of-run cost is high.** `to_pickle(compression="bz2")` is the
   slowest pickle path in pandas; `parse_logs_df` is **O(N²)** because it
   does `pd.concat([pd.DataFrame([row]) for row in ...])`.
3. **Three subsystems are conflated.** Operator stdout (System A), the
   load-bearing per-agent record (System B), and the vestigial
   `summary_log` (System C) share one folder, one config flag
   (`skip_log`), and no documentation about who reads what.

The kernel improvement work has already carved part of the seam:

- `LogWriter` Protocol (`abides_core/log_writer.py`) abstracts the
  *terminal write* (`NullLogWriter`, `BZ2PickleLogWriter`).
- `KernelObserver` Protocol (`abides_core/observers.py`) plus
  `Agent.report_metric()` carries numeric KPIs out-of-band of the event
  stream.

What is missing is the **producer-side** seam — the equivalent of a
publish/subscribe bus between `Agent.logEvent` and the sink — and the
backpressure/threading needed to bound memory.

---

## 1. Goals (in priority order)

1. **Bound memory** during long simulations (today `agent.log` grows
   without limit).
2. **Decouple producers from sinks** so format and destination are
   pluggable without touching agent code.
3. **Cut serialization cost and `parse_logs_df` cost** at simulation
   end.
4. **Preserve bit-for-bit reproducibility** of the default on-disk
   artifact for the deprecation window. After removal of the legacy
   path, only the new sinks need to be reproducible.
5. **Stop conflating three subsystems** in code, configuration, and
   on-disk layout.

---

## 2. Non-negotiable invariants

Any implementation of this plan must satisfy all of the following. CI
checks must enforce them where possible.

- **Determinism.** Sinks must observe events in publication order. Async
  drainers must produce identical on-disk output to the inline drainer
  for any fixed seed. A regression test pins this with byte equality on
  a small reference run.
- **No silent event drops.** Backpressure is **blocking by default**
  (the producer thread waits when the queue is full). Optional
  fail-fast mode is a config flag, never the default. Dropping events
  on overflow is not a supported mode.
- **Hot-path producer cost when no sink wants an event type is one
  dict lookup plus an early return** (analogous to
  `logger.isEnabledFor(DEBUG)` guards).
- **No new mandatory third-party dependencies.** `pyarrow` (for
  `ParquetSink`) stays optional and is import-guarded.
- **Backward-compatible default for the deprecation window.** Until
  Phase 5 lands, a default-config run produces a byte-identical
  `<run_id>/<AgentName>.bz2` artifact compared with the current
  release, validated by an end-to-end test.
- **Reproducibility contracts in `docs/project/reproducibility.md`
  remain intact.**

---

## 3. Architecture

### 3.1 The event bus

A single internal `EventBus` owned by the kernel. One per kernel
instance; not a process global.

Responsibilities:

- Receive `EventRecord` and `MetricRecord` value objects from agents.
- Maintain the **union of subscribed event types** across all
  registered sinks. Compute it once at simulation start; expose a fast
  `is_event_type_wanted(event_type) -> bool` lookup so producers can
  short-circuit before allocating any payload-related state.
- Dispatch to sinks either inline (default for tests / `skip_log=True`)
  or via a single bounded-queue drainer thread (opt-in for production
  runs).
- Call sink lifecycle hooks at simulation start, on `flush()` calls,
  and at simulation end.

The bus owns one **bounded `queue.SimpleQueue`** (or `queue.Queue` for
the bounded variant) shared by all sinks in the threaded mode. Records
are dispatched in publication order regardless of OS scheduling — this
is what guarantees reproducibility across drain modes.

### 3.2 Record types

Four concrete value objects flow through the bus. All are immutable
dataclasses with `slots=True` to keep producer cost low. The bus
treats them uniformly; only the *kind* differs (used by the filter
machinery in §3.3).

- `EventRecord(agent_id, agent_type, sim_time_ns, event_type, payload, seq)`
  — replaces the current `(time, event_type, event)` tuple stored on
  `agent.log`. `seq` is a monotonically increasing per-bus counter
  used to break ties when several records share `sim_time_ns` and to
  make ordering checks trivial in tests.
- `MetricRecord(agent_id, agent_type, key, value, seq)` — replaces
  the direct `KernelObserver.on_metric` fan-out. `report_metric`
  becomes a publish on the same bus; the existing
  `DefaultMetricsObserver` is adapted into a sink (see §3.4).
- `BookSnapshotRecord(symbol, sim_time_ns, bids, asks, depth, seq)`
  — replaces direct mutation of `OrderBook.book_log2`. `bids` and
  `asks` are the same list-of-(price, qty) shape that `book_log2`
  carries today, so existing readers keep working unchanged. Emitted
  by `OrderBook.append_book_log2()` after each top-of-book mutation.
- `OrderBookEventRecord(symbol, sim_time_ns, kind, order_id, price, qty, side, payload, seq)`
  — replaces `OrderBook.history` appends (`LIMIT`, `EXEC`, `CANCEL`,
  `MODIFY`, `REPLACE`, etc.). `kind` is the same string today's
  `entry["type"]` carries; `payload` holds the type-specific extras
  that don't fit the common columns.

`EventRecord.payload` and `OrderBookEventRecord.payload` stay `Any`
for backward compatibility, but new sinks (Parquet) MAY require a
typed schema per `event_type` / `kind`; the bus itself does not
enforce typing.

### 3.3 The `EventSink` Protocol and generalized filtering

A sink declares **(a)** which *record kinds* it wants and **(b)** for
each kind, an optional fine-grained selector. The bus computes the
union per kind once at simulation start and exposes a fast
`is_*_wanted(...)` lookup so producers can short-circuit before
allocating any payload-related state.

```text
RecordKind = Literal["event", "metric", "book_snapshot", "orderbook_event"]

# Selector is a Container (allowlist) OR a callable predicate, OR None
# (= accept all of this kind), OR False (= reject all of this kind).
EventSelector       = Container[str] | Callable[[EventRecord], bool] | None | Literal[False]
BookSymbolSelector  = Container[str] | Callable[[BookSnapshotRecord], bool] | None | Literal[False]
OBEventSelector     = Container[str] | Callable[[OrderBookEventRecord], bool] | None | Literal[False]
MetricSelector      = Container[str] | Callable[[MetricRecord], bool] | None | Literal[False]

class EventSink(Protocol):
    # Per-kind selectors; default-False means "this sink ignores that kind".
    wants_events:           EventSelector       = False
    wants_metrics:          MetricSelector      = False
    wants_book_snapshots:   BookSymbolSelector  = False  # filters on symbol
    wants_orderbook_events: OBEventSelector     = False  # filters on kind

    def on_simulation_start(self, meta: SimulationMeta) -> None: ...
    def on_event(self, record: EventRecord) -> None: ...
    def on_metric(self, record: MetricRecord) -> None: ...
    def on_book_snapshot(self, record: BookSnapshotRecord) -> None: ...
    def on_orderbook_event(self, record: OrderBookEventRecord) -> None: ...
    def flush(self) -> None: ...
    def on_simulation_end(self, meta: SimulationMeta) -> None: ...
```

Filter resolution rules (uniform across all four kinds):

- `False` (default) — sink does not want this kind. The bus does not
  even register it for that kind; producers see this in the union and
  short-circuit if no other sink wants it either.
- `None` — accept every record of this kind.
- `Container[str]` (set, frozenset, list, tuple) — accept iff the
  record's discriminator is in the container. Discriminator is
  `event_type` for events, `key` for metrics, `symbol` for book
  snapshots, `kind` for order-book events.
- `Callable[[Record], bool]` — arbitrary predicate. Evaluated on the
  drainer thread (or producer thread in inline mode). Predicates that
  raise are treated as sink failures (§3.6).

The bus precomputes, for each kind, three things: (i) `is_kind_wanted`
— "any sink at all wants this kind", (ii) per-kind union of
allowlists (used by Container-style selectors as a single membership
check), (iii) the list of sinks that need per-record evaluation
(callable selectors). The hot-path producer cost when nothing wants a
record is **one dict lookup plus an early return**, matching
`logger.isEnabledFor(DEBUG)`.

All hooks are synchronous from the sink's point of view. In threaded
mode, all hooks run on the drainer thread — sinks must not assume the
caller is the kernel/agent thread.

Sinks that want their own batching (Parquet row groups, DB
transactions) buffer internally and flush on `flush()` /
`on_simulation_end()`.

### 3.4 Concrete sinks shipped in this repository

- **`InMemorySink`** — Reproduces today's `agent.log` shape. Default
  registration. `wants_events=None`, everything else `False`.
  Required for `parse_logs_df` and for any in-process consumer
  (notebooks, metrics pipeline) that reads `SimulationResult.logs`.
- **`BZ2PickleSink`** — Wraps the existing `BZ2PickleLogWriter`.
  Drains `InMemorySink` at `on_simulation_end()` and writes the
  legacy `<run_id>/<AgentName>.bz2` artifact. **Marked deprecated
  from Phase 5; removed two minor releases later.**
- **`ParquetSink`** (the recommended new sink) — Columnar; one
  parquet file per `event_type` (and per book-record kind), struct-
  of-arrays schema. Snappy or Zstd compression. Writes one row group
  per `flush()`. Solves the memory ceiling, serialization speed, and
  untyped-payload friction simultaneously. Uses `pyarrow`,
  import-guarded; absence yields a clear error at sink construction,
  not at first write.
- **`MetricsObserverSink`** — Adapter that exposes the existing
  `KernelObserver` Protocol on top of the bus. The shipped
  `DefaultMetricsObserver` continues to work unchanged, registered
  as a metrics-only sink (`wants_metrics=None`). Removes the dual
  dispatch path inside `Agent`.
- **`OrderBookSnapshotMemorySink`** — In-memory capture of
  `BookSnapshotRecord`s, indexed by symbol. `wants_book_snapshots`
  defaults to `None` (all symbols) but accepts a symbol allowlist or
  predicate so a user can request "only ABM, only depth ≤ 5". This
  is the **"book-only fast access" sink**: register it standalone
  (without `InMemorySink` / `BZ2PickleSink`) when the only thing the
  caller needs after the run is the L1/L2 history. Today's
  `runner._extract_l1_series` / `_extract_l2_series` consume this
  sink instead of reading `OrderBook.book_log2` directly.
- **`OrderBookHistoryMemorySink`** — In-memory capture of
  `OrderBookEventRecord`s, indexed by symbol. `wants_orderbook_events`
  defaults to `None` but is typically narrowed (e.g.
  `frozenset({"EXEC"})`) to capture only fills. Today's
  `runner._extract_trades` and the VWAP path in `_extract_liquidity`
  consume this sink instead of walking `OrderBook.history`.

Out-of-tree sinks (JSONL, SQLite, DB drop-copy, message brokers,
custom book aggregators) are documented as a supported extension
point; the library does not ship them.

### 3.5 Async drainer

- One drainer thread per kernel, started lazily on the first published
  record. Reused across episodes within the same process so
  `abides-gym` does not pay startup cost per episode.
- Bounded queue size from config (`event_buffer_size`, default e.g.
  16384).
- Overflow behaviour: **block producer** (default) or **fail-fast**
  (raises `EventBusFull`). Configurable; never silent-drop.
- `inline` mode: no thread, no queue, dispatch on the calling thread.
  Used by default in tests and when `skip_log=True`. Required for
  determinism tests.
- The drainer publishes records in FIFO order; sinks see records in the
  exact order producers emitted them.
- On simulation end, the kernel calls `bus.shutdown()` which drains the
  queue, calls `on_simulation_end()` on each sink, then joins the
  thread. Errors raised by sinks during shutdown propagate to the
  kernel terminate path (see §3.6).

### 3.6 Error handling

- Sink exceptions during `on_event`/`on_metric` in **inline mode**
  propagate to the producer (current behaviour).
- Sink exceptions in **threaded mode** are captured on a per-sink error
  channel; the bus marks the sink as failed, stops dispatching to it,
  and surfaces the exception at `bus.shutdown()` so the kernel can
  decide whether to fail the run.
- Default kernel policy on sink failure: **fail-fast at terminate**
  with a clear log line indicating which sink failed and how many
  records it had buffered. Configurable to "best-effort warn" for
  research workflows where partial logs are better than no logs.
- All terminal disk writes use **temp-file-then-rename** to avoid
  partial files on disk-full / permission-denied. Today a crash at
  terminate after a multi-hour sim leaves no artifact at all; this
  closes that hole.

---

## 4. Concrete changes by module

This is a high-level map, not a final code review.

- `abides_core/agent.py`
  - `logEvent` becomes `bus.publish_event(EventRecord(...))` after the
    `is_event_type_wanted` check.
  - `report_metric` becomes `bus.publish_metric(MetricRecord(...))`.
  - `append_summary_log` parameter on `logEvent` is removed; the
    "important events" set becomes a sink-side allowlist (see §5).
  - `self.log` is removed from `Agent` itself and lives on
    `InMemorySink`. A compatibility shim exposes the historical
    attribute path through `SimulationResult` for one release.
- `abides_core/kernel.py`
  - Kernel owns an `EventBus`. Sinks are constructed from
    `SimulationConfig` (see §6) at `Kernel.__init__` or injected
    directly for tests.
  - `Kernel.append_summary_log` and `Kernel.write_summary_log` are
    removed in Phase 5; until then they delegate to a
    `LegacySummarySink` (see §5).
  - Terminate sequence: stop simulation → `bus.shutdown()` → existing
    `_log_writer` calls (deprecated branch) → observer
    `on_terminate()`.
- `abides_core/log_writer.py`
  - Marked deprecated in Phase 5. Kept until removal so out-of-tree
    code that wraps `BZ2PickleLogWriter` keeps working through the
    deprecation window.
- `abides_core/observers.py`
  - Unchanged externally. Internally re-implemented as a sink adapter.
- `abides_core/event_bus.py` (new) — `EventBus`, drainer thread,
  bounded queue, per-kind filter union.
- `abides_core/event_sinks.py` (new) — `EventSink` Protocol,
  `InMemorySink`, `BZ2PickleSink`, `MetricsObserverSink`.
- `abides_core/parquet_sink.py` (new, optional import) — `ParquetSink`.
- `abides_markets/abides_markets/order_book.py`
  - Each existing `self.book_log2.append(...)` site
    (`order_book.py:391, 461, 539, 587, 639, 682`) becomes a
    `bus.publish_book_snapshot(...)` after the
    `is_book_snapshot_wanted(symbol)` check. The cost when no sink
    wants snapshots collapses to one dict lookup, which is what
    today's `if self.owner.book_logging:` branch wishes it could be.
  - Each `self.history.append(...)` site (LIMIT/EXEC/CANCEL/MODIFY/
    REPLACE) becomes a `bus.publish_orderbook_event(...)` after the
    same kind of guard.
  - `book_log2` and `history` become deprecated read-through
    properties backed by the auto-registered
    `OrderBookSnapshotMemorySink` / `OrderBookHistoryMemorySink` for
    one release, then are removed alongside the legacy pickle path
    in Phase 5+2. A `DeprecationWarning` fires on first access.
- `abides_markets/abides_markets/agents/exchange_agent.py`
  - `book_logging` and `book_log_depth` constructor args become hints
    that translate to auto-registration of the two book memory sinks
    at config-build time (preserves backward-compatible defaults).
    Once `event_sinks` is set explicitly in config, the hints are
    ignored with a one-line info log.
- `abides_markets/abides_markets/simulation/runner.py`
  - `parse_logs_df` rewritten to a single `pd.DataFrame.from_records`
    call (Phase 1 — independent of the rest).
  - `SimulationResult.logs` is sourced from `InMemorySink` instead of
    walking `end_state["agents"]`.
  - `_extract_l1_close`, `_extract_l1_series`, `_extract_l2_series`
    read from `OrderBookSnapshotMemorySink` (still the same
    `book_log2`-shaped row dicts so `compute_l1_*` / `compute_l2_*`
    are unchanged).
  - `_extract_trades` and the VWAP path in `_extract_liquidity` read
    from `OrderBookHistoryMemorySink`.
- `abides_markets/abides_markets/config_system/models.py`
  - New fields described in §6.

---

## 5. Resolution of System C (`summary_log`)

**Decision: deprecate then remove.** (User-confirmed: "it is ok to
deprecate and later remove the old logging/pickle stuff.")

Rationale: every value System C provides is recoverable from System B
plus `report_metric`. The original consumer (a JPMorgan-internal
cross-run aggregator) was never open-sourced, no internal code reads
`summary_log.bz2`, and no documented external workflow depends on it.

Migration path:

1. **Phase 5 (deprecation release).** Introduce `LegacySummarySink`.
   `Kernel.append_summary_log` keeps working but emits a single
   `DeprecationWarning` per run pointing at this document. The
   `summary_log.bz2` artifact is still written so external tooling
   that may exist out of tree keeps functioning.
2. **Phase 5 + 1 (one minor release later).** Stop writing
   `summary_log.bz2` by default. Emit `DeprecationWarning` from
   `Kernel.append_summary_log` even when called.
3. **Phase 5 + 2.** Remove `Kernel.append_summary_log`,
   `Kernel.write_summary_log`, `Kernel.summary_log`,
   `LegacySummarySink`, the `append_summary_log=True` parameter on
   `Agent.logEvent`, and `LogWriter.write_summary_log` from the
   Protocol.

Users who genuinely need a cross-run roll-up of "starting cash, ending
cash, final valuation" can register `MetricsObserverSink` + a small
custom sink, or compute it from the per-agent logs. Both paths are
documented before Phase 5 ships.

---

## 6. Configuration surface

Promoted to `SimulationConfig.simulation`:

- `event_sinks: list[SinkConfig]` — declarative sink registry.
  Examples:
  - `[{kind: "memory"}]` — today's behaviour for agent events.
  - `[{kind: "memory"}, {kind: "orderbook_snapshot_memory"},
    {kind: "orderbook_history_memory"}, {kind: "bz2_pickle", root: "./log"}]`
    — the backward-compatible default during the deprecation window
    (book sinks auto-registered to keep `runner._extract_*` working).
  - `[{kind: "orderbook_snapshot_memory", symbols: ["ABM"]}]` — the
    **"book-only fast access"** setup: nothing else is captured, no
    `agent.log` is built, no pickle is written, and producers
    short-circuit every event/metric publish. The post-sim caller
    reads `SimulationResult.l2_snapshots["ABM"]` and that's it.
  - `[{kind: "parquet", root: "./log", events: ["ORDER_*"],
    book_snapshots: {symbols: ["ABM"]}, compression: "zstd"}]` —
    production analytics setup. The selector keys (`events`,
    `metrics`, `book_snapshots`, `orderbook_events`) map 1:1 to the
    Protocol fields in §3.3.
- `event_buffer_size: int` — bounded queue size for the threaded
  drainer (default 16384).
- `event_drain_mode: "inline" | "thread"` — default `inline` in tests
  and `skip_log=True`, `thread` in `run_simulation` defaults.
- `event_overflow_policy: "block" | "fail_fast"` — default `block`.
- `log_root: str` — currently buried in `Kernel(log_root=...)` only.
  Promoted so configs can drive it without touching the kernel
  constructor.
- `show_trace_messages: bool` — currently undiscoverable. Promoted to
  config.

Per-agent-type convenience (today requires loop-and-mutate at build
time):

- `disable_event_log_for: list[str]` — agent type names whose
  `log_events=False` is forced at build time. Replaces the only
  documented use of per-instance flags.

The existing `Agent.log_events` and `Agent.log_to_file` per-instance
flags remain for advanced cases, but the documented path becomes the
declarative one.

---

## 7. Phased rollout

Each phase is independently mergeable. No breaking change before Phase
5. Each phase ships with its own tests and changelog entry.

### Phase 1 — Quick wins (no architecture)

- Replace `parse_logs_df`'s row-by-row `pd.concat` with
  `pd.DataFrame.from_records`. This is the largest single performance
  win available and is independent of the rest of the plan.
- Add `try/except` + temp-file-then-rename around all terminal
  `LogWriter` calls.
- Honour `skip_log` in `Kernel.write_summary_log` (already filed as
  bug A.1 in the kernel improvement plan).
- No public API change.

### Phase 2 — Introduce the bus and sink Protocol

- New modules: `event_bus.py`, `event_sinks.py`.
- Refactor `Agent.logEvent` and `Agent.report_metric` to publish on
  the bus.
- Default sink set: `InMemorySink` + `BZ2PickleSink` +
  `MetricsObserverSink` — chosen so externally observable behaviour is
  unchanged.
- Inline drain mode only.
- Reproducibility test: byte-identical `<run_id>/<AgentName>.bz2`
  artifact for a fixed seed against the previous release.

### Phase 3 — Ship `ParquetSink` and event-type allowlist

- Optional `pyarrow` import.
- Add the generalized per-kind selector plumbing through the bus
  (`Container | Callable | None | False` per §3.3).
- Add Parquet schema-per-event-type design doc; ship one schema per
  event type emitted by core agents.
- Decision needed before merge: per-event-type files vs single file
  with `payload_json` column (see §9).

### Phase 3a — Move order-book capture onto the bus

Independently mergeable from the rest of Phase 3 (only depends on
Phase 2's bus + the §3.3 generalized filter).

- Add `BookSnapshotRecord` and `OrderBookEventRecord` to the bus.
- Replace each `OrderBook.book_log2.append(...)` and
  `OrderBook.history.append(...)` site with the corresponding
  `bus.publish_*` call, guarded by `is_*_wanted(...)`.
- Ship `OrderBookSnapshotMemorySink` and
  `OrderBookHistoryMemorySink`. Auto-register them when
  `ExchangeAgent.book_logging=True` and the user has not provided
  an explicit `event_sinks` list — preserves the default `runner`
  output byte-for-byte.
- Rewire `runner._extract_l1_close`, `_extract_l1_series`,
  `_extract_l2_series`, `_extract_trades`, and the VWAP path in
  `_extract_liquidity` to read from the two book sinks.
- Turn `OrderBook.book_log2` and `OrderBook.history` into
  deprecated read-through properties; add a
  `DeprecationWarning`-on-first-access test.
- Reproducibility test: `SimulationResult.l1_snapshots`,
  `.l2_snapshots`, `.trades`, and `.liquidity` byte-identical to
  the previous release for a fixed seed.

### Phase 4 — Threaded drainer with bounded queue

- Lazy thread, reused across episodes in the same process.
- `event_drain_mode` and `event_buffer_size` config wiring.
- Reproducibility test extended to assert byte-identical artifact
  between `inline` and `thread` modes for a fixed seed across all
  shipped sinks.

### Phase 5 — Deprecate `summary_log` and the legacy pickle path

- `BZ2PickleSink` and `Kernel.append_summary_log` emit
  `DeprecationWarning` once per run.
- Migration documentation links to this plan and to
  `MetricsObserverSink` / `ParquetSink` examples.
- Default `event_sinks` flips to `[{kind: "memory"},
  {kind: "parquet"}]` in a follow-up minor release.

### Phase 5 + 2 — Remove

- Delete `BZ2PickleLogWriter`, `BZ2PickleSink`,
  `Kernel.append_summary_log`, `Kernel.write_summary_log`,
  `Kernel.summary_log`, the `append_summary_log=True` parameter on
  `logEvent`, `LegacySummarySink`, and
  `LogWriter.write_summary_log` from the Protocol.
- Update `parallel-simulation.md` on-disk layout section.
- Major version bump.

---

## 8. Test plan

A non-exhaustive list of tests required to land each phase. Existing
tests in `abides-markets/tests/test_pandas_integration.py`,
`test_simulation.py`, `test_replace_order_regression.py`,
`test_config_system.py`, and `abides-core/tests/test_kernel.py` are
the baseline; nothing in this plan is allowed to remove or weaken
their assertions.

- **Round-trip equivalence (Phase 2 onward).** A small fixed-seed sim
  is run under the legacy code path and the new bus path; the produced
  `<run_id>/<AgentName>.bz2` files must be byte-identical.
- **Drain-mode equivalence (Phase 4).** Same sim under `inline` vs
  `thread`; all sink outputs must be byte-identical.
- **Backpressure correctness (Phase 4).** With a deliberately tiny
  `event_buffer_size`, no events may be lost; producer must block
  rather than drop.
- **Filter short-circuit (Phase 3 / 3a).** Agents emitting an event
  type — or an order book mutating — when no sink wants the
  resulting record must not allocate the payload (verified via a
  sentinel callable in the test that raises if invoked). Covers all
  four record kinds.
- **Book-only sink isolation (Phase 3a).** A run configured with
  *only* `OrderBookSnapshotMemorySink` produces correct
  `SimulationResult.l2_snapshots`, builds no `agent.log`, writes no
  pickle, and (asserted via patched `Agent.logEvent`) does not even
  call into the event publish path.
- **Book sink equivalence (Phase 3a).** A small fixed-seed sim
  produces byte-identical `SimulationResult.l1_snapshots`,
  `.l2_snapshots`, `.trades`, and `.liquidity` between the legacy
  `book_log2`/`history` path and the new sink-backed path.
- **Selector forms (Phase 3).** A sink with a `Container` selector,
  a sink with a `Callable` selector, and a sink with `None` produce
  the records expected for each — and a `False` (default) selector
  on every sink for a kind makes `is_kind_wanted` return `False`.
- **Sink failure isolation (Phase 2 onward).** A sink that raises in
  `on_event` does not crash the simulation when the configured policy
  is "best-effort warn"; it does crash at terminate when the policy is
  fail-fast.
- **Parquet schema stability (Phase 3).** Round-trip a recorded Parquet
  artifact through `pyarrow` and assert the schema matches the pinned
  reference for each shipped event type.
- **Determinism over multiple episodes (`abides-gym`).** Running N
  episodes back-to-back with the threaded drainer must produce the
  same final per-episode artifacts as N independent runs with the
  inline drainer.
- **Memory bound (Phase 4).** A long-running synthetic sim with a
  small buffer keeps RSS under a configured ceiling — sanity smoke
  test, not a hard assertion.

---

## 9. Open questions — to resolve before the relevant phase

These are intentionally unresolved in this plan. Each must be answered
in the implementation PR for the phase noted.

1. **Parquet layout (Phase 3).** Per-`event_type` parquet files (fast,
   small, typed; more files per run) vs one file with a `payload_json`
   string column (simpler, slower). Recommendation: per-type, but
   confirm with a benchmark on a representative sim.
2. **`pyarrow` weight (Phase 3).** `pyarrow` is ~50MB. Already a
   transitive dep of recent `pandas`, so likely free; confirm by
   inspecting the resolved environment in CI.
3. **Replay path (Phase 3).** Should `parse_logs_df` learn to read
   Parquet sink artifacts, or do we ship a separate
   `read_parquet_logs` helper? Recommendation: separate helper,
   `parse_logs_df` stays the canonical reader for the in-memory path
   only.
4. **Sink failure policy default (Phase 2).** Fail-fast (safe, may
   surprise research workflows) vs best-effort warn (lenient, may hide
   bugs). Recommendation: fail-fast, with a clearly documented config
   override.
5. **Multiprocess parallel runs.** Each worker owns its own bus and
   sinks; no cross-process queueing. Confirm this matches the planned
   cross-run aggregation story before Phase 5.
6. **Deprecation timeline.** This plan says Phase 5 + 2 for removal.
   Confirm the absolute timeline in `docs/project/release-process.md`
   when Phase 5 is scheduled.

---

## 10. What this plan does *not* change

- The stdlib `logging` subsystem (System A). It works. The only
  related improvement is making `show_trace_messages` discoverable via
  config (§6).
- The `KernelObserver` Protocol's external surface. It becomes a sink
  adapter internally; users registering observers see no change.
- The semantics of `Agent.report_metric` from the caller's point of
  view. It still pushes a numeric `(key, value)` to all interested
  observers.
- The `SimulationResult.logs` shape returned by `run_simulation`. It
  remains a `DataFrame` produced by `parse_logs_df` (now over the
  `InMemorySink` contents).
- The `SimulationResult.l1_snapshots` / `.l2_snapshots` / `.trades`
  / `.liquidity` shape. They remain identical; only the source of
  truth moves from `OrderBook.book_log2` / `OrderBook.history` to
  the two book sinks.
- Reproducibility guarantees. The drain-mode-equivalence test pins
  this.
