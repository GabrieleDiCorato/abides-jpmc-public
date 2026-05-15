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
**drop-copy stream**: agents emit business events, an analytics layer
persists or aggregates them out-of-band of trading logic, downstream
tooling reconstructs state from the stream. Treating it as a drop-copy
clarifies the design: producers should not know about the analytics
layer, the analytics layer should be pluggable, and the on-disk format is
one implementation among many.

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

What is missing is the **producer-side** seam between agents/order book
and the analytics layer. This plan adds that seam in a way that is
**lighter than the current implementation on the hot path** — not just a
clean abstraction layered on top.

---

## 1. Goals (in priority order)

1. **Lower per-event hot-path cost** versus the current
   `if owner.book_logging: book_log2.append({...})` and
   `agent.log.append((time, type, payload))` paths. The new path must
   be measurably cheaper when the analytics layer is enabled, and
   essentially free (a single C-level call into a no-op) when it is
   not.
2. **Bound memory** during long simulations.
3. **Fully decouple the kernel from the analytics layer.** The kernel
   knows it has a *channel*; it does not know what consumes it. Sinks
   are pluggable without touching agent or order-book code.
4. **Cut serialization cost** at simulation end and replace
   `parse_logs_df`'s O(N²) `pd.concat` loop.
5. **Preserve bit-for-bit reproducibility** of the default on-disk
   artifact for the deprecation window. After removal of the legacy
   path, only the new sinks need to be reproducible.
6. **Stop conflating three subsystems** in code, configuration, and
   on-disk layout.

---

## 2. Non-negotiable invariants

Any implementation must satisfy all of the following. CI checks must
enforce them where possible.

- **Determinism.** Sinks observe events in publication order. Single-
  threaded dispatch by default (see §3.1) makes this true by
  construction. Optional async I/O sinks (e.g. Parquet) must still
  produce identical on-disk output to the inline path for any fixed
  seed; pinned by a byte-equality regression test on a small reference
  run.
- **No silent event drops.** Backpressure that requires bounding (only
  possible at all on async-I/O sinks; see §3.5) **spills to disk**
  rather than blocking or dropping. The producer is never blocked by
  an analytics buffer.
- **Hot-path producer cost when no sink wants a record kind is one
  C-level call into a pre-bound no-op.** Not a flag check — a literal
  `lambda *a, **k: None` rebound onto the bus at start time.
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

### 3.1 Single-threaded, tick-batched dispatch

A single internal `EventBus` owned by the kernel. One per kernel
instance; not a process global. **The bus does not own a thread, queue,
or lock.**

How it dispatches:

1. `bus.publish_<kind>(...)` appends a record to a tiny per-kind ring
   buffer (a plain Python `list` reused across ticks). No locking, no
   GIL handoff, no allocation beyond the record itself.
2. The kernel's main loop calls `bus.drain()` at well-defined points:
   between event handlers, or every N events (configurable; default 1
   = "after each handler"). `drain()` walks the ring buffers and calls
   the appropriate hook on each registered sink, in publication order.
3. Sinks that need to do real I/O (Parquet flush, network) **own their
   own writer thread internally**, consuming from their own private
   buffer at flush time. The bus knows nothing about that thread; it
   just hands the record over with a synchronous method call. The
   sink's thread is the only place a queue and lock exist.

Why this is faster than a bus-owned drainer thread:

- ABIDES is CPU-bound under CPython. A bus-owned drainer would force a
  GIL handoff on every publish, only for the drainer to immediately
  re-acquire the GIL and do Python work. Net effect: slower than
  inline.
- A drainer thread only pays off when *the consumer releases the GIL*
  (`pyarrow` writes do; `pickle.dump` does not). Pushing the thread
  into the sink that actually benefits keeps the cost where the
  benefit is and keeps the bus zero-overhead.
- Tick-aligned batching gives sinks a natural batch boundary (Parquet
  row groups, DB transactions) without any explicit batching API.

The bus exposes lifecycle hooks `start(meta)`, `flush()`, `shutdown(meta)`
that fan out to all sinks. `shutdown()` calls `drain()` one last time,
then `on_simulation_end()` on each sink.

### 3.2 Tuple wire format, typed views at the edges

Records on the wire are **plain tuples**, not dataclasses. Producers
publish positional payloads; sinks decide whether to reify them.

```text
publish_event(agent_id, agent_type, sim_time_ns, event_type, payload)
publish_metric(agent_id, agent_type, key, value)
publish_book_snapshot(symbol, sim_time_ns, bids, asks, depth)
publish_orderbook_event(symbol, sim_time_ns, kind, order_id, price, qty, side, payload)
```

Internally, each ring-buffer entry is the argument tuple itself
(`tuple` of the args). No `__init__` on the hot path.

Typed dataclass views (`EventRecord`, `MetricRecord`,
`BookSnapshotRecord`, `OrderBookEventRecord`) live in
`abides_core/event_records.py` as a **public read-side API**: sinks
that want them call `EventRecord.from_tuple(t)` (a `classmethod` that
unpacks positionally — ~one attribute assignment per field, still
faster than a dict). `InMemorySink` and `ParquetSink` skip reification
entirely and read positional indices directly.

This eliminates per-event `__init__` cost — the dominant per-event
expense after a survey of CPython microbenchmarks — and matches how
the `OrderBook` itself already deals with `(price, qty)` tuples.

**Ordering.** No `seq` counter is published. Total order is
`(sim_time_ns, agent_id, intra_tick_counter)`, where the intra-tick
counter is a small `int` reset at the start of each kernel tick. The
kernel already ticks deterministically; this gives a stable total
order without a shared monotonic write on every publish.

### 3.3 Two-tier filtering: declarative bus-side, callable sink-side

The bus performs **only** declarative filtering. Arbitrary user logic
never runs on the producer hot path.

```text
class EventSink(Protocol):
    # Declarative — read by the bus at start() to plan dispatch.
    accept_event_types:     frozenset[str] | None = None    # None = all; empty/missing = none
    accept_metric_keys:     frozenset[str] | None = None
    accept_book_symbols:    frozenset[str] | None = None
    accept_orderbook_kinds: frozenset[str] | None = None

    # Optional sink-side predicate, applied after the bus has decided to deliver.
    def predicate_event(self, t: tuple) -> bool: ...           # default: True
    def predicate_metric(self, t: tuple) -> bool: ...
    def predicate_book_snapshot(self, t: tuple) -> bool: ...
    def predicate_orderbook_event(self, t: tuple) -> bool: ...

    def on_simulation_start(self, meta) -> None: ...
    def on_event(self, t: tuple) -> None: ...
    def on_metric(self, t: tuple) -> None: ...
    def on_book_snapshot(self, t: tuple) -> None: ...
    def on_orderbook_event(self, t: tuple) -> None: ...
    def flush(self) -> None: ...
    def on_simulation_end(self, meta) -> None: ...
```

If a sink wants a kind without filtering, it sets the corresponding
`accept_*` to `None` (the default). If it wants nothing of that kind,
it leaves the attribute unset (the Protocol default is "none"). If it
wants a fixed allowlist, it provides a `frozenset[str]`. Anything more
expressive lives in the sink, evaluated *after* the bus has already
decided to deliver — never on the publisher.

### 3.4 Pre-bound no-op publish

At `bus.start()`, for each record kind, the bus computes the union of
acceptors. If no sink accepts a kind, the bus **rebinds the
corresponding `publish_*` method on the instance to a closure that does
nothing**:

```text
NOOP = lambda *a, **k: None
if not any_sink_accepts_events:
    self.publish_event = NOOP
```

Producers see one C-level method call that returns immediately. This
is cheaper than any `if`-guarded path, because the branch is resolved
once at startup rather than on every call site. It is also cheaper
than today's `if self.owner.book_logging:` guard, which performs an
attribute load on `self.owner` plus a truthiness test on every order
book mutation.

When at least one sink accepts a kind, `publish_<kind>` is bound to
the real method, which:

1. Append the argument tuple to the per-kind ring buffer.
2. (Optionally) bump the intra-tick counter.

Two C-level operations. No filter evaluation, no allocation other than
the tuple itself, no method dispatch chain.

### 3.5 Sinks shipped in this repository

All sinks are constructed declaratively from `SimulationConfig` (see
§6). Every sink supports zero-copy access to its accumulated data
through a documented attribute, so the runner / notebooks never walk
the bus.

- **`InMemorySink`** — Captures `EventRecord` tuples in **parallel
  column arrays** (`array.array("q", ...)` for ints, plain `list` for
  objects). No list-of-dicts. Final conversion is a single
  `pd.DataFrame({col: arr, ...})`, which is materially faster than
  `from_records` and uses ~3× less peak memory than today's
  list-of-tuples-of-dicts. Replaces `agent.log` as the source of
  truth for `parse_logs_df`.
- **`BZ2PickleSink`** — Wraps the existing `BZ2PickleLogWriter`.
  Drains the in-memory column store at `on_simulation_end()` and
  writes the legacy `<run_id>/<AgentName>.bz2` artifact. **Marked
  deprecated from Phase 5; removed two minor releases later.**
- **`ParquetSink`** — Columnar; one parquet file per `event_type` (and
  per book-record kind), struct-of-arrays schema. Snappy or Zstd
  compression. Owns its own writer thread and bounded internal
  buffer; the bus delivers tuples synchronously, the sink hands them
  off to its writer. Uses `pyarrow`, import-guarded; absence yields a
  clear error at sink construction, not at first write.
- **`MetricsObserverSink`** — Adapter that exposes the existing
  `KernelObserver` Protocol on top of the bus. The shipped
  `DefaultMetricsObserver` continues to work unchanged, registered as
  a metrics-only sink. Removes the dual dispatch path inside `Agent`.
- **`OrderBookSnapshotMemorySink`** — In-memory capture of book
  snapshots, indexed by symbol, in parallel columns. Ships with
  three **sampling modes**, chosen at construction:
  - `every_update` — record on every L2 mutation. Behavioural parity
    with today's `book_log2`.
  - `on_top_of_book_change` *(default)* — record only when L1 actually
    moves. Typical analytics workload; for a busy book this is
    1–2 orders of magnitude fewer records than `every_update`.
  - `interval_ns(n)` — downsample to a fixed cadence. Useful for
    long-horizon training runs.
  The sampling decision is evaluated **inside the sink**, on the
  drain-side, after the bus has already delivered. This is the
  single biggest memory win available and the reason the snapshot
  sink can be enabled by default without reintroducing the unbounded
  memory problem.
- **`OrderBookHistoryMemorySink`** — In-memory capture of order book
  events, indexed by symbol, in parallel columns. `accept_orderbook_kinds`
  defaults to `frozenset({"EXEC"})` (fills only) — the dominant use
  case in `runner._extract_trades` — and is widened explicitly when
  callers need cancels/modifies.

Out-of-tree sinks (JSONL, SQLite, DB drop-copy, message brokers,
custom book aggregators) are documented as a supported extension
point; the library does not ship them.

### 3.6 Resilience: spill-to-disk, not block

Only async-I/O sinks (`ParquetSink`, future DB sinks) own buffers that
can fill. The bus itself does not buffer. The contract for any sink
that owns a buffer:

- If the internal buffer reaches its high watermark, **spill the
  oldest unwritten chunk to a temp file** in the run's spill
  directory and continue accepting. Reload at `on_simulation_end()`
  before the final write.
- The producer is never blocked. Simulated time is never stalled by
  analytics.
- Sinks expose a `spill_count` and `spill_bytes` counter for
  observability; the kernel logs a one-line summary at terminate.

This is strictly better than blocking-the-producer or fail-fast modes:
it is invisible to the simulator (no time distortion), it never loses
events, and disk pressure is a much more graceful degradation than
either silent stall or hard failure on a multi-hour run.

### 3.7 Error handling

- Sink exceptions during `on_event` / `on_metric` / `on_book_snapshot`
  / `on_orderbook_event` are caught **per sink** by the bus's
  `drain()` loop. The bus marks the sink as failed, stops dispatching
  to it, and surfaces the exception at `bus.shutdown()`.
- Default kernel policy on sink failure: **fail-fast at terminate**
  with a clear log line indicating which sink failed and how many
  records it had accepted before failing. Configurable to
  "best-effort warn" for research workflows where partial logs are
  better than no logs.
- All terminal disk writes use **temp-file-then-rename** to avoid
  partial files on disk-full / permission-denied. Today a crash at
  terminate after a multi-hour sim leaves no artifact at all; this
  closes that hole.
- Sinks with internal threads must surface their own thread errors
  through their `flush()` / `on_simulation_end()` return path; the
  bus does not poll them.

### 3.8 Compatibility shims (deprecation window only)

- `OrderBook.book_log2` and `OrderBook.history` become **materialize-
  once cached properties** that pull from
  `OrderBookSnapshotMemorySink` / `OrderBookHistoryMemorySink` on
  first access, cache the result, and emit `DeprecationWarning`.
  Subsequent accesses return the cached list reference. Any further
  publish after first access invalidates the cache and re-emits the
  warning. Removed in Phase 5+2.
- `Agent.log` becomes a similarly cached property pointing at
  `InMemorySink`'s reconstructed list. Same lifecycle.

The cache prevents notebooks (which often hot-loop on these
attributes) from triggering O(N) sink walks per access.

### 3.9 Payload schema registry — zero per-event allocation

The wire format (§3.2) is positional tuples. That gives consumers a
shape but not a *contract*: today an `ORDER_SUBMITTED` payload is
"whatever `order.to_dict()` happens to return," which makes any
downstream sink (Parquet, drop-copy, third-party) brittle and turns
implicit dict keys into a load-bearing public API the first time
someone writes a SQL query on the result.

We fix this without paying any per-event allocation cost.

**Core idea: schema is metadata, registered once at import time;
records on the wire stay tuples.**

```text
abides_core/event_payloads.py

# Constructed once at import. Never allocated per event.
@dataclass(frozen=True, slots=True)
class PayloadSchema:
    name:       str                         # e.g. "ORDER_EVENT"
    version:    int                         # bump on field rename/remove/reorder
    fields:     tuple[str, ...]             # positional field names
    arrow_types: tuple[pa.DataType, ...]    # for ParquetSink; lazy-built

# Schemas (one instance per logical payload shape, shared across
# many event_types).
ORDER_EVENT = PayloadSchema(
    "ORDER_EVENT", version=1,
    fields=("order_id", "symbol", "side", "qty", "limit_cents",
            "tif", "is_hidden", "is_price_to_comply", ...),
    arrow_types=(pa.int64(), pa.string(), pa.int8(), pa.int64(),
                 pa.int64(), pa.int8(), pa.bool_(), pa.bool_(), ...),
)
HOLDINGS    = PayloadSchema("HOLDINGS", 1, ("cash_cents", "positions"),
                            (pa.int64(), pa.map_(pa.string(), pa.int64())))
CASH        = PayloadSchema("CASH",     1, ("cents",),  (pa.int64(),))
DEPTH       = PayloadSchema("DEPTH",    1, ("levels",), (pa.list_(pa.list_(pa.int64())),))
AGENT_TYPE_ = PayloadSchema("AGENT_TYPE", 1, ("name",), (pa.string(),))
EMPTY       = PayloadSchema("EMPTY",    1, (),         ())

# Single source of truth: event_type → schema.
EVENT_TYPE_SCHEMA: dict[str, PayloadSchema] = {
    "ORDER_SUBMITTED":    ORDER_EVENT,
    "ORDER_ACCEPTED":     ORDER_EVENT,
    "ORDER_EXECUTED":     ORDER_EVENT,
    "ORDER_CANCELLED":    ORDER_EVENT,
    "STOP_TRIGGERED":     ORDER_EVENT,
    "MODIFY_ORDER":       ORDER_EVENT,
    "REPLACE_ORDER":      ORDER_EVENT,
    "CANCEL_SUBMITTED":   ORDER_EVENT,
    "CANCEL_PARTIAL_ORDER": ORDER_EVENT,
    "STARTING_CASH":      CASH,
    "ENDING_CASH":        CASH,
    "MARKED_TO_MARKET":   CASH,
    "FINAL_VALUATION":    CASH,
    "HOLDINGS_UPDATED":   HOLDINGS,
    "BID_DEPTH":          DEPTH,
    "ASK_DEPTH":          DEPTH,
    "AGENT_TYPE":         AGENT_TYPE_,
    "MKT_CLOSED":         EMPTY,
    ...
}
```

**Producer side: tuples replace dicts. No new allocations.**

Today's `ORDER_SUBMITTED` path allocates a `dict` from
`order.to_dict()`. After this change, `Order` grows a single
`to_payload_tuple(self) -> tuple` method that reads its `__slots__`
into a positional tuple matching `ORDER_EVENT.fields`. A tuple of N
ints is **strictly cheaper** in CPython than a dict of N items: no
hash table, no key strings, smaller object header. Net per-event
allocation count goes **down** vs. today.

```python
# Before
self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

# After
bus.publish_event(self.id, "TradingAgent", t,
                  "ORDER_SUBMITTED", order.to_payload_tuple())
```

`HOLDINGS_UPDATED` already builds a small dict (~2–10 keys) once per
trade — keep it as a frozen mapping or convert to
`(cash_cents, tuple(positions.items()))`; this fires infrequently and
allocation cost is irrelevant compared to its semantic stability.

`MKT_CLOSED` and other empty payloads use the **module-level singleton**
`EMPTY_PAYLOAD = ()` — zero allocation per event.

For scalar payloads (`STARTING_CASH`, etc.) the payload **is the
scalar**, not a 1-tuple. The schema's `fields=("cents",)` tells the
consumer "interpret the raw payload as the named scalar `cents`." This
keeps the today-shape (`logEvent("STARTING_CASH", 10_000)`) and avoids
even one tuple allocation per event.

So the rule is: **payload arity matches schema arity**. Arity 0 → the
shared `()` singleton. Arity 1 → the bare scalar. Arity ≥ 2 → a
tuple. The schema records this and consumers honor it via a tiny
`unwrap(schema, payload)` helper that always returns a tuple-shaped
view (constructing a 1-tuple lazily *only* when a consumer asks; the
hot path never does).

**Consumer side: vectorized projection from columns. No per-record
reification.**

`InMemorySink` is already columnar (parallel arrays per kind). Add a
*per-event-type* sub-bucketing inside its event store: when an event
arrives whose `event_type` is in the schema registry and whose schema
has arity ≥ 2, the sink splits the payload tuple into per-field
columns *for that event_type*. End-of-run materialization is one
`pd.DataFrame({field: arr for field, arr in zip(schema.fields, cols)})`
call per event_type — same cost model as today's `parse_logs_df`,
done once, not per record.

`ParquetSink` reads `EVENT_TYPE_SCHEMA` at `start()`, derives one
Arrow schema per `event_type` from `(fields, arrow_types)`, and
opens one writer per type. Per event: one `append` to a typed array
buffer; flush on watermark. No `payload_json` column, no
per-event reification, no JSON encode.

Out-of-tree sinks read the registry the same way. The schema *is* the
public contract.

**Validation, opt-in only.**

A debug-mode bus (`ABIDES_BUS_VALIDATE=1`) wraps `publish_event` with
an arity check (`len(payload) == len(schema.fields)` for arity ≥ 2;
type check for arity 1). Off by default — production runs pay zero.
A unit test (`test_event_payload_schema.py`) `ast.parse`s every call
site of `publish_event` / `logEvent`, extracts the literal
`event_type`, and asserts it appears in `EVENT_TYPE_SCHEMA`. New
event types cannot land without a schema entry.

**Schema evolution.**

Each `PayloadSchema` carries a `version: int`. Renaming or
reordering fields bumps it. `ParquetSink` writes
`{schema_name: version}` into file metadata; `read_parquet_logs`
checks it. The `EventBus` bumps a `bus_format_version` constant
covering the tuple wire format itself, kept separate from individual
payload schemas to allow independent evolution.

**Net cost vs. today.**

| | Today | After §3.9 |
|---|---|---|
| Per `ORDER_SUBMITTED` allocations | 1 dict (8+ str keys) | 1 tuple (no keys) |
| Per `MKT_CLOSED` allocations | 1 `None` ref | 1 `()` ref (shared) |
| Per `STARTING_CASH` allocations | 1 int ref | 1 int ref |
| Schema lookups on hot path | 0 | 0 (debug mode only) |
| Reified payload objects per event | 0 (dict counts as raw) | 0 |

The schema registry is **pure metadata**, allocated once at module
import, with no per-event cost. Consumers gain a versioned, typed,
discoverable contract. Producers shed dict construction in favor of
cheaper tuple construction.

### 3.10 Public vs. internal API boundary

After this refactor, the new modules split cleanly into a small
public surface and a larger internal one. This split is binding for
semver going forward.

**Public (semver-stable, breaking changes require a major bump):**

- `abides_core.event_payloads` — `PayloadSchema`, `EVENT_TYPE_SCHEMA`,
  the named schema instances (`ORDER_EVENT`, `HOLDINGS`, …),
  `unwrap(schema, payload)`. This is the consumer contract.
- `abides_core.event_records` — `EventRecord`, `MetricRecord`,
  `BookSnapshotRecord`, `OrderBookEventRecord` and their
  `from_tuple(...)` classmethods. The read-side reification helpers.
- `abides_core.event_sinks.EventSink` Protocol — the extension point
  for out-of-tree sinks.
- `read_parquet_logs(...)` once Phase 3 lands.
- `SimulationResult.logs` / `.l1_snapshots` / `.l2_snapshots` /
  `.trades` / `.liquidity` shapes (already public; reaffirmed here).

**Internal (may change at any time without notice):**

- `EventBus` itself, including `publish_*` signatures, drain cadence,
  ring-buffer sizing, the no-op rebind trick, `bus_format_version`,
  intra-tick counter mechanics.
- The on-the-wire tuple positions — consumers must read via the
  schema registry (`EVENT_TYPE_SCHEMA[event_type].fields`) or via
  `EventRecord.from_tuple`, never by hard-coded index.
- All concrete sink implementations *except* the `EventSink`
  Protocol surface. `InMemorySink` internals (column splitting,
  bucketing strategy) are free to change.

This boundary is documented in `docs/reference/logging-architecture.md`
when Phase 2 ships, and pinned by an `__all__` audit test.

---

## 4. Concrete changes by module

This is a high-level map, not a final code review.

- `abides_core/agent.py`
  - `logEvent` becomes one method call: `bus.publish_event(...)`. When
    no sink accepts events, `bus.publish_event` is the pre-bound
    no-op.
  - `report_metric` becomes `bus.publish_metric(...)` on the same
    bus.
  - `append_summary_log` parameter on `logEvent` is removed; the
    "important events" set becomes a sink-side allowlist (see §5).
  - `self.log` is removed from `Agent` itself; the deprecated
    cached property in `SimulationResult` covers existing readers.
- `abides_core/kernel.py`
  - Kernel owns an `EventBus`. Sinks are constructed from
    `SimulationConfig` (see §6) at `Kernel.__init__` or injected
    directly for tests.
  - The kernel main loop calls `bus.drain()` between event handlers.
    `drain()` is a no-op in the common case (empty ring buffers if
    nothing was published since the last drain).
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
  - Unchanged externally. Internally re-implemented as a sink
    adapter.
- `abides_core/event_bus.py` (new)
  - `EventBus` with per-kind ring buffers, pre-bound no-op publish,
    `start` / `drain` / `flush` / `shutdown`. No thread, no queue,
    no lock.
- `abides_core/event_records.py` (new)
  - `EventRecord`, `MetricRecord`, `BookSnapshotRecord`,
    `OrderBookEventRecord` dataclasses with `from_tuple` classmethods.
    Read-side API only.
- `abides_core/event_sinks.py` (new) — `EventSink` Protocol,
  `InMemorySink`, `BZ2PickleSink`, `MetricsObserverSink`,
  `OrderBookSnapshotMemorySink`, `OrderBookHistoryMemorySink`.
- `abides_core/parquet_sink.py` (new, optional import) —
  `ParquetSink` with internal writer thread and spill-to-disk.
- `abides_markets/abides_markets/order_book.py`
  - Each existing `if self.owner.book_logging: self.append_book_log2()`
    site (`order_book.py:389-391, 459-461, 538-539, 585-587, 638-639,
    680-682`) becomes a single
    `bus.publish_book_snapshot(symbol, sim_time_ns, bids, asks, depth)`.
    No surrounding `if` — the bus's pre-bound no-op handles the
    "nobody wants this" case more cheaply than the current attribute-
    plus-truthiness check.
  - Each `self.history.append(...)` site (LIMIT/EXEC/CANCEL/MODIFY/
    REPLACE) becomes a `bus.publish_orderbook_event(...)`. Same
    rationale: no surrounding guard needed.
  - `book_log2` and `history` become deprecated cached properties
    (§3.8). Removed in Phase 5+2.
- `abides_markets/abides_markets/agents/exchange_agent.py`
  - `book_logging` and `book_log_depth` constructor args become hints
    that translate to auto-registration of the two book memory sinks
    at config-build time (preserves backward-compatible defaults).
    Once `event_sinks` is set explicitly in config, the hints are
    ignored with a one-line info log.
- `abides_markets/abides_markets/simulation/runner.py`
  - `parse_logs_df` reads from `InMemorySink`'s parallel column
    arrays via a single `pd.DataFrame({col: arr, ...})` call
    (Phase 1 — independent of the rest).
  - `SimulationResult.logs` is sourced from `InMemorySink` instead
    of walking `end_state["agents"]`.
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

Users who need a cross-run roll-up of "starting cash, ending cash,
final valuation" register `MetricsObserverSink` plus a small custom
sink, or compute it from the per-agent logs. Both paths are documented
before Phase 5 ships.

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
  - `[{kind: "orderbook_snapshot_memory", symbols: ["ABM"],
    sampling: "on_top_of_book_change"}]` — the **"book-only fast
    access"** setup: nothing else is captured, no `agent.log` is
    built, no pickle is written, and `bus.publish_event` /
    `bus.publish_metric` / `bus.publish_orderbook_event` are all
    pre-bound no-ops. The post-sim caller reads
    `SimulationResult.l2_snapshots["ABM"]` and that's it.
  - `[{kind: "parquet", root: "./log", events: ["ORDER_*"],
    book_snapshots: {symbols: ["ABM"], sampling: "every_update"},
    compression: "zstd"}]` — production analytics setup. The
    selector keys (`events`, `metrics`, `book_snapshots`,
    `orderbook_events`) map 1:1 to the Protocol fields in §3.3.
- `event_drain_cadence: "per_handler" | int` — how often the kernel
  calls `bus.drain()`. Default `"per_handler"` (after each agent
  handler returns). Setting an integer batches every N publishes,
  reducing per-record overhead at the cost of slightly larger ring
  buffer peaks.
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
5. Each phase ships with its own tests, micro-benchmark deltas, and
changelog entry.

### Phase 0 — Hot-path microbenchmark baseline

Before any architectural change, capture per-publish cost on a
representative `rmsc04`-class config for:

- `agent.log.append((time, type, payload))`
- `if owner.book_logging: book_log2.append({...})`
- `parse_logs_df` end-to-end on a 1M-row capture

These numbers become the **acceptance gate** for Phase 2 and Phase 3a:
the new path must be at least as fast as the old one on every
configuration. Numbers, not priors, pin the design defaults.

### Phase 1 — Quick wins (no architecture)

- Replace `parse_logs_df`'s row-by-row `pd.concat` with a
  single-call construction (`pd.DataFrame.from_records` for the
  legacy path; switches to `pd.DataFrame({col: arr})` once Phase 2
  lands).
- Add `try/except` + temp-file-then-rename around all terminal
  `LogWriter` calls.
- Honour `skip_log` in `Kernel.write_summary_log`.
- No public API change.

### Phase 2 — Introduce the bus and sink Protocol

- New modules: `event_bus.py`, `event_records.py`, `event_sinks.py`.
- Tuple wire format, pre-bound no-op publish, single-threaded
  tick-batched dispatch (§3.1–3.4).
- Refactor `Agent.logEvent` and `Agent.report_metric` to publish
  directly on the bus.
- Default sink set: `InMemorySink` + `BZ2PickleSink` +
  `MetricsObserverSink` — chosen so externally observable behaviour
  is unchanged.
- `InMemorySink` uses parallel column arrays (§3.5).
- Reproducibility test: byte-identical `<run_id>/<AgentName>.bz2`
  artifact for a fixed seed against the previous release.
- **Acceptance gate:** Phase 0 microbenchmark shows the new path is
  at least as fast as the old path with the default sink set, and
  ≥ 10× faster when no sink is registered.

### Phase 3 — Ship `ParquetSink`

- Optional `pyarrow` import.
- `ParquetSink` owns its own writer thread and spill-to-disk buffer
  (§3.6). The bus does not gain a thread.
- Per-event-type Parquet schema doc; ship one schema per event type
  emitted by core agents.
- Decision needed before merge: per-event-type files vs single file
  with `payload_json` column (see §9).

### Phase 3a — Move order-book capture onto the bus

Independently mergeable from the rest of Phase 3 (only depends on
Phase 2's bus).

- Replace each `if owner.book_logging: append_book_log2()` and each
  `self.history.append(...)` site with the corresponding
  `bus.publish_*` call. **Drop the surrounding `if` guard** — the
  pre-bound no-op handles the disabled case more cheaply.
- Ship `OrderBookSnapshotMemorySink` and
  `OrderBookHistoryMemorySink`. Auto-register them when
  `ExchangeAgent.book_logging=True` and the user has not provided an
  explicit `event_sinks` list.
- Snapshot sink default sampling = `on_top_of_book_change`. A
  one-line config flag restores `every_update` for users who need
  byte parity with today's `book_log2`.
- Rewire `runner._extract_l1_close`, `_extract_l1_series`,
  `_extract_l2_series`, `_extract_trades`, and the VWAP path in
  `_extract_liquidity` to read from the two book sinks.
- Turn `OrderBook.book_log2` and `OrderBook.history` into
  deprecated **materialize-once cached** properties (§3.8); add a
  `DeprecationWarning`-on-first-access test.
- Reproducibility test: with `every_update` sampling configured,
  `SimulationResult.l1_snapshots`, `.l2_snapshots`, `.trades`, and
  `.liquidity` are byte-identical to the previous release for a
  fixed seed.
- **Acceptance gate:** on a busy `OrderBook` benchmark, the new
  path with `every_update` is at least as fast as today's
  `if owner.book_logging: book_log2.append(...)`, and the new path
  with the default `on_top_of_book_change` is materially faster
  *and* uses materially less memory.

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
  `logEvent`, `LegacySummarySink`,
  `LogWriter.write_summary_log`, and the cached compatibility
  properties (`OrderBook.book_log2`, `OrderBook.history`,
  `Agent.log`) from the codebase.
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
- **No-op publish (Phase 2).** With every sink unregistered,
  `bus.publish_event` is the literal pre-bound no-op closure
  (asserted via `is`-identity check), and a 100k-publish loop
  completes within the budget set by Phase 0.
- **Backpressure correctness (Phase 3).** With a deliberately small
  Parquet sink buffer and a slow synthetic writer, no events may be
  lost; spill files must appear and be reloaded at terminate; the
  producer wall-clock is unaffected.
- **Filter short-circuit (Phase 3a).** When no sink accepts a kind,
  the call site must not allocate the payload (verified via a
  sentinel argument that raises `__repr__` or `__bool__` if invoked).
  Covers all four record kinds.
- **Book-only sink isolation (Phase 3a).** A run configured with
  *only* `OrderBookSnapshotMemorySink` produces correct
  `SimulationResult.l2_snapshots`, builds no `agent.log`, writes no
  pickle, and (asserted via `is`-identity check on
  `bus.publish_event`) does not even call into a real publish path.
- **Snapshot sampling parity (Phase 3a).** A small fixed-seed sim
  with `every_update` sampling produces byte-identical
  `.l2_snapshots` to the legacy path; `on_top_of_book_change` mode
  produces the L1-aligned subset (unit-checked against a hand-rolled
  filter applied to the `every_update` output).
- **Cached compatibility property (Phase 3a).** Accessing
  `OrderBook.book_log2` twice without a publish in between returns
  the same `list` object (`is`-identity). A publish in between
  invalidates the cache and re-emits the warning.
- **Sink failure isolation (Phase 2 onward).** A sink that raises in
  `on_event` is marked failed and dropped from dispatch; the bus
  surfaces the exception at `shutdown()` per the configured policy.
- **Parquet schema stability (Phase 3).** Round-trip a recorded
  Parquet artifact through `pyarrow` and assert the schema matches
  the pinned reference for each shipped event type.
- **Determinism over multiple episodes (`abides-gym`).** Running N
  episodes back-to-back must produce the same final per-episode
  artifacts as N independent runs.
- **Memory bound (Phase 3).** A long-running synthetic sim with a
  Parquet sink keeps RSS under a configured ceiling; spill files
  appear and are cleaned up at terminate.

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
   surprise research workflows) vs best-effort warn (lenient, may
   hide bugs). Recommendation: fail-fast, with a clearly documented
   config override.
5. **Multiprocess parallel runs.** Each worker owns its own bus and
   sinks; no cross-process queueing. Confirm this matches the planned
   cross-run aggregation story before Phase 5.
6. **Drain cadence default (Phase 2).** `"per_handler"` is the safest
   default (lowest peak ring-buffer size, simplest mental model),
   but a small fixed batch (e.g. 16) might be cheaper on busy
   exchange handlers that publish many records per call. Decide
   from Phase 0 numbers.
7. **Deprecation timeline.** This plan says Phase 5 + 2 for removal.
   Confirm the absolute timeline in
   `docs/project/release-process.md` when Phase 5 is scheduled.
8. **Schema registry phasing (§3.9).** Land the registry and tuple
   payloads in Phase 2 alongside the bus (so `InMemorySink` can be
   columnar from day one), or land in Phase 3 just-in-time for
   `ParquetSink` (smaller blast radius)? Recommendation: Phase 2 —
   the producer-side dict→tuple swap is the disruptive part and
   should ride with the agent-API change, not be a follow-up.

---

## 10. What this plan does *not* change

- The stdlib `logging` subsystem (System A). It works. The only
  related improvement is making `show_trace_messages` discoverable
  via config (§6).
- The `KernelObserver` Protocol's external surface. It becomes a
  sink adapter internally; users registering observers see no change.
- The semantics of `Agent.report_metric` from the caller's point of
  view. It still pushes a numeric `(key, value)` to all interested
  observers.
- The `SimulationResult.logs` shape returned by `run_simulation`. It
  remains a `DataFrame` produced by `parse_logs_df` (now over the
  `InMemorySink` column arrays).
- The `SimulationResult.l1_snapshots` / `.l2_snapshots` / `.trades`
  / `.liquidity` shape. They remain identical; only the source of
  truth moves from `OrderBook.book_log2` / `OrderBook.history` to
  the two book sinks.
- Reproducibility guarantees. Single-threaded dispatch makes
  determinism true by construction.

---

## 11. Performance budget and design constraints

This plan is performance-first. The constraints below are binding on
any implementation; if the implementation cannot meet them, the
design must be revisited rather than the constraint relaxed.

- **Hot-path publish, no-sink case:** one C-level call into a pre-
  bound `lambda *a, **k: None`. No attribute load on the publisher,
  no truthiness test, no dict lookup. Strictly cheaper than today's
  `if self.owner.book_logging:` guard.
- **Hot-path publish, one-sink case:** one method call, one tuple
  construction, one `list.append`. No filter evaluation, no
  dataclass `__init__`, no `seq` increment, no thread handoff.
- **Drain cost:** O(records since last drain) with a single
  attribute lookup per sink per kind. Empty-buffer drains are
  effectively free (one length check per kind).
- **Ordering:** total order is `(sim_time_ns, agent_id,
  intra_tick_counter)`. No shared monotonic counter on the wire.
- **Threads:** zero in core. Sinks that need them own them.
- **Memory:** in-memory sinks use parallel column arrays, not lists
  of dicts. Snapshot sink defaults to `on_top_of_book_change`
  sampling. Async-I/O sinks spill to disk at the high watermark
  rather than block or drop.
- **Determinism:** true by construction (single-threaded dispatch);
  pinned by byte-equality regression tests on every reproducibility-
  sensitive sink.
- **Payload schema:** metadata-only, registered once at import (§3.9).
  No per-event class allocation, no per-event registry lookup on the
  hot path. Producer payloads are bare scalars (arity 1), the shared
  `()` singleton (arity 0), or a tuple (arity ≥ 2) — strictly cheaper
  than today's per-event `dict`.
