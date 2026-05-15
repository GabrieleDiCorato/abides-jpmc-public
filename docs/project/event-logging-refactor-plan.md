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

**Business framing — gym throughput is the headline.** The ultimate
consumer of `abides-ng` is reinforcement-learning training in
`abides-gym`. The gym today already mutes log subsystems by hand
(`abides-gym/abides_gym/envs/markets_environment.py:54-56` sets
`book_logging=False`, `exchange_log_orders=False`) because the current
hot path is too expensive for inner training loops. Even with those
muted, every agent still pays an attribute load + truthiness test +
tuple build + `list.append` per `logEvent` — for data the gym throws
away on `env.reset()`. The §3.4 pre-bound no-op publish is the single
biggest *business* win in this refactor: a gym episode with no sinks
configured pays one C-level call into a `lambda *a, **k: None` per
publish. Across millions of training steps that is the dominant
compute saving.

---

## 1. Goals (in priority order)

0. **Gym training throughput.** `abides-gym` episodes with no sinks
   configured must pay one C-level call into a pre-bound no-op per
   publish — strictly cheaper than today's `if self.log_events:`
   guard. Headline target: ≥1.5× episode throughput on the existing
   `markets_environment` benchmark with default sinks disabled, vs. the
   current release. This is goal #0 because it is the only goal whose
   delta is visible to end users (the RL researchers training on this
   simulator).
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
- **Payloads carry no live references.** No event payload may retain a
  reference to a live `Agent`, `Message`, `OrderBook`, or mutable
  collection owned by a producer. Permitted payload shapes are:
  scalars (`int`/`float`/`str`/`bool`/enum value), tuples of scalars,
  and lists/tuples of tuples-of-scalars. Today
  `abides-markets/abides_markets/agents/exchange_agent.py:406, 414, 420`
  publishes raw `Message` objects as payloads — these pin sender,
  recipient, the order they wrap, and (for some message types) book
  snapshots, for the lifetime of the agent log. This is the most
  likely root cause of long-sim OOM. A debug-mode `isinstance` check
  in the bus enforces this invariant; release builds skip the check.
- **No formatted-string payloads.** Payloads must be structured data,
  not human-readable strings. Today `BEST_BID` / `BEST_ASK` /
  `LAST_TRADE` / `MARK_TO_MARKET` / `STOP_ORDER_ACCEPTED` /
  `FINAL_HOLDINGS` allocate f-strings on the hot path
  (`order_book.py:212, 218, 234`; `trading_agent.py:253, 1552`;
  `exchange_agent.py:722`). Replaced by tuple schemas (`QUOTE`, etc.;
  see §3.9). The schema-validator unit test (§3.9) AST-walks every
  `publish_event` callsite and rejects `f"..."` and `str(x)` payload
  literals.

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

  **Snapshot construction is deferred into the sink.** Today
  `OrderBook.append_book_log2` (`order_book.py:684-691`) runs on
  every L2 mutation (5 callsites: `order_book.py:391, 461, 539, 587,
  639, 682`) and unconditionally allocates two list comprehensions of
  depth `book_log_depth`, two `np.array` headers + buffers, a 3-key
  dict, and an append — *before* anyone has decided whether the
  snapshot will be retained. The bus instead carries a **deferred
  snapshot token**: `(symbol, time_ns, order_book_ref, depth)`. The
  sink calls `get_l2_bid_data` / `get_l2_ask_data` only after the
  sampling decision says "record." For the default
  `on_top_of_book_change` mode the sink first calls cheap
  `get_l1_bid_data()` / `get_l1_ask_data()`; the L2 walk happens
  only on actual L1 movement. Net effect for analytics workloads:
  L1 lookup on every mutation, L2 walk on the (much rarer) L1
  changes, instead of full L2 walk on every mutation. The
  `order_book_ref` in the token is a back-reference held only for
  the duration of the drain — never retained past the sink's
  decision — and is therefore exempt from the §2 "no live
  references in payloads" invariant by construction.
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
HOLDINGS    = PayloadSchema("HOLDINGS", 1,
                            ("symbol", "delta_qty", "qty_after", "cash_after_cents"),
                            (pa.string(), pa.int64(), pa.int64(), pa.int64()))
CASH        = PayloadSchema("CASH",     1, ("cents",),  (pa.int64(),))
DEPTH       = PayloadSchema("DEPTH",    1, ("levels",), (pa.list_(pa.list_(pa.int64())),))
QUOTE       = PayloadSchema("QUOTE",    1,
                            ("symbol", "price_cents", "qty"),
                            (pa.string(), pa.int64(), pa.int64()))
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
    "BEST_BID":           QUOTE,
    "BEST_ASK":           QUOTE,
    "LAST_TRADE":         QUOTE,
    "BID_DEPTH":          DEPTH,
    "ASK_DEPTH":          DEPTH,
    "AGENT_TYPE":         AGENT_TYPE_,
    "MKT_CLOSED":         EMPTY,
    ...
}
```

**Producer side: tuples replace dicts. Allocation count drops ~5×, not just "tuple is cheaper than dict."**

Today's `ORDER_SUBMITTED` path allocates more than a single dict. The
call chain is:

```python
# abides-markets/abides_markets/orders.py:101-104
def to_dict(self) -> dict[str, Any]:
    as_dict = deepcopy(self).__dict__   # invokes LimitOrder.__deepcopy__
    as_dict["time_placed"] = fmt_ts(self.time_placed)
    return as_dict
```

`deepcopy(self)` invokes the subclass `__deepcopy__` (e.g.
`orders.py:176-197` for `LimitOrder`), which constructs a fresh
`LimitOrder` instance plus a `deepcopy(self.tag)`. So every
`ORDER_SUBMITTED` / `ORDER_ACCEPTED` / `ORDER_EXECUTED` /
`ORDER_CANCELLED` / `MODIFY_ORDER` / `REPLACE_ORDER` /
`CANCEL_SUBMITTED` / `CANCEL_PARTIAL_ORDER` / `PARTIAL_CANCELLED` /
`ORDER_MODIFIED` / `ORDER_REPLACED` / `STOP_ORDER_SUBMITTED` event
today allocates roughly **five distinct objects**:

1. A new `LimitOrder` instance (no `__slots__` → dict-backed,
   ~280 B header + dict).
2. A recursive `deepcopy(self.tag)`.
3. The `__dict__` snapshot returned by `deepcopy(self).__dict__`.
4. The `fmt_ts(...)` string for `time_placed`.
5. The outer `(time, type, payload)` tuple in `Agent.logEvent` plus
   the `list.append`.

After this change, `Order` grows a single
`to_payload_tuple(self) -> tuple` method that reads its `__slots__`
into a positional tuple matching `ORDER_EVENT.fields`. The chain
collapses to **one tuple allocation** (the wire payload) plus the
ring-buffer append. That is a ~5× reduction in per-order-event
allocation count, not the marginal "tuple vs dict" win the prior
revision implied. Slotting `Order` (see §4) is a precondition: it
both shrinks per-instance memory and lets `to_payload_tuple()` read
slots positionally instead of doing N attribute lookups by name.

```python
# Before
self.logEvent("ORDER_SUBMITTED", order.to_dict(), deepcopy_event=False)

# After
bus.publish_event(self.id, "TradingAgent", t,
                  "ORDER_SUBMITTED", order.to_payload_tuple())
```

`HOLDINGS_UPDATED` today fires from five sites in
`abides-markets/abides_markets/agents/trading_agent.py` (lines 287,
1145, 1238, 1265, 1295), each with `deepcopy_event=True` against the
*entire* `self.holdings` dict. That cost scales with portfolio
breadth — fine for toy single-symbol configs, expensive the moment a
strategy spans many symbols. The new schema is the **per-fill delta**,
not the snapshot: `(symbol, delta_qty, qty_after, cash_after_cents)`,
one 4-int tuple (~64 B) per fill. Readers that want a holdings
snapshot reconstruct by replaying — `InMemorySink` exposes a
convenience method that materializes a snapshot at a given
`sim_time_ns` if a notebook asks. Eliminates the deepcopy entirely.

`MKT_CLOSED` and other empty payloads use the **module-level singleton**
`EMPTY_PAYLOAD = ()` — zero allocation per event.

For scalar payloads (`STARTING_CASH`, etc.) the payload **is the
scalar**, not a 1-tuple. The schema's `fields=("cents",)` tells the
consumer "interpret the raw payload as the named scalar `cents`." This
keeps the today-shape (`logEvent("STARTING_CASH", 10_000)`) and avoids
even one tuple allocation per event.

**Enums are stored as `int8`, not as Python enum objects.** `Order.side`
is `Side` and `LimitOrder.time_in_force` is `TimeInForce`
(`orders.py:14, 25`). Parquet has no native enum type and pickled
enums are not portable across schema versions. Producers call `.value`
when building the payload tuple (or, after slotting, store an `int8`
slot directly). The schema records the enum class so the read-side
`EventRecord.from_tuple` rehydrates to a typed enum on demand. Hot
path stays at zero added cost.

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
type check for arity 1) **and** an `isinstance` check enforcing the
§2 invariant (no `Message`/`Agent`/`OrderBook` references in
payloads). Off by default — production runs pay zero. A unit test
(`test_event_payload_schema.py`) `ast.parse`s every call site of
`publish_event` / `logEvent`, extracts the literal `event_type`,
asserts it appears in `EVENT_TYPE_SCHEMA`, and rejects payload
expressions that are f-strings (`ast.JoinedStr`) or `str(x)` calls.
New event types cannot land without a schema entry; no callsite can
silently regress to a stringified payload.

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
| Per `ORDER_*` allocations | ~5 (deepcopy chain: new `LimitOrder` + tag deepcopy + `__dict__` snapshot + `fmt_ts` string + outer 3-tuple) | 1 tuple |
| Per `BEST_BID`/`BEST_ASK`/`LAST_TRADE` allocations | 1 f-string (parse spec, intermediates, concat) | 1 tuple of 3 ints |
| Per `HOLDINGS_UPDATED` allocations | 1 deepcopy of full `self.holdings` dict (scales w/ portfolio breadth) | 1 tuple of 4 ints |
| Per `MKT_CLOSED` allocations | 1 `None` ref | 1 `()` ref (shared) |
| Per `STARTING_CASH` allocations | 1 int ref | 1 int ref |
| Schema lookups on hot path | 0 | 0 (debug mode only) |
| Reified payload objects per event | 0 (dict counts as raw) | 0 |
| Live references retained in payload | unbounded (raw `Message` objects from `exchange_agent.py:406, 414, 420`) | none (§2 invariant, debug-checked) |

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

### 3.11 Event vocabulary — final list (gating Phase 2)

Today there are **51 unique `logEvent` callsites across 8 files** with
inconsistent payload shapes for semantically related events — for
example `STOP_ORDER_ACCEPTED` publishes `str(order)`
(`exchange_agent.py:722`) while `STOP_ORDER_SUBMITTED` publishes
`order.to_dict()`. The schema registry in §3.9 needs one row per
surviving event type, and Phase 2 cannot ship until that table is
finalized. This refactor is the right time to **delete dead event
types and standardize the survivors**; doing it later is a breaking
change to the public schema contract.

The deliverable is a table in `docs/reference/logging-architecture.md`
with one row per surviving `event_type`, each row carrying:

| Column | Notes |
|---|---|
| `event_type` | Stable string identifier; the registry key. |
| `producer` | File and class that publishes it. |
| `frequency tier` | `per-tick` / `per-quote` / `per-fill` / `per-order` / `per-run`. Drives the perf budget. |
| `schema` | One of the §3.9 `PayloadSchema` instances. |
| `consumers` | Which sinks/notebooks/runner extractors read it today. |
| `disposition` | `keep` / `rename → X` / `merge with Y` / `delete (no consumer)`. |

The table is built bottom-up by walking every `logEvent` call site
(8 files: `agent.py`, `kernel.py`, `trading_agent.py`,
`exchange_agent.py`, `noise_agent.py`, `value_agent.py`,
`market_maker_agent.py`, `order_book.py`). Standardization rules:

- Every `ORDER_*` event uses `ORDER_EVENT` schema. No mix of
  `order.to_dict()` and `str(order)` survives.
- Every quote event (`BEST_BID`, `BEST_ASK`, `LAST_TRADE`) uses
  `QUOTE` schema. No f-string payloads survive.
- Every `*_CASH` / `MARKED_TO_MARKET` / `FINAL_VALUATION` event uses
  `CASH` schema; the `MARK_TO_MARKET` f-string at
  `trading_agent.py:1552` is split into a structured payload plus an
  optional human-readable summary line at `INFO` log level.
- Event types with no observed consumer — neither `runner._extract_*`,
  nor a notebook in `notebooks/`, nor an out-of-tree caller surfaced
  during Phase 0 — are deleted, not migrated.

**Acceptance:** Phase 2 cannot merge until the table is reviewed
and every surviving `event_type` has a `PayloadSchema` row in
`EVENT_TYPE_SCHEMA`. The AST-walking validator test fails the build
on any callsite whose `event_type` is not in the registry.

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
  - `Agent.log_events` (`agent.py:161`) becomes a **deprecated no-op**
    in Phase 4. Subsumed by sink-side declarative filters
    (`accept_event_types`) and the §3.4 no-op rebind. Configs that
    set it get a `DeprecationWarning` pointing at the equivalent
    sink config. Removed in Phase 5+2. Without this deprecation the
    system has *four* overlapping mute switches after the refactor
    (`log_events`, `log_orders`, `book_logging`, sink filters) and
    every publish pays for both the new declarative gate and the
    legacy `if self.log_events:` check — strictly worse than today.
- `abides_markets/abides_markets/orders.py`
  - Convert `Order`, `LimitOrder`, `MarketOrder`, `StopOrder` to
    `__slots__`. **Precondition** for cheap `to_payload_tuple()`
    (slot-tuple read instead of N attribute lookups by name) and a
    ~280 B/order RSS reduction (no per-instance `__dict__`). The
    existing custom `__deepcopy__` makes pickle compatibility
    painless.
  - Add `to_payload_tuple(self) -> tuple` on each subclass returning
    a positional tuple matching `ORDER_EVENT.fields` (with `Side` and
    `TimeInForce` stored as `int8` via `.value`; see §3.9 enum rule).
  - Drop the `deepcopy(self).__dict__` antipattern in
    `to_dict()` (`orders.py:101-104`). Its only purpose was guarding
    against later mutation of the published dict; the new sink
    contract — sinks must not mutate received tuples — makes the
    guard unnecessary. `to_dict()` is retained only for the
    deprecation window as a thin wrapper around `to_payload_tuple()`
    for any out-of-tree code that depends on it; removed in Phase
    5+2.
- `abides_markets/abides_markets/agents/trading_agent.py`
  - `TradingAgent.log_orders` becomes a **deprecated no-op** in
    Phase 4 (same reasoning as `log_events`). It guards 25+ order-
    related callsites today; after Phase 2 those callsites publish
    unconditionally and the sink filters decide. Removed in Phase
    5+2.
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
    ignored with a one-line info log. The `book_logging` flag itself
    becomes a **deprecated no-op** in Phase 4 (subsumed by the
    snapshot sink's `accept_book_symbols`); removed in Phase 5+2.
  - Replace the three `self.logEvent(message.type(), message)` sites
    (`exchange_agent.py:406, 414, 420`) with payload tuples derived
    from the message's public fields. Today these publish raw
    `Message` objects, which pin sender, recipient, the wrapped
    `Order`, and (for some message types) book snapshots for the
    lifetime of the per-agent log — the most likely root cause of
    long-sim OOM. Each gets a dedicated schema entry in §3.9.
  - Replace the `STOP_ORDER_ACCEPTED` payload `str(order)`
    (`exchange_agent.py:722`) with `order.to_payload_tuple()`
    matching the `ORDER_EVENT` schema, harmonizing with
    `STOP_ORDER_SUBMITTED`.
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

There is no existing `benchmarks/` tree in the repo; Phase 0 ships
one, alongside a CI job that runs it on every PR touching a publisher
or sink. The four numbers below are **named acceptance gates** for
subsequent phases. No phase merges if its corresponding gate
regresses.

| Benchmark | Phase that gates on it | Target |
|---|---|---|
| `gym_episode_throughput_no_sinks` — `markets_environment` episode wall-clock with all sinks unconfigured | Phase 2 | **≥ 1.5×** today's release. Headline gym-throughput goal #0 from §1. |
| `single_agent_run_with_default_sinks` — full default-config simulation | Phase 2, Phase 3a | **≤ 1.0×** today (no regression with the new path under the default sink set). |
| `peak_rss_long_sim` — long-horizon synthetic sim | Phase 3 | Bounded by `O(sink_buffer_high_watermark)`, **not** by event count. Peak in-memory event store is `Θ(M)` columnar arrays in one `InMemorySink`, not `Θ(N·M)` per-agent Python lists. |
| `parse_logs_df_p99` — `parse_logs_df` on a 1M-row capture | Phase 1 | **Linear** in row count; the current O(N²) `pd.concat` loop is gone. |

These numbers, not priors, pin the design defaults. The Phase 2 / 3a
acceptance gates further down ("at least as fast as the old path")
are tied to these named benchmarks.

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
- **Peak in-memory event store:** `Θ(M)` columnar arrays in one
  shared `InMemorySink`, not `Θ(N·M)` per-agent Python lists like
  today's `agent.log`. For typical constant-`M`-per-agent workloads
  this is a `~agent_count×` memory reduction *before* any per-record
  encoding wins. This is a structural improvement that the
  per-event allocation work (§3.9) compounds, not duplicates.
- **Determinism:** true by construction (single-threaded dispatch);
  pinned by byte-equality regression tests on every reproducibility-
  sensitive sink.
- **Payload schema:** metadata-only, registered once at import (§3.9).
  No per-event class allocation, no per-event registry lookup on the
  hot path. Producer payloads are bare scalars (arity 1), the shared
  `()` singleton (arity 0), or a tuple (arity ≥ 2) — strictly cheaper
  than today's per-event `dict`.
