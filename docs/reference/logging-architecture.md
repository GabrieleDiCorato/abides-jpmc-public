# ABIDES Logging — Comprehensive Analysis

**Status:** Analysis (not a plan). Findings only — no decisions encoded.
**Scope:** Every form of "logging" present in `abides-core`, `abides-markets`,
`abides-gym`. The standard Python `logging` module, the per-agent
`Agent.logEvent()` system, the centralized `summary_log`, and how each
flows through to disk and downstream consumers.
**Audience:** A reviewer who needs the full picture before deciding what
to change.

---

## 0. Executive summary — three independent systems, one folder

ABIDES has **three logging subsystems** that share the name "logging" but
do completely different things and barely interact:

| System | What it logs | Where it goes | Who reads it |
|---|---|---|---|
| **Python `logging`** | Lifecycle, periodic stats, debug traces | stdout (via `basicConfig`) | Operator watching the console |
| **Per-agent event log** (`Agent.logEvent`) | Every business event the agent emits | In-memory `agent.log` list → `./log/<run_id>/<AgentName>.bz2` (BZ2-pickled DataFrame) | `parse_logs_df()` → `run_simulation()` result → notebooks, metrics |
| **Summary log** (`Kernel.append_summary_log`) | A handful of "important" events (cash, holdings, valuation) | In-memory `kernel.summary_log` list → `./log/<run_id>/summary_log.bz2` | **Nobody in this fork.** Originally intended for "separate statistical summary programs" (see [§A](#a-original-intent-of-summary_log-archaeology)) that were never released. |

This split is invisible from configuration — `skip_log` controls the
last two; `log_level` only controls the first. They are written into the
same `./log/<run_id>/` directory but follow completely different
lifecycles and purposes. Most of the friction this codebase has around
"logging" comes from conflating them.

---

## 1. System A — Python `logging` module

### 1.1 Loggers

Every module follows the standard `logger = logging.getLogger(__name__)`
pattern. **18 named loggers** across the three packages
([abides-core/abides_core/kernel.py L17](../../abides-core/abides_core/kernel.py#L17),
[abides-core/abides_core/agent.py L15](../../abides-core/abides_core/agent.py#L15),
oracles, every agent type, `order_book.py`, etc.).

Two CLI entry points use a hardcoded `"abides"` name instead:
[abides-core/abides_core/abides.py L17](../../abides-core/abides_core/abides.py#L17)
and [abides-core/scripts/abides L18](../../abides-core/scripts/abides#L18).

### 1.2 Configuration

There is exactly one configuration call, replicated in three places:

```python
logging.basicConfig(
    level=config["stdout_log_level"],
    format="[%(process)d] %(levelname)s %(name)s %(message)s",
)
```

Sites: [abides.py L47-50](../../abides-core/abides_core/abides.py#L47),
[abides.py L150-153](../../abides-core/abides_core/abides.py#L150),
[scripts/abides L95-98](../../abides-core/scripts/abides#L95).

`stdout_log_level` is plumbed from `SimulationConfig.simulation.log_level`
(default `"INFO"`, one of `DEBUG/INFO/WARNING/ERROR/CRITICAL`).

**No `FileHandler`, no `RotatingFileHandler`, nothing else is attached
by ABIDES.** Standard Python logging output goes to stdout/stderr only.
The kernel does not write a `simulation.log` file anywhere.

### 1.3 What the kernel logs

Categorized by purpose ([kernel.py](../../abides-core/abides_core/kernel.py)):

- **Lifecycle (DEBUG):** `Kernel initialized`, `Kernel started`,
  `Agent.kernel_initializing/starting/stopping/terminating`,
  `Kernel Event Queue begins/empty`. Useful for tracing setup, mostly
  silent at INFO.
- **Periodic checkpoint (INFO):** Every 100,000 messages,
  [kernel.py L325](../../abides-core/abides_core/kernel.py#L325)
  emits a one-line snapshot:
  `--- Simulation time: ..., messages processed: ..., wallclock elapsed: ...s ---`.
  This is the only INFO output during the hot loop. For a 7-million
  message sim this fires ~70 times.
- **Trace (DEBUG, gated by `show_trace_messages`):** Per-pop, per-dispatch,
  per-requeue debug lines. ~6 sites in `runner()` and `_enqueue()`. Off
  by default. Very expensive when on.
- **Termination summary (INFO):** Event-queue elapsed, msgs/sec,
  per-agent-type mean ending value (the financial leak — see system C),
  `Simulation ending!`.

### 1.4 The `show_trace_messages` flag

[kernel.py L202](../../abides-core/abides_core/kernel.py#L202) initializes
it to `False`. It is read in 6 places inside the hot loop. **It is set
nowhere in the codebase** — not by any config, not by any test. Users
who want trace output have to do
`kernel.show_trace_messages = True` manually after constructing the
kernel. There is no path from `SimulationConfig` to this flag.

### 1.5 Verdict on system A

Mostly fine. Standard, predictable, well-behaved Python logging. Two
gaps:

- No way to send stdout logs to a file alongside the per-agent `.bz2`
  files. Documentation in
  [parallel-simulation.md L234](../ai/parallel-simulation.md#L234)
  shows users how to attach a `FileHandler` themselves.
- `show_trace_messages` is undiscoverable.

---

## 2. System B — per-agent event log (`Agent.logEvent`)

This is the **real** log: the per-event business record that downstream
analytics consume. Don't confuse it with system A.

### 2.1 Method

[agent.py L137-174](../../abides-core/abides_core/agent.py#L137):

```python
def logEvent(
    self,
    event_type: str,
    event: Any = "",
    append_summary_log: bool = False,
    deepcopy_event: bool = False,
) -> None:
```

Each agent owns `self.log: list[tuple[NanosecondTime, str, Any]]`
([agent.py L68](../../abides-core/abides_core/agent.py#L68)). On every
`logEvent` call, a tuple `(current_time, event_type, event)` is appended.

Two flags:
- `deepcopy_event=True` — copy the event payload before storing, so
  later mutation of the original dict doesn't poison historical
  entries. Default `False` for performance. Used at 5 call sites
  where holdings dicts are logged
  ([trading_agent.py L288, 1148, 1241, 1268, 1298](../../abides-markets/abides_markets/agents/trading_agent.py#L288)
  + [noise_agent.py L124, 132](../../abides-markets/abides_markets/agents/noise_agent.py#L124)
  + [value_agent.py L120](../../abides-markets/abides_markets/agents/value_agent.py#L120)).
- `append_summary_log=True` — also push to the kernel's central
  summary list (system C). Used at the same handful of "final state"
  call sites.

### 2.2 Event vocabulary (~25 unique types)

**Core lifecycle:** `AGENT_TYPE`.
**Cash & holdings:** `STARTING_CASH`, `ENDING_CASH`,
`FINAL_CASH_POSITION`, `FINAL_VALUATION`, `HOLDINGS_UPDATED`,
`MARKED_TO_MARKET`.
**Order submission:** `ORDER_SUBMITTED`, `STOP_ORDER_SUBMITTED`,
`ORDER_ACCEPTED`, `STOP_ORDER_ACCEPTED`.
**Order execution & cancellation:** `ORDER_EXECUTED`, `ORDER_CANCELLED`,
`PARTIAL_CANCELLED`, `CANCEL_SUBMITTED`, `CANCEL_PARTIAL_ORDER`.
**Order modification:** `MODIFY_ORDER`, `ORDER_MODIFIED`,
`REPLACE_ORDER`.
**Market data:** `BID_DEPTH`, `ASK_DEPTH`, `LAST_TRADE`,
`STOP_TRIGGERED`, `MKT_CLOSED`.
**Exchange:** raw `Message.type()` strings — exchange logs every
incoming message at
[exchange_agent.py L420](../../abides-markets/abides_markets/agents/exchange_agent.py#L420).
**Execution algos:** custom strings from `BaseExecutionAgent` and
subclasses.

### 2.3 Disk persistence

Per-agent: at termination, each agent's `self.log` is converted to a
DataFrame `(EventTime, EventType, Event)` indexed by `EventTime`
([agent.py L137-138](../../abides-core/abides_core/agent.py#L137)),
then handed to
[kernel.py write_log() L713-753](../../abides-core/abides_core/kernel.py#L713).

`write_log()`:
- Skips if `self.skip_log`.
- Builds path `./log/<log_dir>/<agent_name_no_spaces>.bz2`.
- Calls `df.to_pickle(path, compression="bz2")`.

Both the path and the format are hardcoded. The kernel decides
filesystem layout, choice of pickle, choice of bz2.

A handful of agents call `write_log()` again with a custom `filename` for
extra artifacts — e.g.
[exchange_agent.py L327](../../abides-markets/abides_markets/agents/exchange_agent.py#L327)
writes `fundamental_<symbol>.bz2`.

### 2.4 Two control flags on `Agent` itself

- `log_events: bool = True` ([agent.py L33](../../abides-core/abides_core/agent.py#L33))
  — if `False`, `logEvent()` becomes a no-op. The agent records
  nothing.
- `log_to_file: bool = True` ([agent.py L34](../../abides-core/abides_core/agent.py#L34))
  — if `False`, the agent's log stays in memory and is never written
  to disk (but is still exposed via `agent.log`).

These are **per-agent-instance** flags. There is no global way to
"disable order logs across all agents". Configs that want to suppress
order logs for, say, noise agents must set the flag per-agent-type at
build time. The config system has helpers (`log_orders` override at
[test_config_system.py L1179](../../abides-markets/tests/test_config_system.py#L1179)).

### 2.5 Downstream: `parse_logs_df`

[abides-core/abides_core/utils.py L154-186](../../abides-core/abides_core/utils.py#L154):

```python
def parse_logs_df(end_state: dict) -> pd.DataFrame:
    # iterate end_state["agents"], walk each agent.log,
    # flatten the Event payload (dict-expanded if dict),
    # add agent_id and agent_type columns,
    # concat one row at a time into a single DataFrame.
```

This is the **canonical reader** of system B. It is called from:

- [abides-markets/abides_markets/simulation/runner.py L349](../../abides-markets/abides_markets/simulation/runner.py#L349)
  — populates `SimulationResult.logs` when the requested `ResultProfile`
  includes agent logs.
- [data-extraction.md](../ai/data-extraction.md) — the
  documented public API for users.
- Notebook examples (e.g. `demo_ABIDES-Markets.ipynb`).

`parse_logs_df` operates on **in-memory `agent.log` lists**, not on the
written `.bz2` files. The `.bz2` files are an export artifact, not the
runtime data path.

The metrics system in
[abides-markets/abides_markets/simulation/metrics.py](../../abides-markets/abides_markets/simulation/metrics.py)
consumes the *parsed* DataFrame, not the raw logs.

### 2.6 Performance and memory

- **Per-event allocation:** every `logEvent` call appends a 3-tuple.
  Cheap. `deepcopy_event=True` is more expensive but rare.
- **End-of-simulation conversion:** `pd.DataFrame(...)` over
  potentially millions of rows, then `to_pickle(compression="bz2")`.
  Both serial, both slow for large sims. No incremental flushing. No
  streaming format.
- **`parse_logs_df` is `pd.concat([pd.DataFrame([row]) for row in ...])`
  one row at a time** — O(N²) due to repeated concat. For a 7M-event
  sim this is the wrong shape. Building once with `pd.DataFrame(rows)`
  would be orders of magnitude faster. This is a real performance
  smell, not theoretical.

### 2.7 Verdict on system B

This is the **load-bearing** logging system. Everything downstream
(metrics, plots, replay tooling) depends on it. It works, but has three
real issues:

1. **Format is fused into the kernel.** No way to swap pickle for
   parquet, no way to mock the writer for tests.
2. **`parse_logs_df` is O(N²).** Hot for large sims.
3. **`log_events` / `log_to_file` are per-instance**, awkward to set
   globally.

---

## 3. System C — `summary_log` (the centralized one)

### 3.1 The mechanism

[kernel.py L87](../../abides-core/abides_core/kernel.py#L87): a list
populated by
[`append_summary_log()`](../../abides-core/abides_core/kernel.py#L755-772):

```python
def append_summary_log(self, sender_id, event_type, event):
    self.summary_log.append({
        "AgentID": sender_id,
        "AgentStrategy": self.agents[sender_id].type,
        "EventType": event_type,
        "Event": event,
    })
```

Triggered from [agent.py L172-174](../../abides-core/abides_core/agent.py#L172)
*only* when the agent passes `append_summary_log=True` to `logEvent`.

### 3.2 Who actually uses it

Verified by grep — only **7 call sites** pass `append_summary_log=True`:

| Event | Caller |
|---|---|
| `STARTING_CASH` | [trading_agent.py L233](../../abides-markets/abides_markets/agents/trading_agent.py#L233) |
| `FINAL_CASH_POSITION` | [trading_agent.py L256](../../abides-markets/abides_markets/agents/trading_agent.py#L256) |
| `ENDING_CASH` | [trading_agent.py L261](../../abides-markets/abides_markets/agents/trading_agent.py#L261) |
| `HOLDINGS_UPDATED` | [trading_agent.py L288](../../abides-markets/abides_markets/agents/trading_agent.py#L288) |
| `FINAL_VALUATION` | [noise_agent.py L124, 132](../../abides-markets/abides_markets/agents/noise_agent.py#L124) |
| `FINAL_VALUATION` | [value_agent.py L120](../../abides-markets/abides_markets/agents/value_agent.py#L120) |

These are all **end-of-day financial summary events**. The intent
(judging from the call sites) was to give downstream tools a fast path
to "the final state of everyone's books" without scanning every agent's
full log.

### 3.3 Disk write

[kernel.py write_summary_log() L774-783](../../abides-core/abides_core/kernel.py#L774):

```python
def write_summary_log(self) -> None:
    path = os.path.join(".", "log", self.log_dir)
    file = "summary_log.bz2"
    if not os.path.exists(path):
        os.makedirs(path)
    df_log = pd.DataFrame(self.summary_log)
    df_log.to_pickle(os.path.join(path, file), compression="bz2")
```

**Bug:** does not honour `skip_log` — writes the file unconditionally.
Already captured as A.1 in the kernel improvement plan.

Called once, from
[`terminate()` at kernel.py L493](../../abides-core/abides_core/kernel.py#L493).

### 3.4 Who reads it

**Nobody.** Verified by grep:
- No code path opens `summary_log.bz2`.
- No `pd.read_pickle(.*summary)` anywhere.
- No tool, no notebook, no test, no documentation page tells the user
  how to use it.
- The file *is* listed in
  [parallel-simulation.md L315](../ai/parallel-simulation.md#L315)
  as part of the on-disk layout, with the description "Kernel summary
  (agent types, final values)" — but no consumer is documented or
  implemented.

### 3.5 Verdict on system C

**Vestigial.** It collects a small subset of system B's data into a
parallel structure, writes it to a file that no internal tool reads,
and survives only because nobody removed it.

If a future feature wants "fast final-state summary", the right path is
to compute it from the (already in-memory) per-agent logs, or to use the
new `report_metric` mechanism planned in B.4 of the kernel improvement
plan. There is no current consumer to break.

---

## 4. Filesystem layout

A run produces a directory `./log/<log_dir>/` containing:

```
./log/<log_dir>/
├── summary_log.bz2                    # System C — written, never read
├── ExchangeAgent0.bz2                 # System B — per-agent log (DataFrame)
├── NoiseAgent1.bz2
├── ValueAgent2.bz2
├── ...                                # one .bz2 per agent that has log_to_file=True
└── fundamental_<symbol>.bz2           # ad-hoc artifacts via custom filename
```

`<log_dir>` defaults to `str(int(wall_clock_seconds))`
([kernel.py L136](../../abides-core/abides_core/kernel.py#L136)). This
**collides** if two simulations start in the same second. The high-level
`run_simulation()` wrapper avoids this by generating a UUID when
`log_dir is None`; the low-level `Kernel(...)` and CLI do not.

The path root `./log/` is hardcoded
([kernel.py L743, 776](../../abides-core/abides_core/kernel.py#L743)).
There is no `log_root` parameter; the kernel writes into the current
working directory.

There is no `simulation.log` for stdout — system A goes only to stdout.

---

## 5. Configuration knobs (every flag, one table)

| Flag | Where defined | Default | Controls | System |
|---|---|---|---|---|
| `Kernel.skip_log` | [kernel.py L54, 133](../../abides-core/abides_core/kernel.py#L54) | `True` | Suppress disk writes for B + C (but see bug 3.3 — C ignores it today) | B + C |
| `Kernel.log_dir` | [kernel.py L56, 136](../../abides-core/abides_core/kernel.py#L56) | `str(unix_seconds)` | Subdirectory under `./log/` | B + C |
| `Kernel.show_trace_messages` | [kernel.py L202](../../abides-core/abides_core/kernel.py#L202) | `False` | Hot-loop DEBUG traces | A |
| `Agent.log_events` | [agent.py L33](../../abides-core/abides_core/agent.py#L33) | `True` | Whether `logEvent()` records anything in memory | B |
| `Agent.log_to_file` | [agent.py L34](../../abides-core/abides_core/agent.py#L34) | `True` | Whether the agent's log is written at termination | B |
| `SimulationConfig.simulation.log_level` | [config_system/models.py L404](../../abides-markets/abides_markets/config_system/models.py#L404) | `"INFO"` | `basicConfig(level=...)` for stdout | A |

Notable: the user-facing config system exposes `log_level` (system A)
and `log_orders` overrides per agent (system B), but **does not expose**
`skip_log`, `log_dir`, or `show_trace_messages`. Those are reachable
only by passing them to `Kernel(...)` directly or by post-construction
mutation.

---

## 6. Tests

Tests confirm the load-bearing behaviour but reveal the asymmetry:

- **System A:** no tests. `basicConfig` is fire-and-forget.
- **System B:** rich coverage —
  [test_pandas_integration.py L186-413](../../abides-markets/tests/test_pandas_integration.py#L186)
  covers `parse_logs_df`, end-to-end disk round-trip, type coercion;
  [test_simulation.py L553-572](../../abides-markets/tests/test_simulation.py#L553)
  covers `SimulationResult.logs` shape and presence per profile;
  [test_replace_order_regression.py L344-388](../../abides-markets/tests/test_replace_order_regression.py#L344)
  covers REPLACE/MODIFY/CANCEL log records; config-system tests at
  [test_config_system.py L1169, 1179, 1437](../../abides-markets/tests/test_config_system.py#L1169)
  exercise `log_level()` and `log_orders` overrides.
- **System C:** no tests. No reader, no round-trip check, nothing
  asserts on `summary_log.bz2`.

Most kernel tests construct with `skip_log=True` to avoid touching the
filesystem ([test_kernel.py L47, 54, 63, 83](../../abides-core/tests/test_kernel.py#L47)).

---

## 7. Issues, smells, and risks (consolidated)

### 7.1 Real correctness bugs

- **`write_summary_log()` ignores `skip_log`**
  ([kernel.py L774-783](../../abides-core/abides_core/kernel.py#L774)).
  Fix queued as A.1 in the kernel improvement plan. Severity: low
  (file is unused), but a unit-test surprise.

### 7.2 Real performance issues

- **`parse_logs_df` is O(N²)** because it concatenates one DataFrame
  per row instead of building once. Hot path for any analysis on a
  large simulation. Easy fix: build a list of dicts, then one
  `pd.DataFrame(rows)`. (Out of scope for the kernel plan, but a
  separate quick win.)
- **`to_pickle(compression="bz2")`** is the slowest pickle path.
  Acceptable for a one-shot serialization but compounds when many
  agents log a lot.

### 7.3 Architectural smells

- **Kernel owns filesystem.** Path construction, format choice,
  directory creation, error handling are all inside
  `kernel.write_log` / `kernel.write_summary_log`. No abstraction.
  Kernel improvement plan's D.1 (LogWriter Protocol) addresses this.
- **Hardcoded `./log/`** — the kernel writes into the CWD, no
  `log_root` parameter. Two kernels in the same process clash; running
  from a different directory silently changes the destination.
  Addressed as A.11 / D.1 in the kernel plan.
- **Three logging systems share one `./log/<run_id>/` directory** with
  different lifecycles and consumers. Not separated, not labelled in
  the directory layout. A user looking at the folder cannot tell what
  is what without reading source.
- **`summary_log` is dead code with an externally visible artifact.**
  Kept for one release for safety, but nobody uses it. Either remove
  it or design it properly.
- **`show_trace_messages` is invisible.** No config path. Discoverable
  only by reading kernel source.
- **`log_events` / `log_to_file` are per-instance, not per-type.**
  Setting them across "all noise agents" requires loop-and-mutate at
  build time.

### 7.4 Robustness

- **No error handling around log writes.** Disk full, permission
  denied, pickle failure → kernel crash at the very end of a run, after
  hours of simulation. No try/except, no temp-file-then-rename.
- **No incremental flushing.** Memory usage grows linearly with event
  count. Not observed as a problem today (most sims < 1M events) but a
  hard ceiling for ABIDES-gym training that runs many episodes.
- **Pickle deserialization is untrusted-input-unsafe.** Loading a
  `.bz2` from a third party can execute arbitrary code. ABIDES does
  not advertise the file as portable, but users do share them.

### 7.5 What is *not* a problem

- Standard Python logging is well-behaved. Lazy formatting in the hot
  loop is the only fix needed (already in PR 3 of the kernel plan).
- The `Agent.logEvent` API is good. Cheap, simple, lossless.
- `parse_logs_df` is the right shape (DataFrame), just implemented
  poorly.
- Test coverage of system B is solid.

---

## 8. The mental model someone should leave with

> **System A** is operator output. It tells you the simulation is alive
> and roughly how fast. It is not a record of the simulation.
>
> **System B** is the simulation's record. Every agent appends to its
> own list at every event; at the end, lists become DataFrames and (if
> not skipped) get pickled to disk. `parse_logs_df` is the official
> reader.
>
> **System C** is dead weight that produces a file nobody reads. It is
> retained only because removing it would break the documented disk
> layout.
>
> The kernel currently entangles all three: it owns the format, the
> filesystem path, the lifecycle, and a financial-summary leak from the
> markets layer. The kernel improvement plan's PR 4 + PR 7 break this
> entanglement.

---

## 9. Open questions — what a redesign would have to decide

These are *not* recommendations. They are the choices a redesign cannot
avoid:

1. **Should `summary_log` survive?** Remove (free), keep (needs a
   reader and a purpose), or replace with the new `report_metric()`
   mechanism (B.4 in the kernel plan)?
2. **Format pluggability.** Hardcode bz2-pickle forever, or expose a
   `LogWriter` Protocol (parquet, JSONL, sqlite)? The kernel plan's
   D.1 provides the seam; the *choice* of formats is open.
3. **Incremental writes.** Do long simulations need streaming flush, or
   is "all at terminate" forever good enough?
4. **`./log/` root.** Make it a `Kernel(log_root=...)` parameter, or
   keep CWD-relative and accept the surprise?
5. **`log_dir` collision.** Make UUID the default in `Kernel` itself
   (today only `run_simulation()` does this), or keep the wall-clock
   default for backwards compatibility?
6. **Trace flag exposure.** Promote `show_trace_messages` to a
   `SimulationConfig` field, or leave it as a programmatic-only escape
   hatch?
7. **Per-instance vs per-type log flags.** Add a config-system
   convenience for "disable order logs for this agent type globally",
   or accept the loop-and-mutate idiom?
8. **Standard logging to file.** Should ABIDES attach a `FileHandler`
   that writes `simulation.log` next to the per-agent files, or keep
   stdout-only and let users add it themselves?
9. **Pickle vs portable format.** Is the `.bz2` file an internal cache
   (pickle is fine) or an interchange artifact (parquet/arrow is
   safer)?
10. **Error handling on write.** Best-effort write with warning, or
    fail-fast and crash the run?

These are decisions for a separate plan. This document is the picture,
not the prescription.

---

## A. Original intent of `summary_log` (archaeology)

`summary_log` is **not new**. It was inherited verbatim from the
upstream JPMorgan ABIDES public release (commit `3abbd6f` — "ABIDES
public commit") and has not been touched since. The upstream
`Kernel.py` carries the answer in two comments that did not survive the
fork's reformatting.

### A.1 The mission statement (upstream `Kernel.__init__`)

```python
# The Kernel maintains a summary log to which agents can write
# information that should be centralized for very fast access
# by separate statistical summary programs.  Detailed event
# logging should go only to the agent's individual log.  This
# is for things like "final position value" and such.
self.summary_log: List[Dict[str, Any]] = []
```

### A.2 The contract (upstream `append_summary_log` docstring)

```
We don't even include a timestamp, because this log is for
one-time-only summary reporting, like starting cash, or ending cash.

Arguments:
    sender_id: The ID of the agent making the call.
    event_type: The type of the event.
    event:      The event to append to the log.
```

### A.3 What this tells us

The original design carved out a **deliberate two-tier logging split**:

| Tier | Per-agent `agent.log` | Central `summary_log` |
|---|---|---|
| **Granularity** | Every event, with timestamp | One-shot final-state events, no timestamp |
| **Audience** | Per-run diagnostics, replay, microstructure analysis | "Separate statistical summary programs" (i.e. cross-run batch analytics) |
| **Cost** | One file per agent per run | One small file per run |
| **Why centralized** | N/A | A batch tool can `pd.read_pickle` one file per run instead of N agent files, and get a denormalized, ready-to-aggregate table of "everyone's bottom line" |

The intent makes sense for a research workflow that runs many parallel
simulations from a shell script (which is how upstream ABIDES was
operated — `summary_log.bz2` was the **cross-simulation roll-up
artifact**).

### A.4 Why no readers exist in this fork

Three plausible explanations, in decreasing order of likelihood:

1. **The "separate statistical summary programs" were never released.**
   The upstream public repo ships the producer half of the contract
   without the consumer half. The aggregation tool likely existed
   inside JPMorgan and was not open-sourced. The fork inherited an
   API with no reachable consumer.
2. **`parse_logs_df` superseded it in practice.** Once the high-level
   `run_simulation()` wrapper landed and returned a parsed DataFrame
   in memory, downstream code in this fork (notebooks, metrics,
   `SimulationResult`) standardized on **the per-agent path**.
   `summary_log.bz2` became redundant for in-process consumers and
   nobody built the cross-run consumer.
3. **Schema friction.** The summary record has no timestamp and no
   uniform schema for `event` (each event type stuffs a different
   dict shape in there). Even an external tool would need
   per-event-type unpacking logic — at which point reading the
   per-agent logs gives strictly more information for similar effort.

### A.5 Implications for the redesign

This reframes the question from "is this dead code?" to "**is the
two-tier split still the right design?**":

- **The need is real.** Cross-run aggregation ("across these 200 sims,
  what is the distribution of final NoiseAgent valuations?") is a
  legitimate and recurring use case for ABIDES. The kernel improvement
  plan's `report_metric()` mechanism (B.4) is one answer, but it
  aggregates *within* a run — not across runs.
- **The current artifact does not serve it.** No reader, no schema,
  no documented format.
- **Three coherent futures:**
  1. **Remove** `summary_log` and rely on per-agent logs +
     `parse_logs_df` for everything. Cross-run aggregation becomes
     "load N pickle files, concat, group". Simple, slower at scale.
  2. **Repurpose** `summary_log` as the on-disk projection of
     `report_metric()`'s aggregated results. Same artifact name,
     well-defined schema (`agent_type`, `key`, `sum`, `count`, `mean`),
     ready for cross-run batch tools.
  3. **Keep as-is, document as legacy**, and add a `WARNING` log line
     when the file is written so users know it exists.

Option 2 is the most honest: it preserves the original two-tier intent,
gives the artifact a real schema, and reuses the `report_metric()`
infrastructure already planned. It would be a follow-up plan, not part
of the current kernel refactor.

The choice is out of scope for this analysis. What is in scope is the
correction: **`summary_log` is not orphaned because it was
ill-conceived. It is orphaned because the consumer half of the original
contract was never open-sourced.**
